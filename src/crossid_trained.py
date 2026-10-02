"""Train and benchmark experimental CROSS-ID Phases 2 and 3."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from benchmark_crossid import digest, read_split
from crossid_common import (cohort, disjoint, fresh, identity, load_vectors, metrics,
                            ranking, raw_matrix, source_hashes, split_check, write_json)
from crossid_phase2 import load_adapter, retrieve
from crossid_phase3 import (directory_digest, ensure_heldout, features, load_fusion,
                            load_reranker, reorder)


def baseline_scores(refs, rx, tx, local=False):
    authors = sorted(refs.author.unique())
    r = rx / np.linalg.norm(rx, axis=1, keepdims=True)
    t = tx / np.linalg.norm(tx, axis=1, keepdims=True)
    scores = np.empty((len(tx), len(authors)), dtype='float32')
    for j, author in enumerate(authors):
        mask = refs.author.to_numpy() == author
        if local:
            sims = t @ r[mask].T
            scores[:, j] = np.sort(sims, axis=1)[:, -min(4, sims.shape[1]):].mean(1)
        else:
            centroid = rx[mask].mean(0)
            norm = np.linalg.norm(centroid)
            if norm == 0:
                raise ValueError('SELMA author centroid has zero norm.')
            cosine = np.clip(t @ centroid / norm, -1., 1.)
            scores[:, j] = np.array([-round(float(1 - value), 4) for value in cosine])
    return scores, authors


def records(targets, authors, orders, shortlists=None):
    rows = []
    for i, (target, order) in enumerate(zip(targets.itertuples(), orders)):
        label = authors.index(target.author)
        row = {'id': target.id, 'genre': target.genre, 'label': target.author,
               'prediction': authors[int(order[0])], 'rank': list(order).index(label) + 1}
        if shortlists is not None:
            row['shortlist_hit'] = label in shortlists[i]
        rows.append(row)
    return rows


def evaluate(args):
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive.')
    if bool(args.reranker) != bool(args.fusion):
        raise ValueError('Phase 3 evaluation requires both --reranker and --fusion.')
    if bool(args.gold_query) != bool(args.gold_target):
        raise ValueError('Provide both gold query and target CSVs.')
    if args.gold_query:
        refs, targets = read_split(args.gold_query), read_split(args.gold_target)
        overlap = split_check(refs, targets, args.allow_text_overlap)
        input_paths = [args.gold_query, args.gold_target]
        role = 'exploratory_gold_test'
    else:
        if args.allow_text_overlap:
            raise ValueError('--allow-text-overlap is permitted only for the original gold protocol.')
        refs, targets, _ = cohort(args.data_dir, 'test', args.reference_set)
        overlap = []
        input_paths = [args.data_dir / 'protocol.json',
                       args.data_dir / f'test/query/CrossNews_{args.reference_set}.csv',
                       args.data_dir / 'test/test/CrossNews.csv']
        role = 'silver_test'
    expected_genres = {'Article', 'Tweet'} if args.reference_set == 'Both' else {args.reference_set}
    if set(refs.genre) != expected_genres:
        raise ValueError('Reference genres differ from the selected condition.')
    adapter, adapter_meta = load_adapter(args.adapter, args.device)
    evaluation_identity = ensure_heldout(adapter_meta, refs, targets)
    rx, tx = load_vectors(args.train_embeddings, args.target_embeddings, refs, targets)
    protected = {str(args.adapter): directory_digest(args.adapter)}
    if args.reranker:
        encoder, reranker_meta = load_reranker(args.reranker, args.adapter, args.device)
        ensure_heldout(reranker_meta, refs, targets)
        fusion = load_fusion(args.fusion, args.adapter, args.reranker, args.reference_set)
        disjoint(fusion['calibration_identity'], evaluation_identity)
        protected[str(args.reranker)] = directory_digest(args.reranker)
        protected[str(args.fusion)] = digest(args.fusion)
        input_paths.extend([args.fusion])
    output = fresh(args.output)
    manifest = {'status': 'running', 'role': role, 'reference_set': args.reference_set,
                'authors': len(set(refs.author)), 'targets': len(targets), 'references': len(refs),
                'source_sha256': source_hashes(), 'checkpoint_sha256': protected,
                'inputs': {str(Path(p).resolve()): digest(Path(p)) for p in
                           [*input_paths, args.train_embeddings, args.target_embeddings]},
                'reference_target_text_overlap': {'count': len(overlap), 'hashes': overlap,
                                                 'explicitly_allowed': args.allow_text_overlap},
                'selection': 'Frozen checkpoints and fusion; no fitting or configuration selection on test.',
                'tie_policy': 'Descending score, then lexicographically sorted author label.',
                'scope': 'Adapter training over frozen SELMA; no backbone fine-tuning.',
                'versions': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'pandas', 'scikit-learn')}}
    try:
        manifest['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        manifest['working_tree_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        manifest['commit'] = None
    write_json(output / 'manifest.json', manifest)
    start = time.perf_counter()
    try:
        runs = {}
        raw_rx = raw_matrix(args.train_embeddings, refs, rx.shape[1])
        raw_tx = raw_matrix(args.target_embeddings, targets, tx.shape[1])
        scores, authors = baseline_scores(refs, raw_rx, raw_tx, False)
        runs['selma'] = records(targets, authors, [ranking(s) for s in scores])
        del raw_rx, raw_tx
        scores, authors = baseline_scores(refs, rx, tx, True)
        runs['reference_only'] = records(targets, authors, [ranking(s) for s in scores])
        adapter_ref, authors, _, _ = retrieve(adapter, refs, rx, tx, args.device, prototype_weight=0.)
        runs['adapter_reference_only'] = records(targets, authors, [ranking(s) for s in adapter_ref])
        scores, authors, bundles, z = retrieve(adapter, refs, rx, tx, args.device)
        orders = [ranking(s) for s in scores]
        runs['adapter_prototypes'] = records(targets, authors, orders)
        if args.reranker:
            rows, x = features(encoder, reranker_meta, refs, targets, scores, authors, bundles, z, args.batch_size)
            fused = ((x - fusion['mean']) / fusion['scale']) @ np.asarray(fusion['coefficient']) + fusion['intercept']
            candidates, ce_values, fusion_values = [[] for _ in targets.id], [[] for _ in targets.id], [[] for _ in targets.id]
            for (i, j, _), values, score in zip(rows, x, fused):
                candidates[i].append(j)
                ce_values[i].append(values[1])
                fusion_values[i].append(float(score))
            ce_orders = [reorder(o, c, s) for o, c, s in zip(orders, candidates, ce_values)]
            fusion_orders = [reorder(o, c, s) for o, c, s in zip(orders, candidates, fusion_values)]
            runs['cross_encoder'] = records(targets, authors, ce_orders, candidates)
            runs['fused'] = records(targets, authors, fusion_orders, candidates)
            manifest['encoder'] = reranker_meta['encoder']
            manifest['shortlist'] = reranker_meta['shortlist']
        summary = {name: metrics(rows) for name, rows in runs.items()}
        for name, rows in runs.items():
            with (output / f'{name}_prediction_ranks.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write_json(output / 'summary.json', summary)
        for path, value in protected.items():
            actual = directory_digest(path) if Path(path).is_dir() else digest(Path(path))
            if actual != value:
                raise ValueError('Inference changed a checkpoint.')
        manifest.update(status='completed', elapsed_seconds=time.perf_counter() - start)
        write_json(output / 'manifest.json', manifest)
        print(json.dumps(summary, indent=2))
        return summary
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'manifest.json', manifest)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    for command in ('train-adapter', 'train-reranker', 'calibrate', 'evaluate'):
        s = subs.add_parser(command)
        s.add_argument('--data-dir', type=Path, default=Path('crossid_training_data'))
        s.add_argument('--train-embeddings', type=Path, required=True)
        s.add_argument('--target-embeddings', type=Path, required=True)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--device', default='cpu')
        if command.startswith('train'):
            s.add_argument('--dev-train-embeddings', type=Path, required=True)
            s.add_argument('--dev-target-embeddings', type=Path, required=True)
            s.add_argument('--epochs', type=int, default=5 if command == 'train-adapter' else 3)
            s.add_argument('--learning-rate', type=float, default=1e-3 if command == 'train-adapter' else 2e-5)
            s.add_argument('--seed', type=int, default=20261001)
        if command != 'train-adapter':
            s.add_argument('--adapter', type=Path, required=True)
            s.add_argument('--batch-size', type=int, default=8)
        if command in ('calibrate', 'evaluate'):
            s.add_argument('--reference-set', choices=['Article', 'Tweet', 'Both'], default='Both')
            s.add_argument('--reranker', type=Path, required=command == 'calibrate')
        if command == 'train-adapter':
            s.add_argument('--steps', type=int, default=200, help='Optimizer steps per epoch.')
            s.add_argument('--authors-per-batch', type=int, default=8)
            s.add_argument('--max-bundle', type=int, default=8, help='References sampled per genre per author.')
            s.add_argument('--topics', type=int, default=16)
            s.add_argument('--output-dim', type=int, default=256)
            s.add_argument('--hidden-dim', type=int, default=512)
            s.add_argument('--prototypes', type=int, default=4)
            s.add_argument('--adversary-strength', type=float, default=.1)
        if command == 'train-reranker':
            s.add_argument('--encoder', default='distilbert/distilbert-base-uncased')
            s.add_argument('--revision', help='Pin a Hugging Face model revision for reproducibility.')
            s.add_argument('--max-length', type=int, default=512)
            s.add_argument('--shortlist', type=int, default=32)
            s.add_argument('--reference-count', type=int, default=2)
            s.add_argument('--negatives', type=int, default=3)
        if command == 'evaluate':
            s.add_argument('--fusion', type=Path)
            s.add_argument('--gold-query', type=Path)
            s.add_argument('--gold-target', type=Path)
            s.add_argument('--allow-text-overlap', action='store_true')
    return p


if __name__ == '__main__':
    p = parser()
    args = p.parse_args()
    try:
        if args.command == 'train-adapter':
            from crossid_phase2 import train
            train(args)
        elif args.command == 'train-reranker':
            from crossid_phase3 import train
            train(args)
        elif args.command == 'calibrate':
            from crossid_phase3 import calibrate
            calibrate(args)
        else:
            evaluate(args)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
