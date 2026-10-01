"""Compare frozen baselines, Phase 2 retrieval, and frozen Phase 3 reranking/fusion."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crossid_learning.common import (ROOT, digest, fresh_directory, load_embedding_view, load_protocol,
                                     protect_evaluation, provenance, read_documents,
                                     require_matching_protocol, text_hash, write_json)
from crossid_learning.fusion import apply_fusion
from crossid_learning.models import encode_matrix, load_adapter
from crossid_learning.pipeline import load_reranker, validate_fusion
from crossid_learning.reranking import build_candidates, candidate_features, cross_encoder_logits
from crossid_learning.retrieval import (RetrievalIndex, descending_order, metrics,
                                       prediction_frame, reranked_order, top_mean)


def frozen_scores(refs, raw_reference, raw_target, authors):
    groups = [np.flatnonzero(refs.author.to_numpy() == a) for a in authors]
    centroids = np.stack([raw_reference[ids].mean(axis=0) for ids in groups])
    if (np.linalg.norm(centroids, axis=1) == 0).any():
        raise ValueError('SELMA reference mean is zero.')
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    targets = raw_target / np.linalg.norm(raw_target, axis=1, keepdims=True)
    selma = -np.round(1-targets @ centroids.T, 4)
    # Match Phase 1's individually normalized float32 references and targets.
    reference = raw_reference.astype(np.float32)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    target32 = raw_target.astype(np.float32)
    target32 /= np.linalg.norm(target32, axis=1, keepdims=True)
    parts = []
    for start in range(0, len(target32), 64):
        z = target32[start:start+64]
        parts.append(np.stack([top_mean(z @ reference[ids].T, 4) for ids in groups], axis=1))
    return selma, np.concatenate(parts)


def benchmark(args):
    if min(args.top_candidates, args.references_per_candidate, args.batch_size) < 1:
        raise ValueError('Candidate/reference/batch budgets must be positive.')
    if args.fusion and not args.phase3:
        raise ValueError('Fusion requires a Phase 3 model.')
    adapter, phase2_meta = load_adapter(args.phase2, args.device)
    protocol_path = args.data_dir.parent / 'learning_protocol.json'
    if protocol_path.is_file():
        protocol, protocol_sha = load_protocol(args.data_dir.parent)
        require_matching_protocol(phase2_meta, protocol_sha)
        if args.data_dir.name != 'test':
            raise ValueError('The final benchmark accepts only the silver test role; dev/cal are reserved.')
    target_path = args.target_file or (args.data_dir / 'targets.csv' if
                                      (args.data_dir / 'targets.csv').exists() else
                                      args.data_dir / 'test/CrossNews.csv')
    targets = read_documents(target_path)
    fusion = json.loads(args.fusion.read_text()) if args.fusion else None
    model = tokenizer = phase3_meta = None
    protected = dict(phase2_meta)
    if args.phase3:
        model, tokenizer, phase3_meta = load_reranker(args.phase3, args.phase2, device=args.device)
        protected['calibration_authors'] = (fusion['metadata']['calibration_authors'] if fusion else [])
    refs_by_condition = {}
    for condition in args.reference_sets:
        refs = read_documents(args.data_dir / 'query' / f'CrossNews_{condition}.csv')
        protect_evaluation(refs, targets, protected, args.allow_reference_text_overlap)
        if fusion:
            routing = {'reference_set': condition, 'top_candidates': args.top_candidates,
                       'references_per_candidate': args.references_per_candidate,
                       'max_length': phase3_meta['max_length'], 'max_chars_per_text': 2000}
            validate_fusion(fusion, args.phase2, args.phase3, routing)
        # Both views are validated independently, since training IDs can appear
        # in both files with intentionally different instruction prompts.
        reference = load_embedding_view(args.reference_embeddings, refs, dtype=np.float64)
        target = load_embedding_view(args.target_embeddings, targets, dtype=np.float64)
        if reference.shape[1] != adapter.config['input_dim'] or target.shape[1] != adapter.config['input_dim']:
            raise ValueError('Embedding view dimension differs from the trained adapter.')
        refs_by_condition[condition] = (refs, reference)
    if args.check:
        print(json.dumps({'status': 'preflight_passed', 'targets': len(targets),
                          'authors': targets.author.nunique(), 'references': args.reference_sets}, indent=2)); return
    output = fresh_directory(args.output); start = time.perf_counter()
    input_paths = [args.phase2, target_path, *args.reference_embeddings, *args.target_embeddings]
    input_paths += [args.data_dir / 'query' / f'CrossNews_{c}.csv' for c in args.reference_sets]
    if args.fusion:
        input_paths.append(args.fusion)
    manifest = {'status': 'running', 'scope': 'closed-world Phase 2 adapter / Phase 3 reranking',
                'inputs': {str(p.resolve()): digest(p) for p in input_paths},
                'phase3_model_digest': phase3_meta['model_digest'] if phase3_meta else None,
                'test_author_count': int(targets.author.nunique()), 'test_document_count': len(targets),
                'top_candidates': args.top_candidates, 'references_per_candidate': args.references_per_candidate,
                'selection': 'Frozen checkpoints/fusion; test labels used only for metrics',
                'tie_policy': 'descending scores, ascending lexicographic author index',
                'baseline': 'SELMA cosine of raw float64 means rounded to 4 decimals; '
                            'frozen reference_only mean of top 4 float32 reference similarities',
                'note': 'Gold has informed previous exploratory designs; silver and gold candidate counts differ.',
                'allow_reference_text_overlap': args.allow_reference_text_overlap,
                **provenance()}
    write_json(output / 'manifest.json', manifest)
    summary = []
    try:
        for condition, (refs, reference) in refs_by_condition.items():
            duplicate_mask = targets.text.map(text_hash).isin(set(refs.text.map(text_hash))).to_numpy()
            shared_texts = set(refs.text.map(text_hash)) & set(targets.text.map(text_hash))
            manifest.setdefault('reference_target_text_overlap', {})[condition] = {
                'unique_shared_texts': len(shared_texts), 'affected_targets': int(duplicate_mask.sum())}
            index = RetrievalIndex(adapter, refs, reference, args.device)
            z = encode_matrix(adapter, target, args.device)
            scores, local, centroid = index.score(z)
            selma, frozen = frozen_scores(refs, reference, target, index.authors)
            outputs = {'selma': descending_order(selma), 'frozen_reference_only': descending_order(frozen),
                       'phase2_reference_only': descending_order(local), 'phase2_learned_profiles': descending_order(scores)}
            if model is not None:
                data = build_candidates(index, targets, z, scores, local, centroid,
                                        args.top_candidates, args.references_per_candidate)
                features = candidate_features(data, cross_encoder_logits(
                    model, tokenizer, data.pairs, args.device, args.batch_size, phase3_meta['max_length']))
                outputs['phase3_cross_encoder'] = reranked_order(scores, data.candidates, features[..., 3])
                if fusion:
                    outputs['phase3_fusion'] = reranked_order(scores, data.candidates, apply_fusion(features, fusion))
            else:
                candidates = descending_order(scores)[:, :min(args.top_candidates, len(index.authors))]
            shortlist = data.candidates if model is not None else candidates
            mapping = {a: i for i, a in enumerate(index.authors)}
            hit = np.asarray([mapping[a] in chosen for a, chosen in zip(targets.author, shortlist)])
            for name, order in outputs.items():
                folder = output / condition / name; folder.mkdir(parents=True)
                predictions = prediction_frame(targets, index.authors, order)
                predictions.to_csv(folder / 'prediction_ranks.csv', index=False)
                result = metrics(predictions); write_json(folder / 'metrics.json', result)
                write_json(folder / 'metrics_without_reference_text_duplicates.json', metrics(predictions[~duplicate_mask]))
                for genre, values in result.items():
                    mask = np.ones(len(targets), dtype=bool) if genre == 'Overall' else targets.genre.to_numpy() == genre
                    recall = float(hit[mask].mean()) if mask.any() else None
                    summary.append({'model': name, 'references': condition, 'targets': genre,
                                    'phase2_candidate_recall': recall,
                                    'reference_text_duplicate_targets': int(duplicate_mask[mask].sum()), **values})
                print(f'{condition}/{name}: accuracy={result["Overall"]["Accuracy"]:.4f}, '
                      f'MRR={result["Overall"]["MRR"]:.4f}', flush=True)
            manifest.setdefault('candidate_counts', {})[condition] = int(shortlist.shape[1])
        write_json(output / 'summary.json', summary)
        pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
        manifest['status'] = 'completed'; manifest['runtime_seconds'] = time.perf_counter()-start
    except Exception as exc:
        manifest.update(status='failed', error=str(exc)); raise
    finally:
        write_json(output / 'manifest.json', manifest)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'attribution_data')
    p.add_argument('--target-file', type=Path)
    p.add_argument('--phase2', type=Path, required=True)
    p.add_argument('--phase3', type=Path)
    p.add_argument('--fusion', type=Path)
    p.add_argument('--reference-embeddings', type=Path, nargs='+', required=True)
    p.add_argument('--target-embeddings', type=Path, nargs='+', required=True)
    p.add_argument('--reference-sets', nargs='+', choices=['Article', 'Tweet', 'Both'], default=['Both'])
    p.add_argument('--top-candidates', type=int, default=20)
    p.add_argument('--references-per-candidate', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, default=ROOT / 'results/crossid_phase23_benchmark')
    p.add_argument('--check', action='store_true')
    p.add_argument('--allow-reference-text-overlap', action='store_true',
                   help='Preserve an existing benchmark split with duplicate text; record overlaps and clean-subset metrics. ID overlap remains prohibited.')
    return p


if __name__ == '__main__':
    p = parser()
    try:
        benchmark(p.parse_args())
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
