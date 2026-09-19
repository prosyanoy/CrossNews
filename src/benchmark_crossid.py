"""Benchmark Phase 1 CROSS-ID and upstream SELMA using identical embeddings."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from attribution_models.crossid import CrossID
from attribution_models.selma import SELMA
from utils import evaluate_attribution_scores

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'attribution_data')
    p.add_argument('--train-embeddings', type=Path, default=ROOT / 'selma_embeddings/mistral/train.json')
    p.add_argument('--test-embeddings', type=Path, default=ROOT / 'selma_embeddings/mistral/test_prompt_taskonly.json')
    p.add_argument('--parameters', type=Path, default=ROOT / 'src/model_parameters/crossid.json')
    p.add_argument('--parameter-sets', nargs='+', default=['default', 'prototype_heavy'])
    p.add_argument('--reference-sets', nargs='+', choices=['Article', 'Tweet', 'Both'], default=['Article', 'Tweet', 'Both'])
    p.add_argument('--output', type=Path, default=ROOT / 'results/crossid_benchmark')
    p.add_argument('--check', action='store_true', help='Validate inputs without scoring or writing results.')
    return p


def read_split(path):
    frame = pd.read_csv(path, dtype={'id': str, 'author': str})
    required = {'id', 'author', 'genre'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f'{path}: expected nonempty id, author, genre columns.')
    if frame[list(required)].isna().any().any() or frame['id'].duplicated().any():
        raise ValueError(f'{path}: null metadata or duplicate document IDs.')
    if not set(frame['genre']).issubset({'Article', 'Tweet'}):
        raise ValueError(f'{path}: unrecognized genre.')
    return frame


def preflight(args):
    queries = {genre: args.data_dir / 'query' / f'CrossNews_{genre}.csv'
               for genre in args.reference_sets}
    target_path = args.data_dir / 'test/CrossNews.csv'
    paths = [args.parameters, args.train_embeddings, args.test_embeddings, target_path, *queries.values()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError('Missing benchmark inputs:\n  ' + '\n  '.join(missing))
    configs = json.loads(args.parameters.read_text())
    for name in args.parameter_sets:
        if name not in configs:
            raise ValueError(f'Unknown parameter set: {name}')
    # Reuse the model loader for finite, dimension, and duplicate-ID validation.
    embeddings = CrossID._load_embeddings(args.train_embeddings, args.test_embeddings)
    train_ids = set(json.loads(args.train_embeddings.read_text()))
    test_ids = set(json.loads(args.test_embeddings.read_text()))
    target = read_split(target_path)
    if not set(target['id']).issubset(test_ids):
        raise ValueError('Target IDs are missing from the test embedding file.')
    counts = {}
    for genre, path in queries.items():
        query = read_split(path)
        if set(query['author']) != set(target['author']):
            raise ValueError(f'{genre}: query and target author sets must match.')
        if set(query['id']) & set(target['id']):
            raise ValueError(f'{genre}: reference/target document overlap.')
        if not set(query['id']).issubset(train_ids):
            raise ValueError(f'{genre}: reference IDs are missing from the train embedding file.')
        if genre != 'Both' and set(query['genre']) != {genre}:
            raise ValueError(f'{genre}: unexpected reference genre.')
        counts[genre] = len(query)
    return configs, queries, target_path, paths, {
        'authors': int(target['author'].nunique()), 'targets': len(target),
        'targets_by_genre': target['genre'].value_counts().to_dict(),
        'references': counts, 'embedding_dimensions': len(next(iter(embeddings.values()))),
    }


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    configs, queries, target_path, inputs, counts = preflight(args)
    print(json.dumps({'preflight': 'passed', **counts}, indent=2))
    if args.check:
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Output directory is not empty: {args.output}; choose a new --output.')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {
        'status': 'running', 'scope': 'Phase 1, frozen embeddings, closed-world attribution',
        'commit': commit, 'working_tree_dirty': dirty, 'counts': counts,
        'inputs': {str(p.resolve()): digest(p) for p in inputs},
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [
            Path(__file__), ROOT / 'src/attribution_models/crossid.py',
            ROOT / 'src/attribution_models/selma.py', ROOT / 'src/attribution_models/attribution_model.py',
            ROOT / 'src/utils.py']},
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'scipy']},
        'tie_policy': 'descending score, ascending lexicographically mapped author ID',
        'selection': 'Fixed configurations reported independently; no test-set selection.',
        'baseline': 'Upstream SELMA: mean raw reference embeddings, negative cosine distance rounded to 4 decimals.',
        'parameters': {name: configs[name] for name in args.parameter_sets},
    }
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    rows = []
    try:
        for genre, query_path in queries.items():
            for name in ['selma', *args.parameter_sets]:
                params = dict(configs[name]) if name != 'selma' else {'name': 'selma'}
                params.update(train_embedding_loc=str(args.train_embeddings.resolve()),
                              test_embedding_loc=str(args.test_embeddings.resolve()))
                model_args = SimpleNamespace(query_file=str(query_path), target_file=str(target_path),
                                             train=True, test=True, load=False, silent=True,
                                             save_folder=str(args.output / name))
                start = time.perf_counter()
                model = (SELMA if name == 'selma' else CrossID)(model_args, params)
                model.train()
                predictions, authors = model.evaluate()
                elapsed = time.perf_counter() - start
                folder = Path(model.model_folder)
                (folder / 'predictions.json').write_text(json.dumps({'predictions': predictions, 'author_list': authors}))
                scores = {}
                for target_genre in ['Overall', 'Article', 'Tweet']:
                    subset = [p for p in predictions if target_genre == 'Overall' or p['genre'] == target_genre]
                    metrics = evaluate_attribution_scores(subset)
                    # Explicit count makes an absent genre distinguishable from zero accuracy.
                    metrics['n'] = len(subset)
                    if not subset:
                        metrics = {k: None for k in metrics if k != 'n'} | {'n': 0}
                    scores[target_genre] = metrics
                    rows.append(dict(model=name, references=genre, targets=target_genre,
                                     seconds=elapsed, **metrics))
                (folder / 'test_results.json').write_text(json.dumps(scores, indent=2))
                print(f'{genre} / {name}: Accuracy={scores["Overall"]["Accuracy"]:.4f}, {elapsed:.2f}s')
                del model
        (args.output / 'summary.json').write_text(json.dumps(rows, indent=2))
        with (args.output / 'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
        manifest['status'] = 'completed'
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
