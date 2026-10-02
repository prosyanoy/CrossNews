"""Shared integrity checks for the experimental trained CROSS-ID pipeline."""
import hashlib
import json
from pathlib import Path
import re

import ijson
import numpy as np
import pandas as pd

from attribution_models.crossid import CrossID
from benchmark_crossid import digest, read_split

STAGES = ('train', 'dev', 'calibration', 'test')


def text_hash(text):
    # Ignore casing and whitespace when rejecting repeated text.
    return hashlib.sha256(' '.join(str(text).casefold().split()).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def fresh(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Choose a fresh output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def identity(*frames):
    return {'authors': sorted(set().union(*(set(f.author) for f in frames))),
            'ids': sorted(set().union(*(set(f.id) for f in frames))),
            'texts': sorted(set().union(*(set(f.text.map(text_hash)) for f in frames)))}


def disjoint(left, right):
    for key in ('authors', 'ids', 'texts'):
        if set(left[key]) & set(right[key]):
            raise ValueError(f'Training/selection and held-out data overlap in {key}.')


def split_check(refs, targets, allow_text_overlap=False):
    if 'text' not in refs or 'text' not in targets:
        raise ValueError('Trained stages require text columns.')
    for frame in (refs, targets):
        if frame.text.isna().any() or (frame.text.str.strip().str.len() < 2).any():
            raise ValueError('Empty or invalid document text.')
    if set(refs.author) != set(targets.author):
        raise ValueError('Reference/target author sets differ.')
    if set(refs.id) & set(targets.id):
        raise ValueError('Reference/target IDs overlap.')
    overlap = set(refs.text.map(text_hash)) & set(targets.text.map(text_hash))
    if overlap and not allow_text_overlap:
        raise ValueError(f'{len(overlap)} repeated reference/target texts; evaluation would leak.')
    return sorted(overlap)


def cohort(root, stage, reference='Both'):
    root = Path(root)
    if stage not in STAGES:
        raise ValueError('Unknown cohort role.')
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol.get('format') != 'crossid_training_protocol_v1':
        raise ValueError('Expected prepared CROSS-ID training protocol.')
    assigned = set()
    for role in STAGES:
        labels = set(protocol['cohorts'][role])
        if len(labels) < 2 or assigned & labels or labels & set(protocol['excluded_gold_authors']):
            raise ValueError('Prepared cohorts must have distinct non-gold authors.')
        assigned.update(labels)
    refs = root / stage / 'query' / f'CrossNews_{reference}.csv'
    targets = root / stage / 'test' / 'CrossNews.csv'
    for path in (refs, targets):
        if protocol['files'].get(str(path.relative_to(root))) != digest(path):
            raise ValueError(f'Prepared split modified: {path}')
    ref, target = read_split(refs), read_split(targets)
    split_check(ref, target)
    if set(ref.author) != set(protocol['cohorts'][stage]):
        raise ValueError('Author set differs from prepared role.')
    return ref, target, protocol


def matrix(frame, embeddings):
    missing = set(frame.id) - set(embeddings)
    if missing:
        raise ValueError(f'Missing {len(missing)} embeddings, e.g. {sorted(missing)[:3]}')
    return np.stack([embeddings[i] for i in frame.id]).astype('float32')


def load_vectors(refs_path, targets_path, refs, targets):
    # Also checks vector dimensions, finite values and conflicting duplicate IDs.
    vectors = CrossID._load_embeddings(refs_path, targets_path)
    if not set(refs.id) <= set(json.loads(Path(refs_path).read_text())):
        raise ValueError('References must use unprompted reference embeddings.')
    if not set(targets.id) <= set(json.loads(Path(targets_path).read_text())):
        raise ValueError('Targets must use prompted target embeddings.')
    return matrix(refs, vectors), matrix(targets, vectors)


def raw_matrix(path, frame, dimensions):
    """Float64 source vectors for parity with upstream SELMA's JSON loader."""
    positions = {doc_id: i for i, doc_id in enumerate(frame.id)}
    out = np.empty((len(frame), dimensions), dtype='float64')
    remaining = set(positions)
    with Path(path).open('rb') as stream:
        for doc_id, vector in ijson.kvitems(stream, '', use_float=True):
            if doc_id in positions:
                out[positions[doc_id]] = vector
                remaining.discard(doc_id)
    if remaining:
        raise ValueError('Missing raw SELMA vectors.')
    return out


def ranking(scores):
    return np.argsort(-np.asarray(scores), kind='stable')


def metrics(records):
    result = {}
    for genre in ('Overall', 'Article', 'Tweet'):
        rows = [r for r in records if genre == 'Overall' or r['genre'] == genre]
        if not rows:
            continue
        ranks = np.array([r['rank'] for r in rows])
        result[genre] = {'n': len(rows), 'accuracy': float(np.mean(ranks == 1)),
                         'mrr': float(np.mean(1 / ranks)),
                         **{f'recall@{k}': float(np.mean(ranks <= k)) for k in (8, 16, 32, 64)}}
        if 'shortlist_hit' in rows[0]:
            result[genre]['shortlist_recall'] = float(np.mean([r['shortlist_hit'] for r in rows]))
    return result


STYLE_NAMES = ['log_chars', 'log_words', 'mean_word_chars', 'word_chars_std',
               'uppercase', 'digits', 'spaces', 'newlines', 'lexical_diversity',
               *[f'punct_{i}' for i in range(12)]]


def stylometry(text):
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    lengths = np.array([len(w) for w in words])
    n = max(1, len(text))
    return np.array([np.log1p(len(text)), np.log1p(len(words)),
                     float(lengths.mean()) if len(words) else 0.,
                     float(lengths.std()) if len(words) else 0.,
                     sum(c.isupper() for c in text) / n,
                     sum(c.isdigit() for c in text) / n,
                     sum(c.isspace() for c in text) / n, text.count('\n') / n,
                     len(set(w.casefold() for w in words)) / max(1, len(words)),
                     *[text.count(c) / n for c in '.,;:!?-()"\'@']], dtype='float32')


def source_hashes():
    root = Path(__file__).parent
    paths = [*root.glob('crossid_*.py'), root / 'prepare_crossid_training.py',
             root / 'benchmark_crossid.py', root / 'attribution_models/crossid.py']
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def check_sources(metadata):
    if metadata['source_sha256'] != source_hashes():
        raise ValueError('Training source changed; retrain with the current source.')
