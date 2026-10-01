"""Input validation and provenance shared by both learning phases."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ('Article', 'Tweet')
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def directory_digest(path):
    files = sorted(p for p in Path(path).rglob('*') if p.is_file())
    return hashlib.sha256(json.dumps({str(p.relative_to(path)): digest(p)
                                     for p in files}, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Choose a fresh output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_documents(path):
    df = pd.read_csv(path, dtype={'id': str, 'author': str, 'genre': str, 'text': str})
    required = ['id', 'author', 'genre', 'text']
    if not set(required).issubset(df) or df.empty:
        raise ValueError(f'{path}: need nonempty id, author, genre, text columns.')
    if df[required].isna().any().any() or df.id.duplicated().any():
        raise ValueError(f'{path}: null fields or duplicate IDs.')
    if not set(df.genre).issubset(GENRES) or (df.text.str.strip().str.len() < 2).any():
        raise ValueError(f'{path}: invalid genre or empty text.')
    return df.reset_index(drop=True)


def text_hash(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()


def check_reference_target(refs, targets, allow_text_overlap=False):
    if set(refs.id) & set(targets.id):
        raise ValueError('Reference and target IDs overlap.')
    if not allow_text_overlap and set(refs.text.map(text_hash)) & set(targets.text.map(text_hash)):
        raise ValueError('Reference and target text overlaps under different IDs.')
    if set(refs.author) != set(targets.author):
        raise ValueError('Reference and target authors must match.')
    if refs.author.nunique() < 2:
        raise ValueError('Attribution requires at least two candidate authors.')


def load_protocol(data_dir):
    data_dir = Path(data_dir)
    path = data_dir / 'learning_protocol.json'
    p = json.loads(path.read_text())
    if p.get('role') != 'crossid_phase23_silver' or p.get('status') != 'completed':
        raise ValueError('Need a completed Phase 2/3 silver protocol.')
    for name, sha in p['files'].items():
        if digest(data_dir / name) != sha:
            raise ValueError(f'Learning split changed: {name}')
    seen = set()
    for role in ['train', 'dev', 'calibration', 'test']:
        authors = set(p['author_splits'][role])
        if not authors or seen & authors or authors & set(p['excluded_gold_authors']):
            raise ValueError('Learning author splits overlap or contain gold authors.')
        seen |= authors
    return p, digest(path)


def protect_evaluation(refs, targets, metadata, allow_text_overlap=False):
    check_reference_target(refs, targets, allow_text_overlap)
    excluded = set(metadata['train_authors']) | set(metadata['dev_authors'])
    excluded |= set(metadata.get('calibration_authors', []))
    if excluded & (set(refs.author) | set(targets.author)):
        raise ValueError('Evaluation authors overlap training/development/calibration authors.')


def load_embedding_view(paths, frame, dtype=np.float32):
    wanted = set(frame.id)
    found, dimension = {}, None
    for path in paths:
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError(f'{path}: need a document ID to vector mapping.')
        for doc_id in wanted & values.keys():
            v = np.asarray(values[doc_id], dtype=dtype)
            if (v.ndim != 1 or not v.size or not np.isfinite(v).all()
                    or not np.isfinite(np.linalg.norm(v)) or np.linalg.norm(v) == 0):
                raise ValueError(f'{path}: invalid embedding for {doc_id}.')
            if dimension is None:
                dimension = len(v)
            if len(v) != dimension:
                raise ValueError('Inconsistent embedding dimensions.')
            if doc_id in found and not np.array_equal(found[doc_id], v):
                raise ValueError(f'Conflicting embeddings in the same view: {doc_id}')
            found[doc_id] = v
        del values
    if wanted - found.keys():
        missing = sorted(wanted - found.keys())
        raise ValueError(f'Missing {len(missing)} embeddings; examples: {missing[:5]}')
    return np.stack([found[doc_id] for doc_id in frame.id])


def provenance():
    files = sorted(Path(__file__).parent.glob('*.py')) + [ROOT / 'src' / name for name in
             ['prepare_crossid_learning.py', 'train_crossid_phase2.py',
              'train_crossid_phase3.py', 'benchmark_crossid_phase23.py']]
    return {'source_sha256': {p.name: digest(p) for p in files},
            'versions': {name: importlib.metadata.version(name)
                         for name in ['numpy', 'pandas', 'scikit-learn', 'torch']}}


def require_matching_protocol(metadata, sha):
    if metadata['protocol_sha256'] != sha:
        raise ValueError('Checkpoint and learning protocol do not match.')


def require_matching_source(metadata, filenames):
    for name in filenames:
        if metadata['source_sha256'].get(name) != digest(Path(__file__).parent / name):
            raise ValueError(f'Inference source changed since fitting the artifact: {name}')
