"""Frozen Phase 2 retrieval and Phase 3 artifact validation."""
import json
from pathlib import Path

from .common import (check_reference_target, digest, directory_digest,
                     load_embedding_view, require_matching_protocol, require_matching_source)
from .models import encode_matrix, load_adapter
from .reranking import build_candidates
from .retrieval import RetrievalIndex


def retrieve(model, refs, targets, reference_paths, target_paths, device='cpu', prototype_weight=0.5):
    check_reference_target(refs, targets)
    reference = load_embedding_view(reference_paths, refs)
    target = load_embedding_view(target_paths, targets)
    index = RetrievalIndex(model, refs, reference, device)
    encoded = encode_matrix(model, target, device)
    retrieval, local, centroid = index.score(encoded, prototype_weight=prototype_weight)
    return index, encoded, retrieval, local, centroid


def candidate_data(model, refs, targets, reference_paths, target_paths, device='cpu',
                   top_candidates=20, references_per_candidate=3, training=False):
    index, encoded, scores, local, centroid = retrieve(
        model, refs, targets, reference_paths, target_paths, device)
    data = build_candidates(index, targets, encoded, scores, local, centroid,
                            top_candidates, references_per_candidate, training)
    return index, scores, data


def load_reranker(folder, phase2_path, protocol_sha=None, device='cpu'):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    folder = Path(folder)
    metadata = json.loads((folder / 'reranker_manifest.json').read_text())
    if metadata['status'] != 'completed' or metadata['phase2_sha256'] != digest(phase2_path):
        raise ValueError('Incomplete reranker or changed Phase 2 checkpoint.')
    require_matching_source(metadata, ['reranking.py', 'retrieval.py'])
    if protocol_sha is not None:
        require_matching_protocol(metadata, protocol_sha)
    if metadata['model_digest'] != directory_digest(folder / 'model'):
        raise ValueError('Reranker model files changed.')
    model = AutoModelForSequenceClassification.from_pretrained(folder / 'model', local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(folder / 'model', local_files_only=True)
    return model, tokenizer, metadata


def validate_fusion(artifact, phase2, phase3, routing):
    m = artifact['metadata']
    if m.get('status') != 'completed' or m['phase2_sha256'] != digest(phase2):
        raise ValueError('Incomplete fusion or changed Phase 2 checkpoint.')
    if m['phase3_model_digest'] != directory_digest(Path(phase3) / 'model'):
        raise ValueError('Fusion does not match the cross-encoder weights.')
    if m['routing'] != routing:
        raise ValueError('Fusion shortlist/reference/token budgets do not match evaluation.')
    require_matching_source(m, ['reranking.py', 'retrieval.py', 'fusion.py'])
