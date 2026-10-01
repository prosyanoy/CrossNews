"""Train candidate-conditioned reranking, then fit fusion on separate silver calibration authors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from crossid_learning.common import (ROOT, digest, directory_digest, fresh_directory, load_protocol,
                                     provenance, read_documents, require_matching_protocol, write_json)
from crossid_learning.fusion import fit_fusion
from crossid_learning.models import load_adapter
from crossid_learning.pipeline import candidate_data, load_reranker
from crossid_learning.reranking import candidate_features, cross_encoder_logits, train_cross_encoder_epoch
from crossid_learning.retrieval import metrics, prediction_frame, reranked_order


def load_inputs(args, role, model, limit=None, training=False):
    refs = read_documents(args.data_dir / role / 'query/CrossNews_Both.csv')
    targets = read_documents(args.data_dir / role / 'targets.csv')
    if limit and len(targets) > limit:
        # Keep every candidate author represented, with balanced genres.
        groups = targets.groupby(['author', 'genre'], group_keys=False)
        per_group = min(max(1, limit//groups.ngroups), int(groups.size().min()))
        targets = groups.sample(n=per_group, random_state=args.seed)
        targets = targets.reset_index(drop=True)
    index, scores, data = candidate_data(model, refs, targets, args.reference_embeddings,
                                        args.target_embeddings, args.device, args.top_candidates,
                                        args.references_per_candidate, training)
    return refs, targets, index, scores, data


def train(args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if min(args.epochs, args.batch_size, args.max_pairs, args.top_candidates,
           args.references_per_candidate) < 1 or args.max_length < 8 or args.dev_queries < 2 or args.lr <= 0:
        raise ValueError('Invalid cross-encoder training settings.')
    protocol, protocol_sha = load_protocol(args.data_dir)
    adapter, phase2_meta = load_adapter(args.phase2, args.device)
    require_matching_protocol(phase2_meta, protocol_sha)
    _, targets, index, scores, training = load_inputs(args, 'train', adapter, training=True)
    _, dev_targets, dev_index, dev_scores, dev = load_inputs(args, 'dev', adapter, args.dev_queries)
    if args.check:
        print(json.dumps({'status': 'preflight_passed', 'train_pairs': len(training.pairs),
                          'dev_pairs': len(dev.pairs)}, indent=2)); return
    output = fresh_directory(args.output)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, num_labels=1, ignore_mismatched_sizes=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    metadata = {'status': 'running', 'phase2_sha256': digest(args.phase2),
                'protocol_sha256': protocol_sha, 'train_authors': protocol['author_splits']['train'],
                'dev_authors': protocol['author_splits']['dev'], 'base_model': args.model,
                'requested_revision': args.revision, 'resolved_revision': getattr(model.config, '_commit_hash', None),
                'seed': args.seed, 'max_length': args.max_length, 'max_chars_per_text': 2000,
                'top_candidates': args.top_candidates, 'references_per_candidate': args.references_per_candidate,
                'training_positive_injection': True, 'evaluation_positive_injection': False,
                'selection': 'highest author attribution MRR on fixed silver dev subset',
                'dev_queries_used': len(dev_targets),
                'inputs': {str(p.resolve()): digest(p) for p in args.reference_embeddings + args.target_embeddings},
                **provenance()}
    write_json(output / 'reranker_manifest.json', metadata)
    best, history = -float('inf'), []
    try:
        for epoch in range(args.epochs):
            loss = train_cross_encoder_epoch(model, tokenizer, training, optimizer, rng, args.device,
                                             args.batch_size, args.max_length, args.max_pairs)
            features = candidate_features(dev, cross_encoder_logits(
                model, tokenizer, dev.pairs, args.device, args.batch_size, args.max_length))
            order = reranked_order(dev_scores, dev.candidates, features[..., 3])
            result = metrics(prediction_frame(dev_targets, dev_index.authors, order))
            record = {'epoch': epoch+1, 'train_loss': loss, 'dev': result}
            history.append(record); write_json(output / 'history.json', history)
            print(json.dumps(record), flush=True)
            if result['Overall']['MRR'] > best:
                best = result['Overall']['MRR']; metadata['selected_epoch'] = epoch+1
                metadata['selected_dev_metrics'] = result
                model.save_pretrained(output / 'model', safe_serialization=True)
                tokenizer.save_pretrained(output / 'model')
        metadata['status'] = 'completed'; metadata['model_digest'] = directory_digest(output / 'model')
    except Exception as exc:
        metadata.update(status='failed', error=str(exc)); raise
    finally:
        write_json(output / 'reranker_manifest.json', metadata)


def calibrate(args):
    if min(args.top_candidates, args.references_per_candidate, args.batch_size) < 1:
        raise ValueError('Candidate/reference/batch counts must be positive.')
    protocol, protocol_sha = load_protocol(args.data_dir)
    adapter, phase2_meta = load_adapter(args.phase2, args.device)
    require_matching_protocol(phase2_meta, protocol_sha)
    model, tokenizer, phase3_meta = load_reranker(args.phase3, args.phase2, protocol_sha, args.device)
    routing = {'reference_set': 'Both', 'top_candidates': args.top_candidates,
               'references_per_candidate': args.references_per_candidate,
               'max_length': phase3_meta['max_length'], 'max_chars_per_text': 2000}
    _, targets, index, scores, data = load_inputs(args, 'calibration', adapter)
    if args.check:
        print(json.dumps({'status': 'preflight_passed', 'calibration_pairs': len(data.pairs)}, indent=2)); return
    if args.output.exists():
        raise ValueError('Fusion output already exists; use a fresh path.')
    features = candidate_features(data, cross_encoder_logits(
        model, tokenizer, data.pairs, args.device, args.batch_size, phase3_meta['max_length']))
    labels = np.asarray([[index.authors[j] == author for j in chosen]
                         for author, chosen in zip(targets.author, data.candidates)], dtype=int)
    metadata = {'status': 'completed', 'protocol_sha256': protocol_sha,
                'phase2_sha256': digest(args.phase2), 'phase3_model_digest': phase3_meta['model_digest'],
                'routing': routing, 'train_authors': protocol['author_splits']['train'],
                'dev_authors': protocol['author_splits']['dev'],
                'calibration_authors': protocol['author_splits']['calibration'],
                'calibration_targets_sha256': digest(args.data_dir / 'calibration/targets.csv'),
                'calibration_references_sha256': digest(args.data_dir / 'calibration/query/CrossNews_Both.csv'),
                'inputs': {str(p.resolve()): digest(p) for p in args.reference_embeddings + args.target_embeddings},
                'fit': 'StandardScaler + L2 logistic membership fusion, C=1; no test-based tuning',
                'calibration_candidates': int(labels.size), 'retrieved_positives': int(labels.sum()),
                **provenance()}
    artifact = fit_fusion(features, labels, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True); write_json(args.output, artifact)
    print(json.dumps({'status': 'completed', 'fusion_path': str(args.output),
                      'candidate_recall': float(labels.sum()/len(targets))}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ['train', 'calibrate']:
        s = sub.add_parser(name)
        s.add_argument('--data-dir', type=Path, default=ROOT / 'crossid_learning_data')
        s.add_argument('--phase2', type=Path, required=True)
        s.add_argument('--reference-embeddings', type=Path, nargs='+', required=True)
        s.add_argument('--target-embeddings', type=Path, nargs='+', required=True)
        s.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        s.add_argument('--top-candidates', type=int, default=20)
        s.add_argument('--references-per-candidate', type=int, default=3)
        s.add_argument('--batch-size', type=int, default=16)
        s.add_argument('--seed', type=int, default=20261001)
        s.add_argument('--check', action='store_true')
        if name == 'train':
            s.add_argument('--model', default='distilbert-base-uncased')
            s.add_argument('--revision', default=None)
            s.add_argument('--epochs', type=int, default=3)
            s.add_argument('--lr', type=float, default=2e-5)
            s.add_argument('--max-length', type=int, default=512)
            s.add_argument('--max-pairs', type=int, default=100000)
            s.add_argument('--dev-queries', type=int, default=300)
            s.add_argument('--output', type=Path, default=ROOT / 'results/crossid_phase3')
        else:
            s.add_argument('--phase3', type=Path, required=True)
            s.add_argument('--output', type=Path, default=ROOT / 'results/crossid_phase3/fusion.json')
    return p


if __name__ == '__main__':
    p = parser(); args = p.parse_args()
    try:
        (train if args.command == 'train' else calibrate)(args)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
