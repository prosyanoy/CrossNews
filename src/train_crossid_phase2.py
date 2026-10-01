"""Train a CROSS-ID adapter on frozen SELMA views; select checkpoints on silver dev."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from crossid_learning.common import (ROOT, check_reference_target, digest, fresh_directory,
                                     load_embedding_view, load_protocol, provenance,
                                     read_documents, write_json)
from crossid_learning.models import CrossIDAdapter, encode_matrix, training_loss
from crossid_learning.retrieval import RetrievalIndex, descending_order, metrics, prediction_frame
from crossid_learning.sampling import BundleSampler, pseudo_topics


def train(args):
    if (min(args.epochs, args.steps_per_epoch, args.max_bundle, args.hidden_dim, args.output_dim,
            args.prototypes) < 1 or args.authors_per_batch < 2 or args.topic_clusters < 2
            or args.lr <= 0 or args.temperature <= 0 or args.reversal < 0
            or min(args.adversary_weight, args.bundle_weight) < 0):
        raise ValueError('Invalid training dimensions, counts, or loss settings.')
    protocol, protocol_sha = load_protocol(args.data_dir)
    frame = read_documents(args.data_dir / 'train/documents.csv')
    refs = read_documents(args.data_dir / 'dev/query/CrossNews_Both.csv')
    targets = read_documents(args.data_dir / 'dev/targets.csv')
    check_reference_target(refs, targets)
    if set(frame.author) != set(protocol['author_splits']['train']):
        raise ValueError('Training CSV does not match the author protocol.')
    reference = load_embedding_view(args.reference_embeddings, frame)
    target = load_embedding_view(args.target_embeddings, frame)
    dev_ref = load_embedding_view(args.reference_embeddings, refs)
    dev_target = load_embedding_view(args.target_embeddings, targets)
    if len({a.shape[1] for a in [reference, target, dev_ref, dev_target]}) != 1:
        raise ValueError('Reference/target embedding dimensions must match.')
    if args.check:
        print(json.dumps({'status': 'preflight_passed', 'train_documents': len(frame),
                          'train_authors': frame.author.nunique(), 'dev_targets': len(targets),
                          'input_dim': reference.shape[1]}, indent=2)); return
    output = fresh_directory(args.output)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    topics, topic_count = pseudo_topics(frame.text, args.topic_clusters, args.seed)
    sampler = BundleSampler(frame, topics, args.seed)
    model = CrossIDAdapter(reference.shape[1], args.hidden_dim, args.output_dim,
                           args.prototypes, topic_count).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    metadata = {'status': 'running', 'scope': 'Phase 2 trainable SELMA adapter; backbone remains frozen',
                'protocol_sha256': protocol_sha, 'train_authors': protocol['author_splits']['train'],
                'dev_authors': protocol['author_splits']['dev'], 'seed': args.seed,
                'pseudo_topics': 'training-only TF-IDF MiniBatchKMeans content proxies',
                'inputs': {str(p.resolve()): digest(p) for p in args.reference_embeddings + args.target_embeddings},
                'selection': 'highest dev MRR, Both references; ties retain earlier epoch',
                'retrieval': {'prototype_weight': 0.5, 'reference_top_k': 4},
                'arguments': {k: str(v) if isinstance(v, Path) else [str(p) for p in v]
                              if isinstance(v, list) else v for k, v in vars(args).items()},
                **provenance()}
    write_json(output / 'training_manifest.json', metadata)
    history, best, checkpoint_path = [], -float('inf'), output / 'adapter.pt'
    try:
        for epoch in range(args.epochs):
            model.train(); totals = []
            for step in range(args.steps_per_epoch):
                anchor_ids, positive_ids, bundle_ids = sampler.sample(args.authors_per_batch, args.max_bundle)
                size = max(map(len, bundle_ids))
                bundles = np.zeros((len(bundle_ids), size, reference.shape[1]), dtype=np.float32)
                mask = np.zeros(bundles.shape[:2], dtype=bool)
                for i, ids in enumerate(bundle_ids):
                    bundles[i, :len(ids)] = reference[ids]; mask[i, :len(ids)] = True
                ids = np.concatenate([anchor_ids, positive_ids])
                genre = (frame.iloc[ids].genre == 'Tweet').to_numpy().astype(np.int64)
                tensor = lambda x: torch.as_tensor(x, device=args.device)
                loss, components = training_loss(
                    model, tensor(target[anchor_ids]), tensor(reference[positive_ids]), tensor(bundles),
                    tensor(mask), tensor(topics[ids]), tensor(genre), temperature=args.temperature,
                    reversal=args.reversal, adversary_weight=args.adversary_weight,
                    bundle_weight=args.bundle_weight)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss.')
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                totals.append(float(loss.detach()))
            model.eval()
            index = RetrievalIndex(model, refs, dev_ref, args.device)
            z = encode_matrix(model, dev_target, args.device)
            scores, _, _ = index.score(z)
            dev = metrics(prediction_frame(targets, index.authors, descending_order(scores)))
            record = {'epoch': epoch+1, 'mean_train_loss': float(np.mean(totals)), 'dev': dev}
            history.append(record); write_json(output / 'history.json', history)
            print(json.dumps(record), flush=True)
            if dev['Overall']['MRR'] > best:
                best = dev['Overall']['MRR']; metadata['selected_epoch'] = epoch+1
                metadata['selected_dev_metrics'] = dev
                torch.save({'config': model.config, 'state_dict': model.cpu().state_dict(),
                            'metadata': metadata}, checkpoint_path)
                model.to(args.device)
        metadata['status'] = 'completed'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        checkpoint['metadata'] = metadata
        torch.save(checkpoint, checkpoint_path)
        metadata['checkpoint_sha256'] = digest(checkpoint_path)
    except Exception as exc:
        metadata.update(status='failed', error=str(exc)); raise
    finally:
        write_json(output / 'training_manifest.json', metadata)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'crossid_learning_data')
    p.add_argument('--reference-embeddings', type=Path, nargs='+', required=True)
    p.add_argument('--target-embeddings', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'results/crossid_phase2')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for name, default in [('epochs', 10), ('steps-per-epoch', 200), ('authors-per-batch', 16),
                          ('max-bundle', 8), ('hidden-dim', 256), ('output-dim', 256),
                          ('prototypes', 4), ('topic-clusters', 32), ('seed', 20261001)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('lr', 1e-3), ('temperature', .07), ('reversal', .1),
                          ('adversary-weight', .1), ('bundle-weight', .5)]:
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--check', action='store_true')
    return p


if __name__ == '__main__':
    p = parser()
    try:
        train(p.parse_args())
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
