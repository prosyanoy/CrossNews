"""Candidate-conditioned text cross-encoder and held-out stylometry fusion."""
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from benchmark_crossid import digest
from crossid_common import (STYLE_NAMES, check_sources, cohort, disjoint, fresh,
                            identity, load_vectors, ranking, source_hashes, stylometry, write_json)
from crossid_phase2 import load_adapter, retrieve

FEATURE_NAMES = ['retrieval', 'cross_encoder', 'style_cosine', 'style_negative_distance']


def directory_digest(folder):
    folder = Path(folder)
    h = hashlib.sha256()
    for path in sorted(p for p in folder.rglob('*') if p.is_file()):
        h.update(str(path.relative_to(folder)).encode())
        h.update(bytes.fromhex(digest(path)))
    return h.hexdigest()


class TinyPairEncoder(nn.Module):
    """Offline test fixture: byte-token transformer, not a pretrained baseline."""
    def __init__(self, max_length=256):
        super().__init__()
        self.max_length = max_length
        self.tokens = nn.Embedding(259, 32, padding_idx=0)
        self.positions = nn.Embedding(max_length, 32)
        self.segments = nn.Embedding(2, 32)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            32, 4, dim_feedforward=64, dropout=.1, batch_first=True), 1, enable_nested_tensor=False)
        self.head = nn.Linear(32, 1)

    def forward(self, tokens, segments):
        hidden = self.tokens(tokens) + self.segments(segments)
        hidden = hidden + self.positions(torch.arange(tokens.shape[1], device=tokens.device))
        hidden = self.encoder(hidden, src_key_padding_mask=tokens == 0)
        return self.head(hidden[:, 0]).squeeze(-1)


class PairEncoder:
    def __init__(self, name, device='cpu', max_length=512, revision=None, load=False):
        if max_length < 8:
            raise ValueError('Cross-encoder max length must be >=8.')
        self.name, self.device, self.max_length = name, device, max_length
        self.tiny = name == 'tiny'
        self.tokenizer = None
        if self.tiny:
            self.model = TinyPairEncoder(max_length).to(device)
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            options = {'local_files_only': True} if load else {'revision': revision}
            self.tokenizer = AutoTokenizer.from_pretrained(name, **options)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                name, num_labels=1, **options).to(device)

    def forward(self, pairs):
        left, right = zip(*pairs)
        if not self.tiny:
            inputs = self.tokenizer(list(left), list(right), padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors='pt')
            return self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits.flatten()
        budget = (self.max_length - 3) // 2
        encoded = []
        for a, b in pairs:
            a, b = [v + 3 for v in a.encode()[:budget]], [v + 3 for v in b.encode()[:budget]]
            encoded.append(([1, *a, 2, *b, 2], [0] * (len(a) + 2) + [1] * (len(b) + 1)))
        width = max(len(t) for t, _ in encoded)
        tokens = torch.tensor([t + [0] * (width - len(t)) for t, _ in encoded], device=self.device)
        segments = torch.tensor([s + [0] * (width - len(s)) for _, s in encoded], device=self.device)
        return self.model(tokens, segments)

    def predict(self, pairs, batch_size=16):
        self.model.eval()
        with torch.inference_mode():
            return np.concatenate([self.forward(pairs[i:i + batch_size]).cpu().numpy()
                                   for i in range(0, len(pairs), batch_size)])

    def save(self, folder):
        folder = fresh(folder)
        if self.tiny:
            torch.save(self.model.state_dict(), folder / 'weights.pt')
        else:
            self.model.save_pretrained(folder)
            self.tokenizer.save_pretrained(folder)


def checkpoint_hash(adapter):
    return directory_digest(adapter)


def load_reranker(path, adapter, device='cpu'):
    path = Path(path)
    meta = json.loads((path / 'metadata.json').read_text())
    if meta.get('format') != 'crossid_reranker_v1':
        raise ValueError('Unsupported reranker checkpoint.')
    check_sources(meta)
    if meta['adapter_sha256'] != checkpoint_hash(adapter):
        raise ValueError('Reranker was trained with another adapter.')
    if meta['weights_sha256'] != directory_digest(path / 'encoder'):
        raise ValueError('Cross-encoder weights changed after training.')
    name = 'tiny' if meta['encoder'] == 'tiny' else str(path / 'encoder')
    encoder = PairEncoder(name, device, meta['max_length'], load=True)
    if encoder.tiny:
        encoder.model.load_state_dict(torch.load(path / 'encoder/weights.pt',
                                                 map_location=device, weights_only=True))
    encoder.model.eval()
    return encoder, meta


def candidate_pair(refs, target_text, target_vector, bundle, reference_count, target_genre=None):
    indices = bundle['indices']
    # During training, encourage a cross-genre positive bundle. Inference never
    # consults the true author and uses only target genre/text/embedding.
    eligible = np.arange(len(indices))
    if target_genre:
        cross = np.flatnonzero(refs.iloc[indices].genre.to_numpy() != target_genre)
        if len(cross):
            eligible = cross
    order = eligible[ranking(bundle['references'][eligible] @ target_vector)[:reference_count]]
    texts = refs.iloc[indices[order]].text.tolist()
    return target_text, '\n\n'.join(texts)


def candidate_rows(scores, authors, targets, shortlist, training=False, negatives=3):
    rows = []
    for i, target in enumerate(targets.itertuples()):
        order = ranking(scores[i])[:shortlist].tolist()
        label = authors.index(target.author)
        if training:
            order = [label, *[j for j in order if j != label][:negatives]]
            if len(order) < 2:
                raise ValueError('Need at least one negative author per training target.')
        rows.extend((i, j, int(j == label)) for j in order)
    return rows


def ensure_heldout(metadata, refs, targets):
    ident = identity(refs, targets)
    for key in ('train_identity', 'dev_identity'):
        disjoint(metadata[key], ident)
    return ident


def train(args):
    if min(args.epochs, args.batch_size, args.shortlist, args.reference_count, args.negatives) < 1:
        raise ValueError('Training counts must be positive.')
    if args.shortlist < 2 or args.learning_rate <= 0:
        raise ValueError('Need shortlist >=2 and positive learning rate.')
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    adapter, adapter_meta = load_adapter(args.adapter, args.device)
    refs, targets, _ = cohort(args.data_dir, 'train')
    dev_refs, dev_targets, _ = cohort(args.data_dir, 'dev')
    if identity(refs, targets) != adapter_meta['train_identity'] or identity(dev_refs, dev_targets) != adapter_meta['dev_identity']:
        raise ValueError('Cross-encoder must use the adapter train/dev cohorts.')
    for path in (args.train_embeddings, args.target_embeddings, args.dev_train_embeddings, args.dev_target_embeddings):
        if adapter_meta['inputs'].get(str(Path(path).resolve())) != digest(Path(path)):
            raise ValueError('Cross-encoder must reuse the adapter train/dev embedding files.')
    rx, tx = load_vectors(args.train_embeddings, args.target_embeddings, refs, targets)
    drx, dtx = load_vectors(args.dev_train_embeddings, args.dev_target_embeddings, dev_refs, dev_targets)
    # Stage 2 representations remain fixed while mining cross-encoder examples.
    scores, authors, bundles, z = retrieve(adapter, refs, rx, tx, args.device)
    rows = candidate_rows(scores, authors, targets, args.shortlist, training=True, negatives=args.negatives)
    labels = np.array([label for _, _, label in rows], dtype='float32')
    ds, da, db, dz = retrieve(adapter, dev_refs, drx, dtx, args.device)
    dev_rows = candidate_rows(ds, da, dev_targets, args.shortlist)
    encoder = PairEncoder(args.encoder, args.device, args.max_length, args.revision)
    optim = torch.optim.AdamW(encoder.model.parameters(), lr=args.learning_rate)
    positive_weight = torch.tensor(float(np.sum(labels == 0) / np.sum(labels == 1)), device=args.device)
    output = fresh(args.output)
    best, best_state, history = -1., None, []
    for epoch in range(args.epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        encoder.model.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            pairs = [candidate_pair(refs, targets.iloc[i].text, z[i], bundles[j], args.reference_count,
                                    targets.iloc[i].genre if label else None)
                     for index in indices for i, j, label in [rows[index]]]
            logits = encoder.forward(pairs)
            loss = F.binary_cross_entropy_with_logits(logits, torch.as_tensor(labels[indices], device=args.device),
                                                      pos_weight=positive_weight)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite cross-encoder loss.')
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.)
            optim.step()
            losses.append(float(loss.detach()))
        predictions = []
        for start in range(0, len(dev_rows), args.batch_size):
            pairs = [candidate_pair(dev_refs, dev_targets.iloc[i].text, dz[i], db[j], args.reference_count)
                     for i, j, _ in dev_rows[start:start + args.batch_size]]
            predictions.extend(encoder.predict(pairs, args.batch_size).tolist())
        grouped = [[] for _ in range(len(dev_targets))]
        for (i, j, _), score in zip(dev_rows, predictions):
            grouped[i].append((float(score), j))
        chosen = [max(values, key=lambda v: (v[0], -v[1]))[1] for values in grouped]
        accuracy = float(np.mean(np.array(da)[chosen] == dev_targets.author.to_numpy()))
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'dev_accuracy': accuracy})
        print(history[-1], flush=True)
        if accuracy > best:
            best, best_state = accuracy, copy.deepcopy({k: v.cpu() for k, v in encoder.model.state_dict().items()})
    encoder.model.load_state_dict(best_state)
    encoder.save(output / 'encoder')
    style = np.stack(refs.text.map(stylometry))
    style_scaler = StandardScaler().fit(style)
    meta = {'format': 'crossid_reranker_v1', 'encoder': args.encoder, 'revision': args.revision,
            'max_length': args.max_length, 'shortlist': args.shortlist, 'reference_count': args.reference_count,
            'adapter_sha256': checkpoint_hash(args.adapter), 'weights_sha256': directory_digest(output / 'encoder'),
            'train_identity': adapter_meta['train_identity'], 'dev_identity': adapter_meta['dev_identity'],
            'source_sha256': source_hashes(), 'style_names': list(STYLE_NAMES),
            'resolved_revision': getattr(encoder.model.config, '_commit_hash', None) if not encoder.tiny else None,
            'style_mean': style_scaler.mean_.tolist(), 'style_scale': style_scaler.scale_.tolist(),
            'selection': 'Highest dev end-to-end cross-encoder top-1; shortlist misses count as incorrect.',
            'history': history, 'best_dev_accuracy': best,
            'hyperparameters': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))},
            'inputs': {str(Path(p).resolve()): digest(Path(p)) for p in
                       (args.train_embeddings, args.target_embeddings, args.dev_train_embeddings,
                        args.dev_target_embeddings, Path(args.data_dir) / 'protocol.json')},
            'scope': 'tiny is only an offline wiring test; use a pretrained encoder for experiments.'}
    write_json(output / 'metadata.json', meta)
    return meta


def features(encoder, meta, refs, targets, scores, authors, bundles, z, batch_size=16):
    rows = candidate_rows(scores, authors, targets, meta['shortlist'])
    ref_style = (np.stack(refs.text.map(stylometry)) - meta['style_mean']) / meta['style_scale']
    target_style = (np.stack(targets.text.map(stylometry)) - meta['style_mean']) / meta['style_scale']
    profiles = [ref_style[b['indices']].mean(0) for b in bundles]
    output = np.empty((len(rows), len(FEATURE_NAMES)), dtype='float64')
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        pairs = [candidate_pair(refs, targets.iloc[i].text, z[i], bundles[j], meta['reference_count'])
                 for i, j, _ in batch]
        ce = encoder.predict(pairs, batch_size)
        for offset, ((i, j, _), value) in enumerate(zip(batch, ce)):
            a, b = target_style[i], profiles[j]
            cosine = float(a @ b / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))
            output[start + offset] = [scores[i, j], value, cosine, -np.linalg.norm(a - b)]
    if not np.isfinite(output).all():
        raise ValueError('Non-finite fusion features.')
    return rows, output


def calibrate(args):
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive.')
    adapter, adapter_meta = load_adapter(args.adapter, args.device)
    encoder, reranker_meta = load_reranker(args.reranker, args.adapter, args.device)
    refs, targets, _ = cohort(args.data_dir, 'calibration', args.reference_set)
    calibration_identity = ensure_heldout(adapter_meta, refs, targets)
    ensure_heldout(reranker_meta, refs, targets)
    rx, tx = load_vectors(args.train_embeddings, args.target_embeddings, refs, targets)
    scores, authors, bundles, z = retrieve(adapter, refs, rx, tx, args.device)
    rows, x = features(encoder, reranker_meta, refs, targets, scores, authors, bundles, z, args.batch_size)
    y = np.array([label for _, _, label in rows])
    if len(np.unique(y)) != 2:
        raise ValueError('Calibration needs both correct and incorrect retrieved candidates.')
    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(C=1., class_weight='balanced', max_iter=2000, random_state=0).fit(scaler.transform(x), y)
    if lr.n_iter_.max() >= 2000:
        raise ValueError('Fusion did not converge.')
    output = fresh(args.output)
    fusion = {'format': 'crossid_fusion_v1', 'features': list(FEATURE_NAMES),
              'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
              'coefficient': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]),
              'adapter_sha256': checkpoint_hash(args.adapter), 'reranker_sha256': directory_digest(args.reranker),
              'calibration_identity': calibration_identity, 'reference_set': args.reference_set,
              'source_sha256': source_hashes(), 'shortlist': reranker_meta['shortlist'],
              'candidate_count': len(rows), 'shortlist_recall': float(y.sum() / len(targets)),
              'inputs': {str(Path(p).resolve()): digest(Path(p)) for p in
                         (args.train_embeddings, args.target_embeddings, Path(args.data_dir) / 'protocol.json')},
              'fitting': 'Fixed C=1 balanced logistic classifier on calibration candidates; logit used only for ranking.'}
    write_json(output / 'fusion.json', fusion)
    return fusion


def load_fusion(path, adapter, reranker, reference):
    fusion = json.loads(Path(path).read_text())
    if fusion.get('format') != 'crossid_fusion_v1' or fusion.get('features') != FEATURE_NAMES:
        raise ValueError('Unsupported fusion schema.')
    check_sources(fusion)
    if fusion['adapter_sha256'] != checkpoint_hash(adapter) or fusion['reranker_sha256'] != directory_digest(reranker):
        raise ValueError('Fusion checkpoints changed after calibration.')
    if fusion['reference_set'] != reference:
        raise ValueError('Reference condition differs from fusion calibration.')
    return fusion


def reorder(retrieval_order, candidate_indices, candidate_scores):
    candidates = sorted(zip(candidate_indices, candidate_scores), key=lambda item: (-item[1], item[0]))
    front = [j for j, _ in candidates]
    if set(front) != set(retrieval_order[:len(front)]):
        raise ValueError('Reranker may reorder only the retrieved shortlist.')
    return front + list(retrieval_order[len(front):])
