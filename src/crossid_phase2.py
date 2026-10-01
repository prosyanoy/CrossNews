"""Contrastive adapter and author-conditioned learned attention prototypes.

The SELMA backbone stays frozen. Prototype queries are shared across authors,
so inference builds profiles for unseen authors from references only.
"""
import copy
import json
from pathlib import Path
import random

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch import nn
from torch.nn import functional as F

from benchmark_crossid import digest
from crossid_common import (check_sources, cohort, disjoint, fresh, identity,
                            load_vectors, source_hashes, write_json)


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, strength):
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim=256, hidden_dim=512, prototypes=4):
        super().__init__()
        self.config = dict(input_dim=input_dim, output_dim=output_dim,
                           hidden_dim=hidden_dim, prototypes=prototypes)
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.residual = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, output_dim))
        self.queries = nn.Parameter(torch.randn(prototypes, output_dim) / output_dim ** .5)

    def forward(self, value):
        value = F.normalize(value, dim=-1)
        return F.normalize(self.projection(value) + self.residual(value), dim=-1)

    def pool(self, references):
        attention = torch.softmax(4 * F.normalize(self.queries, dim=-1) @ references.T, dim=-1)
        return F.normalize(attention @ references, dim=-1)


def contrastive(z, authors, genres, topics, temperature=.1, hard_negative_bias=.2):
    logits = z @ z.T / temperature
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = (authors[:, None] == authors[None, :]) & (genres[:, None] != genres[None, :])
    negatives = authors[:, None] != authors[None, :]
    same_topic = topics[:, None] == topics[None, :]
    logits = logits + hard_negative_bias * (negatives & same_topic)
    log_prob = logits - torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True)
    counts = positives.sum(1)
    if (counts == 0).any():
        raise ValueError('Every contrastive anchor needs a cross-genre same-author positive.')
    return -(log_prob.masked_fill(~positives, 0).sum(1) / counts).mean()


def encode(model, values, device='cpu', batch_size=256):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(torch.as_tensor(values[i:i + batch_size], device=device)).cpu().numpy()
                               for i in range(0, len(values), batch_size)])


def profiles(model, refs, ref_vectors, device='cpu'):
    vectors = encode(model, ref_vectors, device)
    authors = sorted(refs.author.unique())
    bundles = []
    model.eval()
    with torch.inference_mode():
        for author in authors:
            indices = np.flatnonzero(refs.author.to_numpy() == author)
            matrix = vectors[indices]
            prototypes = model.pool(torch.as_tensor(matrix, device=device)).cpu().numpy()
            bundles.append({'indices': indices, 'references': matrix, 'prototypes': prototypes})
    return authors, bundles


def retrieve(model, refs, ref_vectors, target_vectors, device='cpu', prototype_weight=.5):
    authors, bundles = profiles(model, refs, ref_vectors, device)
    targets = encode(model, target_vectors, device)
    scores = np.empty((len(targets), len(authors)), dtype='float32')
    for j, bundle in enumerate(bundles):
        refs_scores = targets @ bundle['references'].T
        k = min(4, refs_scores.shape[1])
        local = np.sort(refs_scores, axis=1)[:, -k:].mean(1)
        # Smooth maximum matches prototype training, independent of K.
        similarities = torch.from_numpy(targets @ bundle['prototypes'].T)
        proto = (.1 * (torch.logsumexp(similarities / .1, dim=1)
                       - np.log(similarities.shape[1]))).numpy()
        scores[:, j] = prototype_weight * proto + (1 - prototype_weight) * local
    return scores, authors, bundles, targets


def load_adapter(path, device='cpu'):
    path = Path(path)
    meta = json.loads((path / 'metadata.json').read_text())
    if meta.get('format') != 'crossid_adapter_v1':
        raise ValueError('Unsupported adapter checkpoint.')
    check_sources(meta)
    if meta['weights_sha256'] != digest(path / 'adapter.pt'):
        raise ValueError('Adapter weights changed after training.')
    model = Adapter(**meta['config']).to(device)
    model.load_state_dict(torch.load(path / 'adapter.pt', map_location=device, weights_only=True))
    return model.eval(), meta


def train(args):
    if min(args.epochs, args.steps, args.authors_per_batch, args.max_bundle, args.topics) < 1:
        raise ValueError('Training counts must be positive.')
    if args.authors_per_batch < 2 or args.learning_rate <= 0:
        raise ValueError('Need at least 2 authors per batch and positive learning rate.')
    if min(args.output_dim, args.hidden_dim, args.prototypes) < 1 or args.adversary_strength < 0:
        raise ValueError('Dimensions/prototypes must be positive and adversary strength nonnegative.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_ref, train_target, _ = cohort(args.data_dir, 'train')
    dev_ref, dev_target, _ = cohort(args.data_dir, 'dev')
    train_identity, dev_identity = identity(train_ref, train_target), identity(dev_ref, dev_target)
    disjoint(train_identity, dev_identity)
    rx, tx = load_vectors(args.train_embeddings, args.target_embeddings, train_ref, train_target)
    drx, dtx = load_vectors(args.dev_train_embeddings, args.dev_target_embeddings, dev_ref, dev_target)
    if len(rx[0]) != len(drx[0]):
        raise ValueError('Train/dev embedding dimensions differ.')
    authors = sorted(train_ref.author.unique())
    if len(authors) < args.authors_per_batch:
        raise ValueError('Requested batch contains more authors than training cohort.')
    # Topic supervision is a train-only lexical proxy, not a gold topic annotation.
    texts = train_ref.text.tolist() + train_target.text.tolist()
    tfidf = TfidfVectorizer(max_features=12000, min_df=1).fit_transform(texts)
    topic_model = MiniBatchKMeans(n_clusters=min(args.topics, len(texts)), random_state=args.seed, n_init=3)
    topics = topic_model.fit_predict(tfidf).astype('int64')
    model = Adapter(rx.shape[1], args.output_dim, args.hidden_dim, args.prototypes).to(args.device)
    genre_head = nn.Linear(args.output_dim, 2).to(args.device)
    topic_head = nn.Linear(args.output_dim, topic_model.n_clusters).to(args.device)
    optim = torch.optim.AdamW([*model.parameters(), *genre_head.parameters(), *topic_head.parameters()],
                             lr=args.learning_rate)
    ref_groups = {(a, g): np.flatnonzero((train_ref.author == a) & (train_ref.genre == g)).tolist()
                  for a in authors for g in ('Article', 'Tweet')}
    target_groups = {(a, g): np.flatnonzero((train_target.author == a) & (train_target.genre == g)).tolist()
                     for a in authors for g in ('Article', 'Tweet')}
    if any(not v for v in [*ref_groups.values(), *target_groups.values()]):
        raise ValueError('Training needs both genres for references and targets of each author.')
    output = fresh(args.output)
    history, best, best_state = [], -1., None
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step in range(args.steps):
            selected = rng.sample(authors, args.authors_per_batch)
            reference_ids, target_ids, ref_labels, ref_genres = [], [], [], []
            for label, author in enumerate(selected):
                for genre_id, genre in enumerate(('Article', 'Tweet')):
                    pool = ref_groups[author, genre]
                    ids = rng.sample(pool, rng.randint(1, min(args.max_bundle, len(pool))))
                    reference_ids.extend(ids)
                    ref_labels.extend([label] * len(ids))
                    ref_genres.extend([genre_id] * len(ids))
                    target_ids.append(rng.choice(target_groups[author, genre]))
            refs_z = model(torch.as_tensor(rx[reference_ids], device=args.device))
            query_z = model(torch.as_tensor(tx[target_ids], device=args.device))
            label_tensor = torch.tensor(ref_labels, device=args.device)
            query_labels = torch.arange(len(selected), device=args.device).repeat_interleave(2)
            genre_labels = torch.tensor(ref_genres + [0, 1] * len(selected), device=args.device)
            labels = torch.cat([label_tensor, query_labels])
            topic_labels = torch.tensor(np.concatenate([topics[reference_ids],
                                          topics[len(rx) + np.array(target_ids)]]),
                                        dtype=torch.long, device=args.device)
            z = torch.cat([refs_z, query_z])
            con = contrastive(z, labels, genre_labels, topic_labels)
            pooled = torch.stack([model.pool(refs_z[label_tensor == i]) for i in range(len(selected))])
            similarity = torch.einsum('qd,akd->qak', query_z, pooled) / .1
            author_logits = torch.logsumexp(similarity, dim=-1)
            bundle_loss = F.cross_entropy(author_logits, query_labels)
            reversed_z = Reverse.apply(z, args.adversary_strength)
            adv = F.cross_entropy(genre_head(reversed_z), genre_labels)
            adv += F.cross_entropy(topic_head(reversed_z), topic_labels)
            # Penalize collapsed prototypes while preserving author evidence.
            gram = pooled @ pooled.transpose(1, 2)
            mask = ~torch.eye(args.prototypes, dtype=torch.bool, device=args.device)
            diversity = F.relu(gram[:, mask] - .8).mean() if args.prototypes > 1 else z.sum() * 0
            loss = con + bundle_loss + .1 * adv + .05 * diversity
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss.')
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*model.parameters(), *genre_head.parameters(), *topic_head.parameters()], 1.)
            optim.step()
            losses.append(float(loss.detach()))
        scores, dev_authors, _, _ = retrieve(model, dev_ref, drx, dtx, args.device)
        predictions = np.array(dev_authors)[np.argmax(scores, axis=1)]
        accuracy = float(np.mean(predictions == dev_target.author.to_numpy()))
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'dev_accuracy': accuracy})
        print(history[-1], flush=True)
        if accuracy > best:
            best, best_state = accuracy, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    torch.save(best_state, output / 'adapter.pt')
    meta = {'format': 'crossid_adapter_v1', 'config': model.config,
            'scope': 'Frozen SELMA backbone; trained projection, residual adapter and shared attention prototype queries.',
            'selection': 'Highest dev top-1; earliest epoch on ties.', 'best_dev_accuracy': best,
            'train_identity': train_identity, 'dev_identity': dev_identity,
            'inputs': {str(Path(p).resolve()): digest(Path(p)) for p in
                       (args.train_embeddings, args.target_embeddings, args.dev_train_embeddings,
                        args.dev_target_embeddings, Path(args.data_dir) / 'protocol.json')},
            'source_sha256': source_hashes(), 'weights_sha256': digest(output / 'adapter.pt'),
            'hyperparameters': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))},
            'history': history, 'torch_version': str(torch.__version__),
            'topic_proxy': 'Train-only TF-IDF MiniBatchKMeans; heads discarded at inference.'}
    write_json(output / 'metadata.json', meta)
    return meta
