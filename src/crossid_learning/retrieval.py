"""Learned profile retrieval, full rankings, and candidate/reference selection."""
import numpy as np
import torch

from .common import check_reference_target
from .models import encode_matrix


def top_mean(values, k, axis=-1):
    k = min(k, values.shape[axis])
    if k < 1:
        raise ValueError('Top-k must be positive.')
    return np.sort(values, axis=axis).take(indices=range(values.shape[axis]-k, values.shape[axis]),
                                          axis=axis).mean(axis=axis)


class RetrievalIndex:
    def __init__(self, model, refs, reference_matrix, device='cpu'):
        self.refs = refs
        self.authors = sorted(refs.author.unique())
        self.encoded = encode_matrix(model, reference_matrix, device)
        self.indices = [np.flatnonzero(refs.author.to_numpy() == author) for author in self.authors]
        prototypes, centroids = [], []
        with torch.inference_mode():
            for indices in self.indices:
                x = torch.as_tensor(self.encoded[indices][None, ...], device=device)
                mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
                prototypes.append(model.bundle(x, mask)[0].cpu().numpy())
                centroid = self.encoded[indices].mean(axis=0)
                centroids.append(centroid / max(np.linalg.norm(centroid), 1e-12))
        self.prototypes, self.centroids = np.stack(prototypes), np.stack(centroids)

    def score(self, targets, prototype_weight=0.5, reference_top_k=4, batch_size=64):
        if not 0 <= prototype_weight <= 1 or batch_size < 1 or reference_top_k < 1:
            raise ValueError('Invalid retrieval mixture, batch size, or reference top-k.')
        scores, local, centroid = [], [], []
        for start in range(0, len(targets), batch_size):
            z = targets[start:start+batch_size]
            prototype = top_mean(np.einsum('bd,akd->bak', z, self.prototypes), 2)
            reference = np.stack([top_mean(z @ self.encoded[indices].T, reference_top_k)
                                  for indices in self.indices], axis=1)
            scores.append(prototype_weight * prototype + (1-prototype_weight) * reference)
            local.append(reference); centroid.append(z @ self.centroids.T)
        return np.concatenate(scores), np.concatenate(local), np.concatenate(centroid)

    def select_references(self, target, author_index, count):
        indices = self.indices[author_index]
        # IDs break ties without exposing author names to the text model.
        order = np.lexsort((self.refs.iloc[indices].id.to_numpy(), -(self.encoded[indices] @ target)))
        return indices[order[:count]]


def descending_order(scores):
    if np.asarray(scores).ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('Need a finite two-dimensional author score matrix.')
    return np.argsort(-scores, axis=1, kind='stable')


def reranked_order(retrieval, candidates, candidate_scores):
    order = descending_order(retrieval)
    for i, chosen in enumerate(candidates):
        # Rerank only the shortlist. Unretrieved authors retain retrieval order.
        ranking = np.lexsort((chosen, -np.asarray(candidate_scores[i])))
        selected = set(chosen)
        order[i] = np.concatenate([np.asarray(chosen)[ranking],
                                   [a for a in order[i] if a not in selected]])
    return order


def prediction_frame(targets, authors, order):
    # Author labels enter here, after inference and ranking, only for evaluation.
    if order.shape != (len(targets), len(authors)) or not np.array_equal(
            np.sort(order, axis=1), np.broadcast_to(np.arange(len(authors)), order.shape)):
        raise ValueError('Every prediction must rank all candidate authors exactly once.')
    author_to_index = {a: i for i, a in enumerate(authors)}
    true = np.asarray([author_to_index[a] for a in targets.author])
    ranks = (order == true[:, None]).argmax(axis=1) + 1
    return targets[['id', 'genre']].assign(label=targets.author.to_numpy(),
                                         prediction=np.asarray(authors)[order[:, 0]], rank=ranks)


def metrics(predictions):
    result = {}
    for genre in ['Overall', 'Article', 'Tweet']:
        d = predictions if genre == 'Overall' else predictions[predictions.genre == genre]
        rank = d['rank'].to_numpy()
        result[genre] = {'n': len(d), 'Accuracy': float((rank == 1).mean()) if len(d) else None,
                         'MRR': float((1/rank).mean()) if len(d) else None,
                         'Mean_Rank': float(rank.mean()) if len(d) else None,
                         'Median_Rank': float(np.median(rank)) if len(d) else None}
        result[genre].update({f'R@{k}': float((rank <= k).mean()) if len(d) else None
                              for k in [8, 16, 32, 64]})
    return result
