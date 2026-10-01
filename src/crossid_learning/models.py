"""Embedding adapter, adversarial heads, and identity-independent set encoder."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class CrossIDAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=256, prototypes=4, topics=32):
        super().__init__()
        if min(input_dim, hidden_dim, output_dim, prototypes, topics) < 1:
            raise ValueError('Model dimensions must be positive.')
        self.config = dict(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                           prototypes=prototypes, topics=topics)
        self.skip = nn.Linear(input_dim, output_dim, bias=False)
        self.adapter = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
        self.slots = nn.Parameter(torch.randn(prototypes, output_dim) / output_dim**0.5)
        self.topic_head = nn.Linear(output_dim, topics)
        self.genre_head = nn.Linear(output_dim, 2)

    def encode(self, x):
        x = F.normalize(x, dim=-1)
        return F.normalize(self.skip(x) + self.adapter(x), dim=-1)

    def bundle(self, encoded, mask):
        """Masked attention from shared learned slots to any-size author bundle."""
        if encoded.ndim != 3 or mask.shape != encoded.shape[:2] or not mask.any(dim=1).all():
            raise ValueError('Every author bundle needs at least one unmasked reference.')
        attention = torch.einsum('kd,brd->bkr', F.normalize(self.slots, dim=-1), encoded) * 5
        attention = attention.masked_fill(~mask[:, None, :], float('-inf')).softmax(dim=-1)
        return F.normalize(torch.einsum('bkr,brd->bkd', attention, encoded), dim=-1)

    def adversarial(self, encoded, strength):
        reversed_features = GradientReverse.apply(encoded, strength)
        return self.topic_head(reversed_features), self.genre_head(reversed_features)


def bundle_scores(anchors, prototypes, temperature=0.07):
    values = torch.einsum('bd,akd->bak', anchors, prototypes)
    return values.topk(min(2, prototypes.shape[1]), dim=-1).values.mean(dim=-1) / temperature


def training_loss(model, anchors, positives, bundles, mask, topic_labels, genre_labels,
                  temperature=0.07, reversal=0.1, adversary_weight=0.1, bundle_weight=0.5):
    if anchors.shape[0] < 2 or temperature <= 0:
        raise ValueError('Contrastive training needs >=2 different authors and positive temperature.')
    z1, z2 = model.encode(anchors), model.encode(positives)
    z_bundle = model.encode(bundles)
    prototypes = model.bundle(z_bundle, mask)
    labels = torch.arange(len(anchors), device=anchors.device)
    similarity = z1 @ z2.T / temperature
    contrastive = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)) / 2
    bundle_loss = F.cross_entropy(bundle_scores(z1, prototypes, temperature), labels)
    topic_logits, genre_logits = model.adversarial(torch.cat([z1, z2]), reversal)
    adversary = F.cross_entropy(topic_logits, topic_labels.long()) + F.cross_entropy(genre_logits, genre_labels.long())
    # Penalize coincident prototypes; slots are shared, not tied to train identities.
    k = prototypes.shape[1]
    diversity = ((prototypes @ prototypes.transpose(1, 2)) -
                 torch.eye(k, device=anchors.device)).square().mean()
    loss = contrastive + bundle_weight * bundle_loss + adversary_weight * adversary + 0.01 * diversity
    return loss, {'contrastive': float(contrastive.detach()), 'bundle': float(bundle_loss.detach()),
                  'adversary': float(adversary.detach()), 'diversity': float(diversity.detach())}


def encode_matrix(model, matrix, device='cpu', batch_size=256):
    if matrix.shape[1] != model.config['input_dim'] or batch_size < 1:
        raise ValueError('Embedding dimension or batch size does not match the adapter.')
    model.eval()
    with torch.inference_mode():
        parts = [model.encode(torch.as_tensor(matrix[i:i+batch_size], dtype=torch.float32, device=device)).cpu().numpy()
                 for i in range(0, len(matrix), batch_size)]
    return np.concatenate(parts)


def load_adapter(path, device='cpu'):
    from .common import require_matching_source
    # Only tensor state and primitive metadata, never a pickled module instance.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['metadata'].get('status') != 'completed':
        raise ValueError('Phase 2 training did not complete.')
    require_matching_source(checkpoint['metadata'], ['models.py'])
    model = CrossIDAdapter(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model, checkpoint['metadata']
