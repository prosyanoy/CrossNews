"""Candidate-conditioned text pairs and low-level stylometry features."""
from dataclasses import dataclass
import re
import unicodedata

import numpy as np
import torch
from torch.nn import functional as F

from .retrieval import descending_order

FUNCTION_WORDS = ('the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on',
                  'for', 'with', 'is', 'was', 'it', 'that', 'this', 'not', 'as', 'by')
STYLE_NAMES = ['log_characters', 'log_words', 'mean_word_length', 'std_word_length',
               'upper_ratio', 'digit_ratio', 'whitespace_ratio', 'newline_ratio',
               'unicode_punctuation_ratio', 'non_ascii_ratio', 'words_per_sentence']
STYLE_NAMES += ['punctuation_'+str(i) for i in range(len('.,;:!?"\'()-'))]
STYLE_NAMES += ['function_'+word for word in FUNCTION_WORDS]
FEATURE_NAMES = ['retrieval', 'local_reference', 'centroid', 'cross_encoder_logit']
FEATURE_NAMES += ['style_distance_'+name for name in STYLE_NAMES]


def stylometry(text):
    words = re.findall(r'\b\w+\b', text.casefold())
    lengths = np.asarray([len(w) for w in words])
    n, w = max(len(text), 1), max(len(words), 1)
    sentence_count = max(len(re.findall(r'[.!?]+', text)), 1)
    result = [np.log1p(len(text)), np.log1p(len(words)),
              float(lengths.mean()) if len(lengths) else 0,
              float(lengths.std()) if len(lengths) else 0,
              sum(c.isupper() for c in text)/n, sum(c.isdigit() for c in text)/n,
              sum(c.isspace() for c in text)/n, text.count('\n')/n,
              sum(unicodedata.category(c).startswith('P') for c in text)/n,
              sum(ord(c) > 127 for c in text)/n, len(words)/sentence_count]
    result += [text.count(c)/n for c in '.,;:!?"\'()-']
    result += [words.count(word)/w for word in FUNCTION_WORDS]
    return np.asarray(result, dtype=np.float32)


@dataclass
class CandidateData:
    candidates: np.ndarray
    base_features: np.ndarray
    pairs: list
    owners: np.ndarray
    labels: np.ndarray


def build_candidates(index, targets, encoded_targets, retrieval, local, centroid,
                     top_candidates=20, references_per_candidate=3, force_training_positive=False):
    if min(top_candidates, references_per_candidate) < 1:
        raise ValueError('Candidate and reference budgets must be positive.')
    n = min(top_candidates, len(index.authors))
    candidates = descending_order(retrieval)[:, :n].copy()
    author_to_index = {a: i for i, a in enumerate(index.authors)}
    if force_training_positive:
        # Explicitly used ONLY for training pair mining; never on dev/cal/test.
        for i, author in enumerate(targets.author):
            true = author_to_index[author]
            if true not in candidates[i]:
                candidates[i, -1] = true
    ref_style = np.stack([stylometry(t) for t in index.refs.text])
    profiles = np.stack([ref_style[ids].mean(axis=0) for ids in index.indices])
    target_style = np.stack([stylometry(t) for t in targets.text])
    style_distance = np.abs(target_style[:, None, :] - profiles[candidates])
    features = np.concatenate([np.take_along_axis(s, candidates, axis=1)[..., None]
                               for s in [retrieval, local, centroid]] + [style_distance], axis=2)
    pairs, owners, labels = [], [], []
    for i, (_, row) in enumerate(targets.iterrows()):
        for j, author_index in enumerate(candidates[i]):
            refs = index.select_references(encoded_targets[i], author_index, references_per_candidate)
            for ref in refs:
                pairs.append((row.text, index.refs.iloc[ref].text))
                owners.append(i*n+j)
                labels.append(float(row.author == index.authors[author_index]))
    return CandidateData(candidates, features, pairs, np.asarray(owners), np.asarray(labels, dtype=np.float32))


def tokenize_pairs(tokenizer, pairs, device, max_length=512):
    # Balanced pair truncation; no author labels or external identity information.
    return tokenizer([p[0][:2000] for p in pairs], [p[1][:2000] for p in pairs],
                     padding=True, truncation='longest_first', max_length=max_length,
                     return_tensors='pt').to(device)


def cross_encoder_logits(model, tokenizer, pairs, device='cpu', batch_size=16, max_length=512):
    if batch_size < 1 or max_length < 8:
        raise ValueError('Invalid cross-encoder batch size or token budget.')
    model.eval(); output = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            encoded = tokenize_pairs(tokenizer, pairs[i:i+batch_size], device, max_length)
            logits = model(**encoded).logits
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise ValueError('Cross-encoder must have one binary compatibility logit.')
            output.append(logits[:, 0].float().cpu().numpy())
    return np.concatenate(output)


def candidate_features(data, pair_logits):
    if len(pair_logits) != len(data.pairs) or not np.isfinite(pair_logits).all():
        raise ValueError('Invalid cross-encoder pair logits.')
    count = data.candidates.size
    sums = np.bincount(data.owners, weights=pair_logits, minlength=count)
    counts = np.bincount(data.owners, minlength=count)
    if (counts == 0).any():
        raise ValueError('Every candidate needs at least one reference pair.')
    logits = (sums/counts).reshape(data.candidates.shape)
    features = np.concatenate([data.base_features[..., :3], logits[..., None],
                               data.base_features[..., 3:]], axis=-1)
    return features


def train_cross_encoder_epoch(model, tokenizer, data, optimizer, rng, device='cpu',
                              batch_size=16, max_length=512, max_pairs=100000):
    indices = rng.permutation(len(data.pairs))[:max_pairs]
    if len(np.unique(data.labels[indices])) < 2:
        raise ValueError('Cross-encoder training needs positive and negative pairs.')
    model.train(); losses = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start+batch_size]
        encoded = tokenize_pairs(tokenizer, [data.pairs[i] for i in batch], device, max_length)
        labels = torch.as_tensor(data.labels[batch], device=device)
        logits = model(**encoded).logits[:, 0]
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite cross-encoder loss.')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))
