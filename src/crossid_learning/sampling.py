"""Cross-genre positive bundles and same-pseudo-topic negative author batches."""
from collections import defaultdict

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def pseudo_topics(texts, topics=32, seed=20261001):
    # Unsupervised content clusters fitted only on training documents. These
    # are proxy topics, not CrossNews ground-truth topic annotations.
    vectorizer = TfidfVectorizer(max_features=20000, stop_words='english', min_df=1)
    matrix = vectorizer.fit_transform(texts)
    count = min(topics, len(texts), matrix.shape[1])
    if count < 2:
        raise ValueError('Need at least two usable pseudo-topic clusters.')
    labels = MiniBatchKMeans(n_clusters=count, random_state=seed, n_init=3,
                            batch_size=1024).fit_predict(matrix)
    return labels, count


class BundleSampler:
    def __init__(self, frame, topics, seed=20261001):
        self.frame, self.topics = frame, np.asarray(topics)
        self.rng = np.random.default_rng(seed)
        self.by_author = defaultdict(dict)
        self.by_topic = defaultdict(set)
        for (author, genre), rows in frame.groupby(['author', 'genre']):
            self.by_author[author][genre] = rows.index.to_numpy()
        self.authors = sorted(a for a, groups in self.by_author.items()
                              if set(groups) == {'Article', 'Tweet'})
        if len(self.authors) < 2:
            raise ValueError('Need >=2 training authors with both genres.')
        for i, row in frame.iterrows():
            if row.author in self.authors:
                self.by_topic[int(topics[i])].add(row.author)

    def sample(self, authors_per_batch=16, max_bundle=8):
        if authors_per_batch < 2 or max_bundle < 1:
            raise ValueError('Need >=2 authors per batch and a positive bundle size.')
        first = self.rng.choice(self.authors)
        genre = self.rng.choice(['Article', 'Tweet'])
        anchor = int(self.rng.choice(self.by_author[first][genre]))
        topic = int(self.topics[anchor])
        hard = sorted(self.by_topic[topic] - {first})
        self.rng.shuffle(hard)
        chosen = [first] + hard[:authors_per_batch-1]
        available = sorted(set(self.authors) - set(chosen)); self.rng.shuffle(available)
        chosen += available[:max(0, authors_per_batch-len(chosen))]
        anchors, positives, bundles = [], [], []
        for author in chosen:
            indices = self.by_author[author][genre]
            hard_indices = indices[self.topics[indices] == topic]
            anchors.append(int(self.rng.choice(hard_indices if len(hard_indices) else indices)))
            other = self.by_author[author]['Tweet' if genre == 'Article' else 'Article']
            size = int(self.rng.integers(1, min(max_bundle, len(other))+1))
            bundle = self.rng.choice(other, size=size, replace=False)
            positives.append(int(bundle[0])); bundles.append(bundle)
        return np.asarray(anchors), np.asarray(positives), bundles
