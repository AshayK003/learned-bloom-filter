"""Pure-NumPy learned bloom filter — fast queries with good accuracy.

Uses more hash buckets (2048) for better accuracy-latency tradeoff.
"""

import math
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression


class NumpyFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features as a dense NumPy array."""

    def __init__(self, ngram_range=(1, 5), n_buckets=2048):
        self.ngram_range = ngram_range
        self.n_buckets = n_buckets
        self.n_struct = 8

    def fit(self, X, y=None):
        return self

    def transform(self, urls):
        n = len(urls)
        features = np.zeros((n, self.n_buckets + self.n_struct), dtype=np.float32)
        for i, url in enumerate(urls):
            features[i] = self._extract(url)
        return features

    def _extract(self, url):
        vec = np.zeros(self.n_buckets + self.n_struct, dtype=np.float32)

        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(url) - n + 1):
                gram = url[i:i + n]
                h = hash(gram) % self.n_buckets
                vec[h] += 1

        url_len = len(url) if url else 1
        vec[self.n_buckets + 0] = url_len
        vec[self.n_buckets + 1] = url.count('.')
        vec[self.n_buckets + 2] = url.count('/')
        vec[self.n_buckets + 3] = url.count('-')
        vec[self.n_buckets + 4] = sum(c.isdigit() for c in url) / url_len
        vec[self.n_buckets + 5] = url.count(':')
        vec[self.n_buckets + 6] = int(url.startswith('http://'))
        vec[self.n_buckets + 7] = int(url.startswith('https://'))

        return vec


class NumpyStaticLBF:
    """Static LBF with pure-NumPy features."""

    def __init__(self, n_buckets=2048, backup_error_rate=0.001):
        self.extractor = NumpyFeatureExtractor(n_buckets=n_buckets)
        self.classifier = LogisticRegression(C=100, class_weight='balanced', max_iter=1000)
        self.backup_bf = None
        self.backup_error_rate = backup_error_rate
        self._n_items = 0
        self._n_backup_hits = 0
        self._n_classifier_hits = 0

    def fit(self, X, y):
        self.extractor.fit(X)
        X_t = self.extractor.transform(np.array(X))
        self.classifier.fit(X_t, y)

        from pybloom_live import BloomFilter
        positives = [X[i] for i in range(len(X)) if y[i] == 1]
        self.backup_bf = BloomFilter(capacity=max(len(positives), 100), error_rate=self.backup_error_rate)
        for item in positives:
            self.backup_bf.add(item)
        return self

    def query(self, item):
        X_t = self.extractor.transform([item])
        proba = self.classifier.predict_proba(X_t)[0]

        if proba[1] >= 0.9:
            self._n_classifier_hits += 1
            return True
        elif proba[1] <= 0.1:
            self._n_classifier_hits += 1
            return False
        else:
            self._n_backup_hits += 1
            return item in self.backup_bf

    def add(self, item, label=1):
        if label == 1 and self.backup_bf:
            try:
                self.backup_bf.add(item)
            except IndexError:
                from pybloom_live import BloomFilter
                new_bf = BloomFilter(capacity=self.backup_bf.capacity * 2, error_rate=self.backup_error_rate)
                self.backup_bf = new_bf
                new_bf.add(item)
        self._n_items += 1

    @property
    def n_items(self): return self._n_items
    @property
    def n_backup_hits(self): return self._n_backup_hits
    @property
    def n_classifier_hits(self): return self._n_classifier_hits
    @property
    def size_bytes(self):
        clf_size = self.classifier.coef_.nbytes if hasattr(self.classifier, 'coef_') else 0
        bf_size = len(self.backup_bf.bitarray) // 8 if self.backup_bf else 0
        return clf_size + bf_size
