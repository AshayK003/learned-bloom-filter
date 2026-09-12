"""Static Learned Bloom Filter with integrated feature extraction."""

import numpy as np
from typing import Optional

from pybloom_live import ScalableBloomFilter


class StaticLearnedBloomFilter:
    """Static Learned Bloom Filter (Kraska et al. 2018) with integrated features."""

    def __init__(
        self,
        classifier,
        feature_extractor,
        backup_error_rate: float = 0.001,
        confidence_threshold: float = 0.9,
    ):
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.backup_error_rate = backup_error_rate
        self.confidence_threshold = confidence_threshold
        self.backup_bf = None
        self._n_items = 0
        self._n_backup_hits = 0
        self._n_classifier_hits = 0
        self._is_fitted = False

    def fit(self, X, y):
        """Train the classifier and build backup BF."""
        self.feature_extractor.fit(X)
        X_transformed = self.feature_extractor.transform(X)
        self.classifier.fit(X_transformed, y)

        # Initialize backup BF with scalable mode
        self.backup_bf = ScalableBloomFilter(
            initial_capacity=max(len(X), 100),
            error_rate=self.backup_error_rate,
        )

        # Add positive items to backup BF
        positives = X[y == 1] if len(np.unique(y)) > 1 else X
        for item in positives:
            self.backup_bf.add(item)

        self._is_fitted = True
        return self

    def query(self, item: str) -> bool:
        """Query: classifier first, then backup BF."""
        if not self._is_fitted:
            return False

        X_transformed = self.feature_extractor.transform([item])
        proba = self.classifier.predict_proba(X_transformed)[0]

        if proba[1] >= self.confidence_threshold:
            self._n_classifier_hits += 1
            return True
        elif proba[1] <= (1 - self.confidence_threshold):
            self._n_classifier_hits += 1
            return False
        else:
            self._n_backup_hits += 1
            return item in self.backup_bf

    def add(self, item: str, label: int = 1) -> None:
        """Add item to backup BF."""
        if label == 1:
            self.backup_bf.add(item)
        self._n_items += 1

    @property
    def n_items(self) -> int:
        return self._n_items

    @property
    def n_backup_hits(self) -> int:
        return self._n_backup_hits

    @property
    def n_classifier_hits(self) -> int:
        return self._n_classifier_hits

    @property
    def size_bytes(self) -> int:
        """Total size: classifier + backup BF."""
        clf_size = 0
        if hasattr(self.classifier, 'coef_'):
            clf_size += self.classifier.coef_.nbytes
        if hasattr(self.classifier, 'intercept_'):
            clf_size += self.classifier.intercept_.nbytes

        backup_size = sum(len(f.bitarray) // 8 for f in self.backup_bf.filters) if self.backup_bf else 0
        return clf_size + backup_size
