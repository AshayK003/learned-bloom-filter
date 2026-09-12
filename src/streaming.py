"""Streaming Learned Bloom Filter with hybrid drift-triggered retraining.

Architecture:
  Tier 1: Classifier (Logistic Regression)
  Tier 2: Scalable Bloom Filter backup (auto-grows, no false negatives)
  Tier 3: Drift-triggered retraining (periodic)
"""

import numpy as np
from typing import Optional


class StreamingLearnedBloomFilter:
    """Streaming LBF with drift-triggered retraining."""

    def __init__(
        self,
        classifier,
        feature_extractor,
        backup_error_rate: float = 0.001,
        retrain_every: int = 1000,
        confidence_threshold: float = 0.9,
        min_holdout_size: int = 200,
    ):
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.backup_error_rate = backup_error_rate
        self.retrain_every = retrain_every
        self.confidence_threshold = confidence_threshold
        self.min_holdout_size = min_holdout_size

        self.backup_bf = None

        # Tracking
        self._n_items = 0
        self._n_backup_hits = 0
        self._n_classifier_hits = 0
        self._buffer_X = []
        self._buffer_y = []
        self._is_fitted = False

        # Retrain history
        self.retrain_points = []
        self.auc_history = []

    def fit(self, X, y):
        """Initial fit on training data."""
        self.feature_extractor.fit(X)
        X_transformed = self.feature_extractor.transform(X)
        self.classifier.fit(X_transformed, y)

        # Initialize backup BF with scalable mode
        from pybloom_live import ScalableBloomFilter
        self.backup_bf = ScalableBloomFilter(
            initial_capacity=max(len(X), 100),
            error_rate=self.backup_error_rate,
        )
        positives = X[y == 1] if len(np.unique(y)) > 1 else X
        for item in positives:
            self.backup_bf.add(item)

        self._is_fitted = True
        return self

    def partial_fit(self, X, y):
        """Incremental fit on new data (used internally during retraining)."""
        X_transformed = self.feature_extractor.transform(X)
        self.classifier.fit(X_transformed, y)
        return self

    def add(self, item: str, label: int = 1):
        """Add item to the filter."""
        self._buffer_X.append(item)
        self._buffer_y.append(label)

        if label == 1 and self.backup_bf:
            self.backup_bf.add(item)

        self._n_items += 1

        # Check if retraining needed
        if self._n_items % self.retrain_every == 0:
            self._maybe_retrain()

    def query(self, item: str) -> bool:
        """Query membership: classifier first, then backup BF."""
        if not self._is_fitted:
            return False

        proba = self._predict_proba(item)

        if proba >= self.confidence_threshold:
            self._n_classifier_hits += 1
            return True
        elif proba <= (1 - self.confidence_threshold):
            self._n_classifier_hits += 1
            return False
        else:
            self._n_backup_hits += 1
            if self.backup_bf:
                return item in self.backup_bf
            return False

    def _predict_proba(self, item: str) -> float:
        """Get probability of item being in the set."""
        X_transformed = self.feature_extractor.transform([item])
        return self.classifier.predict_proba(X_transformed)[0][1]

    def _maybe_retrain(self):
        """Check if retraining is needed and perform it."""
        if len(self._buffer_X) < self.min_holdout_size:
            return

        buffer_X_arr = np.array(self._buffer_X)
        buffer_y_arr = np.array(self._buffer_y)

        # Need both classes to retrain
        if len(np.unique(buffer_y_arr)) < 2:
            return

        self.partial_fit(buffer_X_arr, buffer_y_arr)
        self.retrain_points.append(self._n_items)

        # Clear buffer (keep some for holdout)
        keep = min(self.min_holdout_size, len(self._buffer_X))
        self._buffer_X = self._buffer_X[-keep:]
        self._buffer_y = self._buffer_y[-keep:]

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
    def backup_size_bytes(self) -> int:
        """Size of backup BF in bytes."""
        if self.backup_bf:
            return sum(len(f.bitarray) // 8 for f in self.backup_bf.filters)
        return 0

    @property
    def total_size_bytes(self) -> int:
        """Total memory usage estimate."""
        clf_size = 0
        if hasattr(self.classifier, 'coef_'):
            clf_size += self.classifier.coef_.nbytes
        if hasattr(self.classifier, 'intercept_'):
            clf_size += self.classifier.intercept_.nbytes
        return clf_size + self.backup_size_bytes
