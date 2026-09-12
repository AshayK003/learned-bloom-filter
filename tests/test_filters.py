"""Unit tests for filter implementations."""

import pytest
import numpy as np
from src.baseline import ClassicBloomFilter, ScalableBloomFilterWrapper
from src.static_lbf import StaticLearnedBloomFilter
from src.numpy_lbf import NumpyStaticLBF
from src.features import build_feature_extractor


class TestClassicBloomFilter:
    """Tests for ClassicBloomFilter."""

    def test_insert_and_query(self):
        bf = ClassicBloomFilter(capacity=100, error_rate=0.01)
        bf.add("http://example.com")
        assert bf.query("http://example.com") is True

    def test_negative_query(self):
        bf = ClassicBloomFilter(capacity=100, error_rate=0.01)
        bf.add("http://example.com")
        assert bf.query("http://other.com") is False

    def test_n_items(self):
        bf = ClassicBloomFilter(capacity=100, error_rate=0.01)
        bf.add("a")
        bf.add("b")
        assert bf.n_items == 2

    def test_size_bytes(self):
        bf = ClassicBloomFilter(capacity=100, error_rate=0.01)
        assert bf.size_bytes > 0


class TestScalableBloomFilterWrapper:
    """Tests for ScalableBloomFilterWrapper."""

    def test_insert_and_query(self):
        bf = ScalableBloomFilterWrapper(initial_capacity=100, error_rate=0.01)
        bf.add("http://example.com")
        assert bf.query("http://example.com") is True

    def test_negative_query(self):
        bf = ScalableBloomFilterWrapper(initial_capacity=100, error_rate=0.01)
        bf.add("http://example.com")
        assert bf.query("http://other.com") is False

    def test_n_items(self):
        bf = ScalableBloomFilterWrapper(initial_capacity=100, error_rate=0.01)
        bf.add("a")
        bf.add("b")
        assert bf.n_items == 2


class TestStaticLearnedBloomFilter:
    """Tests for StaticLearnedBloomFilter."""

    def _make_lbf(self, n_samples=200):
        feats = build_feature_extractor(ngram_range=(1, 3), max_features=500)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1, class_weight='balanced', max_iter=500)
        lbf = StaticLearnedBloomFilter(clf, feats, backup_error_rate=0.01)
        X = [f"http://192.168.1.{i}/mal" for i in range(n_samples // 2)] + \
            [f"http://example{i}.com" for i in range(n_samples // 2)]
        y = [1] * (n_samples // 2) + [0] * (n_samples // 2)
        lbf.fit(np.array(X), np.array(y))
        return lbf

    def test_fit_and_query(self):
        lbf = self._make_lbf()
        assert lbf.query("http://192.168.1.5/mal") is True

    def test_negative_query(self):
        lbf = self._make_lbf()
        assert lbf.query("http://google.com") is False

    def test_n_items(self):
        lbf = self._make_lbf()
        lbf.add("http://192.168.1.100/mal")
        assert lbf.n_items == 1

    def test_size_bytes(self):
        lbf = self._make_lbf()
        assert lbf.size_bytes > 0


class TestNumpyStaticLBF:
    """Tests for NumpyStaticLBF."""

    def _make_lbf(self, n_samples=200):
        lbf = NumpyStaticLBF(n_buckets=256, backup_error_rate=0.01)
        X = [f"http://192.168.1.{i}/mal" for i in range(n_samples // 2)] + \
            [f"http://example{i}.com" for i in range(n_samples // 2)]
        y = [1] * (n_samples // 2) + [0] * (n_samples // 2)
        lbf.fit(np.array(X), np.array(y))
        return lbf

    def test_fit_and_query(self):
        lbf = self._make_lbf()
        assert lbf.query("http://192.168.1.5/mal") is True

    def test_negative_query(self):
        lbf = self._make_lbf()
        assert lbf.query("http://google.com") is False

    def test_n_items(self):
        lbf = self._make_lbf()
        lbf.add("http://192.168.1.100/mal")
        assert lbf.n_items == 1

    def test_size_bytes(self):
        lbf = self._make_lbf()
        assert lbf.size_bytes > 0
