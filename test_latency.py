"""Compare all feature extraction methods for latency."""

import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, 'src')

from features import build_feature_extractor
from fast_features import build_fast_feature_extractor
from numpy_lbf import NumpyStaticLBF
from harness import generate_benign_urls


def benchmark_extractor(extractor, name, n_queries=1000):
    """Measure per-query latency."""
    queries = generate_benign_urls(n_queries, seed=42)

    start = time.perf_counter_ns()
    for url in queries:
        extractor.transform([url])
    elapsed = time.perf_counter_ns() - start

    per_query_us = (elapsed / n_queries) / 1000
    print(f"  {name:<35} {per_query_us:>8.1f} μs/query")
    return per_query_us


def benchmark_lbf(lbf, name, n_queries=1000):
    """Measure per-query latency for a full LBF."""
    queries = generate_benign_urls(n_queries, seed=42)

    start = time.perf_counter_ns()
    for url in queries:
        lbf.query(url)
    elapsed = time.perf_counter_ns() - start

    per_query_us = (elapsed / n_queries) / 1000
    print(f"  {name:<35} {per_query_us:>8.1f} μs/query")
    return per_query_us


def main():
    print("=" * 70)
    print("QUERY LATENCY COMPARISON")
    print("=" * 70)

    # Generate training data
    mal = generate_benign_urls(5000, seed=1)  # proxy
    ben = generate_benign_urls(5000, seed=2)
    X = np.array(mal + ben)
    y = np.array([1] * len(mal) + [0] * len(ben))

    print("\n[1/3] Training all extractors...")

    # Slow: TfidfVectorizer
    slow_feats = build_feature_extractor(ngram_range=(1, 5), max_features=3000)
    slow_feats.fit(X)
    slow_clf = LogisticRegression(C=100, class_weight='balanced', max_iter=1000)
    slow_clf.fit(slow_feats.transform(X), y)

    # Fast: HashingVectorizer
    fast_feats = build_fast_feature_extractor(ngram_range=(1, 5), n_features=2048)
    fast_feats.fit(X)
    fast_clf = LogisticRegression(C=100, class_weight='balanced', max_iter=1000)
    fast_clf.fit(fast_feats.transform(X), y)

    # NumPy: pure Python features
    numpy_lbf = NumpyStaticLBF(n_buckets=256)
    numpy_lbf.fit(X, y)

    print("\n[2/3] Benchmarking query latency (1000 queries):\n")

    # Feature extraction only
    benchmark_extractor(slow_feats, "TfidfVectorizer (slow)")
    benchmark_extractor(fast_feats, "HashingVectorizer (fast)")

    print()

    # Full LBF query (extraction + classifier)
    # Slow LBF
    from static_lbf import StaticLearnedBloomFilter
    slow_lbf = StaticLearnedBloomFilter(slow_clf, slow_feats)
    slow_lbf.backup_bf = numpy_lbf.backup_bf
    benchmark_lbf(slow_lbf, "Full LBF (Tfidf)")

    # Fast LBF
    from static_lbf import StaticLearnedBloomFilter
    fast_lbf = StaticLearnedBloomFilter(fast_clf, fast_feats)
    fast_lbf.backup_bf = numpy_lbf.backup_bf
    benchmark_lbf(fast_lbf, "Full LBF (Hashing)")

    # NumPy LBF
    benchmark_lbf(numpy_lbf, "NumPy LBF (pure Python)")

    print("\n[3/3] Done.")


if __name__ == '__main__':
    main()
