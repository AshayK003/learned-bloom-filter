"""Complete benchmark: Classic BF, Static LBF, Streaming LBF."""

import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, 'src')

from features import build_feature_extractor
from baseline import ClassicBloomFilter
from static_lbf import StaticLearnedBloomFilter
from streaming import StreamingLearnedBloomFilter
from harness import load_urlhaus_data, generate_benign_urls, run_benchmark, print_results


def main():
    print("=" * 60)
    print("LABS 02: STREAMING LEARNED BLOOM FILTER — BENCHMARK")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    malicious = load_urlhaus_data('data/urlhaus_full.csv', max_rows=10000)
    benign = generate_benign_urls(10000)
    print(f"  Malicious: {len(malicious)} URLs")
    print(f"  Benign:    {len(benign)} URLs")

    # Split
    mal_train, mal_test = train_test_split(malicious, test_size=0.3, random_state=42)
    ben_train, ben_test = train_test_split(benign, test_size=0.3, random_state=42)
    print(f"\n[2/4] Split:")
    print(f"  Train: {len(mal_train)} mal, {len(ben_train)} ben")
    print(f"  Test:  {len(mal_test)} mal, {len(ben_test)} ben")

    # Build features
    print("\n[3/4] Building features...")
    feats = build_feature_extractor(ngram_range=(1, 5), max_features=3000)
    X_train = np.array(mal_train + ben_train)
    y_train = np.array([1] * len(mal_train) + [0] * len(ben_train))
    feats.fit(X_train)
    X_train_t = feats.transform(X_train)
    print(f"  Feature dim: {X_train_t.shape[1]}")

    results = []

    # 1. Classic Bloom Filter
    print("\n[4/4] Running benchmarks...\n")
    print("--- Classic Bloom Filter ---")
    cbf = ClassicBloomFilter(capacity=len(mal_train) * 2, error_rate=0.01)
    for url in mal_train:
        cbf.add(url)
    res = run_benchmark(cbf, "Classic BF", mal_train, mal_test, ben_test)
    results.append(res)
    print(f"  FPR: {res.fpr:.4f}, FNR: {res.fnr:.4f}, Mem: {res.memory_bytes/1024:.1f}KB")

    # 2. Static LBF
    print("\n--- Static LBF ---")
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
    slbf = StaticLearnedBloomFilter(
        classifier=clf,
        feature_extractor=feats,
        backup_capacity=len(mal_train),
        backup_fpr=0.001,
        confidence_threshold=0.9,
    )
    slbf.fit(X_train, y_train)
    for url in mal_train:
        slbf.add(url)
    res = run_benchmark(slbf, "Static LBF", mal_train, mal_test, ben_test)
    results.append(res)
    print(f"  FPR: {res.fpr:.4f}, FNR: {res.fnr:.4f}, Mem: {res.memory_bytes/1024:.1f}KB")
    print(f"  Classifier hits: {slbf.n_classifier_hits}, Backup hits: {slbf.n_backup_hits}")

    # 3. Streaming LBF (with retraining)
    print("\n--- Streaming LBF ---")
    clf_stream = SGDClassifier(
        loss='log_loss', class_weight='balanced', random_state=42, max_iter=5
    )
    slbf_stream = StreamingLearnedBloomFilter(
        classifier=clf_stream,
        feature_extractor=build_feature_extractor(ngram_range=(1, 5), max_features=3000),
        initial_capacity=len(mal_train),
        backup_error_rate=0.001,
        retrain_every=500,
        confidence_threshold=0.9,
        min_holdout_size=200,
    )
    slbf_stream.fit(X_train, y_train)

    # Add with labels
    for url in mal_train:
        slbf_stream.add(url, label=1)
    for url in ben_train:
        slbf_stream.add(url, label=0)

    res = run_benchmark(slbf_stream, "Streaming LBF", mal_train, mal_test, ben_test)
    results.append(res)
    print(f"  FPR: {res.fpr:.4f}, FNR: {res.fnr:.4f}, Mem: {res.memory_bytes/1024:.1f}KB")
    print(f"  Classifier hits: {slbf_stream.n_classifier_hits}, Backup hits: {slbf_stream.n_backup_hits}")
    print(f"  Retrain points: {len(slbf_stream.retrain_points)}")

    # Summary
    print("\n")
    print_results(results)


if __name__ == '__main__':
    main()
