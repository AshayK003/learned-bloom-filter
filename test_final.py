"""Final benchmark: NumPy LBF (fast) vs Classic BF."""

import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, 'src')

from numpy_lbf import NumpyStaticLBF
from baseline import ClassicBloomFilter
from harness import load_urlhaus_data, generate_benign_urls, run_benchmark, print_results
from hard_benign import generate_hard_benign_urls


def main():
    print("=" * 70)
    print("FINAL BENCHMARK: NumPy LBF vs Classic BF")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    malicious = load_urlhaus_data('data/urlhaus_full.csv', max_rows=10000)
    benign = generate_hard_benign_urls(10000, seed=42)
    print(f"  Malicious: {len(malicious)} URLs")
    print(f"  Benign: {len(benign)} URLs")

    # Split
    mal_train, mal_test = malicious[:7000], malicious[7000:]
    ben_train, ben_test = benign[:7000], benign[7000:]
    print(f"\n[2/4] Split:")
    print(f"  Train: {len(mal_train)} mal, {len(ben_train)} ben")
    print(f"  Test: {len(mal_test)} mal, {len(ben_test)} ben")

    # Train
    print("\n[3/4] Training...")
    X_train = np.array(mal_train + ben_train)
    y_train = np.array([1]*len(mal_train) + [0]*len(ben_train))

    # Classic BF
    cbf = ClassicBloomFilter(capacity=len(mal_train)*2, error_rate=0.01)
    for url in mal_train:
        cbf.add(url)

    # NumPy LBF
    numpy_lbf = NumpyStaticLBF(n_buckets=256)
    numpy_lbf.fit(X_train, y_train)

    results = []

    # Benchmark
    print("\n[4/4] Running benchmarks...\n")

    print("--- Classic Bloom Filter ---")
    res = run_benchmark(cbf, "Classic BF", mal_train, mal_test, ben_test)
    results.append(res)
    print(f"  FPR: {res.fpr:.4f}, FNR: {res.fnr:.4f}, Mem: {res.memory_bytes/1024:.1f}KB")

    print("\n--- NumPy LBF ---")
    res = run_benchmark(numpy_lbf, "NumPy LBF", mal_train, mal_test, ben_test)
    results.append(res)
    print(f"  FPR: {res.fpr:.4f}, FNR: {res.fnr:.4f}, Mem: {res.memory_bytes/1024:.1f}KB")
    print(f"  Classifier hits: {numpy_lbf.n_classifier_hits}, Backup hits: {numpy_lbf.n_backup_hits}")

    # Latency comparison
    print("\n--- Query Latency ---")
    import time
    queries = ben_test[:1000]

    start = time.perf_counter_ns()
    for q in queries:
        cbf.query(q)
    cbf_us = (time.perf_counter_ns() - start) / len(queries) / 1000

    start = time.perf_counter_ns()
    for q in queries:
        numpy_lbf.query(q)
    lbf_us = (time.perf_counter_ns() - start) / len(queries) / 1000

    print(f"  Classic BF:  {cbf_us:.1f} μs/query")
    print(f"  NumPy LBF:   {lbf_us:.1f} μs/query")
    print(f"  Overhead:    {lbf_us/cbf_us:.1f}x")

    print("\n")
    print_results(results)


if __name__ == '__main__':
    main()
