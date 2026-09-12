"""Benchmark harness for comparing bloom filter variants."""

import time
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    n_items: int = 0
    fpr: float = 0.0
    fnr: float = 0.0
    memory_bytes: int = 0
    query_latency_ns: float = 0.0
    insert_latency_ns: float = 0.0
    backup_hit_rate: float = 0.0
    classifier_hit_rate: float = 0.0
    retrain_count: int = 0


def load_urlhaus_data(filepath: str, max_rows: Optional[int] = None) -> List[str]:
    """Load URLhaus CSV and return malicious URLs (online only)."""
    urls = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Skip header
            if line.startswith('id,'):
                continue
            # Parse CSV line (handle quoted fields)
            # Format: id,dateadded,url,url_status,last_online,threat,tags,urlhaus_link,reporter
            # URL is field index 2
            # Use csv reader for proper parsing
            try:
                reader = csv.reader([line])
                for row in reader:
                    if len(row) >= 4:
                        url = row[2]
                        status = row[3]
                        if status == 'online':
                            urls.append(url)
                        if max_rows and len(urls) >= max_rows:
                            return urls
            except Exception:
                continue
    return urls


def generate_benign_urls(n: int, seed: int = 42) -> List[str]:
    """Generate synthetic benign URLs for FPR testing."""
    rng = np.random.RandomState(seed)
    tlds = ['com', 'org', 'net', 'io', 'dev', 'app', 'co', 'info', 'biz', 'us']
    words = [
        'google', 'facebook', 'amazon', 'microsoft', 'apple', 'netflix',
        'twitter', 'github', 'stackoverflow', 'wikipedia', 'reddit', 'linkedin',
        'medium', 'dev', 'blog', 'shop', 'store', 'news', 'mail', 'cloud',
        'api', 'cdn', 'static', 'assets', 'images', 'video', 'docs', 'support',
    ]

    urls = []
    for _ in range(n):
        scheme = 'https' if rng.random() > 0.2 else 'http'
        www = 'www.' if rng.random() > 0.5 else ''
        domain = rng.choice(words) + '.' + rng.choice(tlds)
        path_depth = rng.randint(0, 4)
        path = '/'.join(rng.choice(words, size=path_depth))
        url = f'{scheme}://{www}{domain}'
        if path:
            url += '/' + path
        urls.append(url)
    return urls


def evaluate_filter(filter_obj, malicious_test: List[str], benign_test: List[str]) -> Dict:
    """Evaluate a filter on test data."""
    y_true = np.array([1] * len(malicious_test) + [0] * len(benign_test))
    all_items = malicious_test + benign_test

    start = time.perf_counter_ns()
    y_pred = np.array([filter_obj.query(item) for item in all_items])
    elapsed = time.perf_counter_ns() - start

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    avg_latency = elapsed / len(all_items) if all_items else 0

    return {
        'fpr': fpr,
        'fnr': fnr,
        'query_latency_ns': avg_latency,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def run_benchmark(
    filter_obj,
    name: str,
    train_malicious: List[str],
    test_malicious: List[str],
    test_benign: List[str],
) -> BenchmarkResult:
    """Run a complete benchmark on a filter."""
    result = BenchmarkResult(name=name)

    # Insert training items
    start = time.perf_counter_ns()
    for url in train_malicious:
        filter_obj.add(url, label=1)
    insert_time = time.perf_counter_ns() - start
    result.insert_latency_ns = insert_time / len(train_malicious) if train_malicious else 0

    # Evaluate
    metrics = evaluate_filter(filter_obj, test_malicious, test_benign)
    result.fpr = metrics['fpr']
    result.fnr = metrics['fnr']
    result.query_latency_ns = metrics['query_latency_ns']
    result.n_items = len(train_malicious)

    # Memory
    if hasattr(filter_obj, 'total_size_bytes'):
        result.memory_bytes = filter_obj.total_size_bytes
    elif hasattr(filter_obj, 'size_bytes'):
        result.memory_bytes = filter_obj.size_bytes

    # Hit rates
    total_queries = len(test_malicious) + len(test_benign)
    if hasattr(filter_obj, 'n_backup_hits'):
        result.backup_hit_rate = filter_obj.n_backup_hits / total_queries if total_queries > 0 else 0
    if hasattr(filter_obj, 'n_classifier_hits'):
        result.classifier_hit_rate = filter_obj.n_classifier_hits / total_queries if total_queries > 0 else 0

    return result


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results as a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Filter':<25} {'Items':>8} {'FPR':>8} {'FNR':>8} {'Mem(KB)':>10} {'Q-μs':>8} {'I-μs':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<25} {r.n_items:>8} {r.fpr:>8.4f} {r.fnr:>8.4f} "
              f"{r.memory_bytes/1024:>10.1f} {r.query_latency_ns/1000:>8.1f} {r.insert_latency_ns/1000:>8.1f}")
    print("=" * 80)
