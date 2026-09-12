"""30-day URLhaus replay: FPR grows as new URLs arrive.

Test: Add ALL URLs from ALL 30 days to all filters. Measure FPR on a fixed
benign test set. Classic BF has no false negatives by design, so we only
track FPR. Static/Streaming LBF have small FNR from classifier.
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from features import build_feature_extractor
from baseline import ClassicBloomFilter
from static_lbf import StaticLearnedBloomFilter
from streaming import StreamingLearnedBloomFilter
from harness import generate_benign_urls
from hard_benign import generate_hard_benign_urls


def load_urlhaus_by_date(filepath, max_days=30):
    date_urls = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('id,'):
                continue
            try:
                import csv
                reader = csv.reader([line])
                for row in reader:
                    if len(row) >= 4:
                        date_str = row[1][:10]
                        if row[3] == 'online':
                            date_urls[date_str].append(row[2])
            except:
                continue
    sorted_dates = sorted(date_urls.keys())[-max_days:]
    return {d: date_urls[d] for d in sorted_dates}


def run_replay(data_path, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("30-DAY URLHAUS REPLAY — FPR Over Time")
    print("=" * 70)

    date_urls = load_urlhaus_by_date(data_path)
    dates = sorted(date_urls.keys())
    print(f"  Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    total = sum(len(v) for v in date_urls.values())
    print(f"  Total URLs: {total}")

    # Fixed benign test set
    ben_test = generate_hard_benign_urls(3000, seed=42)

    # Train on day 1
    train_urls = date_urls[dates[0]]
    train_benign = generate_benign_urls(len(train_urls), seed=100)
    X_train = np.array(train_urls + train_benign)
    y_train = np.array([1]*len(train_urls) + [0]*len(train_benign))

    print(f"\n  Training on {dates[0]}: {len(train_urls)} malicious + {len(train_benign)} benign")

    # Initialize filters
    cbf = ClassicBloomFilter(capacity=3000, error_rate=0.01)

    feats_static = build_feature_extractor(ngram_range=(1,5), max_features=3000)
    slbf_static = StaticLearnedBloomFilter(
        LogisticRegression(C=100, class_weight='balanced', max_iter=1000),
        feats_static, backup_error_rate=0.001, confidence_threshold=0.9
    )
    slbf_static.fit(X_train, y_train)

    feats_stream = build_feature_extractor(ngram_range=(1,5), max_features=3000)
    slbf_stream = StreamingLearnedBloomFilter(
        LogisticRegression(C=100, class_weight='balanced', max_iter=1000),
        feats_stream, backup_error_rate=0.001, retrain_every=500,
        confidence_threshold=0.9, min_holdout_size=200
    )
    slbf_stream.fit(X_train, y_train)

    for url in train_urls:
        cbf.add(url)

    # Track cumulative results
    results = {'dates': [], 'classic_fpr': [], 'static_fpr': [], 'stream_fpr': [],
               'static_fnr': [], 'stream_fnr': [],
               'classic_mem': [], 'static_mem': [], 'stream_mem': []}

    print(f"\n  Adding URLs cumulatively, measuring FPR after each day:\n")

    for day_idx in range(1, len(dates)):
        date_str = dates[day_idx]
        day_urls = date_urls[date_str]
        if not day_urls:
            continue

        # Add day's URLs to all filters
        for url in day_urls:
            cbf.add(url)
            slbf_static.add(url)
            slbf_stream.add(url, label=1)

        # Measure FPR on fixed benign set
        c_fp = sum(1 for u in ben_test if cbf.query(u))
        s_fp = sum(1 for u in ben_test if slbf_static.query(u))
        st_fp = sum(1 for u in ben_test if slbf_stream.query(u))

        # FNR on training URLs (should stay low)
        s_fn = sum(1 for u in train_urls if not slbf_static.query(u))
        st_fn = sum(1 for u in train_urls if not slbf_stream.query(u))

        results['dates'].append(date_str)
        results['classic_fpr'].append(c_fp / len(ben_test))
        results['static_fpr'].append(s_fp / len(ben_test))
        results['stream_fpr'].append(st_fp / len(ben_test))
        results['static_fnr'].append(s_fn / len(train_urls))
        results['stream_fnr'].append(st_fn / len(train_urls))
        results['classic_mem'].append(cbf.size_bytes)
        results['static_mem'].append(slbf_static.size_bytes)
        results['stream_mem'].append(slbf_stream.total_size_bytes)

        print(f"  [{day_idx+1:2d}/30] {date_str} | +{len(day_urls):4d} URLs | "
              f"Classic FPR: {c_fp/len(ben_test):.4f} | "
              f"Static FPR: {s_fp/len(ben_test):.4f} FNR: {s_fn/len(train_urls):.4f} | "
              f"Stream FPR: {st_fp/len(ben_test):.4f} FNR: {st_fn/len(train_urls):.4f}")

    # Plot FPR
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax = axes[0]
    ax.plot(results['dates'], results['classic_fpr'], 'r-o', label='Classic BF', markersize=3)
    ax.plot(results['dates'], results['static_fpr'], 'b-s', label='Static LBF', markersize=3)
    ax.plot(results['dates'], results['stream_fpr'], 'g-^', label='Streaming LBF', markersize=3)
    ax.set_ylabel('False Positive Rate', fontsize=11)
    ax.set_title('FPR Growth Over 30 Days (cumulative insertions)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1]
    ax.plot(results['dates'], results['static_fnr'], 'b-s', label='Static LBF FNR', markersize=3)
    ax.plot(results['dates'], results['stream_fnr'], 'g-^', label='Streaming LBF FNR', markersize=3)
    ax.set_ylabel('False Negative Rate', fontsize=11)
    ax.set_xlabel('Date')
    ax.set_title('FNR on Training URLs (should stay ~0)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'replay_30day.png'), dpi=150, bbox_inches='tight')

    with open(os.path.join(output_dir, 'replay_30day.csv'), 'w') as f:
        f.write('date,classic_fpr,static_fpr,static_fnr,stream_fpr,stream_fnr\n')
        for i in range(len(results['dates'])):
            f.write(f"{results['dates'][i]},{results['classic_fpr'][i]:.6f},"
                    f"{results['static_fpr'][i]:.6f},{results['static_fnr'][i]:.6f},"
                    f"{results['stream_fpr'][i]:.6f},{results['stream_fnr'][i]:.6f}\n")

    print("\n" + "=" * 70)
    print(f"  Classic BF  — Final FPR: {results['classic_fpr'][-1]:.4f}")
    print(f"  Static LBF  — Final FPR: {results['static_fpr'][-1]:.4f}, FNR: {results['static_fnr'][-1]:.4f}")
    print(f"  Stream LBF  — Final FPR: {results['stream_fpr'][-1]:.4f}, FNR: {results['stream_fnr'][-1]:.4f}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    run_replay(sys.argv[1] if len(sys.argv) > 1 else 'data/urlhaus_full.csv')
