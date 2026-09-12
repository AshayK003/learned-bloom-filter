"""Parameter sweep for classifier optimization."""

import sys
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score
import itertools

sys.path.insert(0, 'src')

from features import build_feature_extractor
from harness import load_urlhaus_data, generate_benign_urls


def sweep():
    print("=" * 60)
    print("PARAMETER SWEEP")
    print("=" * 60)

    malicious = load_urlhaus_data('data/urlhaus_full.csv', max_rows=5000)
    benign = generate_benign_urls(5000)

    X = np.array(malicious + benign)
    y = np.array([1] * len(malicious) + [0] * len(benign))

    configs = [
        # (name, classifier, ngram_range, max_features)
        ("LR C=0.1", LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000), (1, 5), 3000),
        ("LR C=1.0", LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000), (1, 5), 3000),
        ("LR C=10", LogisticRegression(C=10, class_weight='balanced', max_iter=1000), (1, 5), 3000),
        ("LR C=100", LogisticRegression(C=100, class_weight='balanced', max_iter=1000), (1, 5), 3000),
        ("LR ngram=1-3", LogisticRegression(C=10, class_weight='balanced', max_iter=1000), (1, 3), 3000),
        ("LR ngram=1-4", LogisticRegression(C=10, class_weight='balanced', max_iter=1000), (1, 4), 3000),
        ("LR ngram=1-5", LogisticRegression(C=10, class_weight='balanced', max_iter=1000), (1, 5), 3000),
        ("LR maxf=5k", LogisticRegression(C=10, class_weight='balanced', max_iter=1000), (1, 5), 5000),
        ("LR maxf=10k", LogisticRegression(C=10, class_weight='balanced', max_iter=1000), (1, 5), 10000),
        ("SGD iter=100", SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42, max_iter=100), (1, 5), 3000),
        ("SGD iter=500", SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42, max_iter=500), (1, 5), 3000),
    ]

    results = []

    for name, clf, ngram, max_feat in configs:
        print(f"\n--- {name} (ngram={ngram}, maxf={max_feat}) ---")

        feats = build_feature_extractor(ngram_range=ngram, max_features=max_feat)
        feats.fit(X)
        X_t = feats.transform(X)

        # Cross-val AUC
        scores = cross_val_score(clf, X_t, y, cv=5, scoring='roc_auc', n_jobs=-1)
        mean_auc = scores.mean()
        std_auc = scores.std()

        # Also check FPR/FNR on holdout
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.3, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tp = np.sum((y_pred == 1) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        results.append((name, mean_auc, std_auc, fpr, fnr))
        print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Config':<20} {'AUC':>12} {'FPR':>10} {'FNR':>10}")
    print("-" * 80)
    results.sort(key=lambda x: -x[1])
    for name, auc, std, fpr, fnr in results:
        print(f"{name:<20} {auc:.4f}±{std:.4f} {fpr:>10.4f} {fnr:>10.4f}")

    print("\nBest:", results[0][0], f"(AUC={results[0][1]:.4f})")


if __name__ == '__main__':
    sweep()
