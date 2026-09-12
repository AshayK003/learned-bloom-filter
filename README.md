# Streaming Learned Bloom Filters for Dynamic URL Blocklists

**Author:** [Ashay Kushwaha](https://github.com/AshayK003) ([CypherLabs](https://github.com/AshayK003))
**License:** [MIT](LICENSE)
**Report:** [internals/report.pdf](internals/report.pdf)

---

## TL;DR

Classic Bloom Filters cannot generalize to unseen URLs — they produce false negatives for any URL not in their exact training set. **Learned Bloom Filters reduce false negative rate by 50× (from 99.93% to 1.90%)** by learning URL patterns, at a latency cost of ~1000×.

**Read the full report:** [internals/report.pdf](internals/report.pdf)

---

## Problem

URL blocklists must answer: "Is this URL malicious?" Current solutions face a tradeoff:

| Approach | Problem |
|----------|---------|
| Classic Bloom Filter | No generalization — fails on unseen URLs (99.93% FNR) |
| Database lookup | Too slow for real-time (milliseconds per query) |
| Full ML classifier | Too heavy for edge deployment |

## Solution

Learned Bloom Filters combine a lightweight classifier with a small backup Bloom filter:

1. **Classifier** learns patterns in malicious URLs (IP-based, high entropy, suspicious TLDs)
2. **Backup Bloom filter** catches classifier false negatives
3. **Query:** classifier first (fast path), backup BF only for uncertain queries

## Results

Benchmarked on **10,000+ URLs** from [URLhaus](https://urlhaus.abuse.ch) (30 days, Aug–Sep 2026):

| Filter | FPR | FNR | Memory | Query Latency |
|--------|-----|-----|--------|---------------|
| **Classic BF** | 0.03% | 99.93% | 16.4 KB | 2.0 μs |
| **Static LBF (Tfidf)** | 0.00% | **1.90%** | 48.6 KB | 1959 μs |
| **NumPy LBF (Hash)** | 0.03% | 4.27% | 20.3 KB | 236 μs |

**Key findings:**
- **50× FNR reduction** with Tfidf-based LBF
- **Backup BF rarely used** (only 56-82 of 6000 queries)
- **Classifier handles 95%+** of queries confidently

## Quick Start

```bash
# Clone
git clone https://github.com/AshayK003/learned-bloom-filter
cd learned-bloom-filter

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download data
curl -A "Mozilla/5.0" https://urlhaus.abuse.ch/downloads/csv/ -o data/urlhaus_raw

# Run benchmarks
python test_final.py
python test_latency.py
python sweep.py
python replay_30day.py
```

## Repository Structure

```
learned-bloom-filter/
├── internals/
│   └── report.pdf              # Full research report (read this!)
├── src/
│   ├── baseline.py             # Classic Bloom Filter
│   ├── static_lbf.py           # Static Learned BF (Tfidf)
│   ├── streaming.py            # Streaming LBF
│   ├── numpy_lbf.py            # Fast NumPy LBF
│   ├── features.py             # Feature extraction
│   ├── hard_benign.py          # Test data generation
│   └── harness.py              # Benchmark framework
├── test_final.py               # Main benchmark
├── test_latency.py             # Latency comparison
├── sweep.py                    # Parameter sweep
├── replay_30day.py             # 30-day replay
├── data/                       # URLhaus data (gitignored)
├── results/                    # Benchmark results (gitignored)
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Feature Extraction

Two strategies are compared:

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **TfidfVectorizer** | 1959 μs/query | 1.90% FNR | Accuracy-critical |
| **NumPy Hashing** | 236 μs/query | 4.27% FNR | Speed-critical |

## Citation

```bibtex
@software{kushwaha2026,
  author = {Kushwaha, Ashay},
  title = {Streaming Learned Bloom Filters for Dynamic URL Blocklists},
  year = {2026},
  url = {https://github.com/AshayK003/learned-bloom-filter},
}
```

## Related Work

1. **Kraska et al. 2018** — The Case for Learned Index Structures (SIGMOD)
2. **Mitzenmacher 2018** — A Model for Learned Bloom Filters (NeurIPS)
3. **Liu et al. 2020** — Stable Learned Bloom Filters for Data Streams (PVLDB)
4. **Bäumel et al. 2021** — Adaptive Learned Bloom Filters (CIKM)
5. **Malchiodi et al. 2025** — Classifiers and Data Complexity in LBF (J. Big Data)

## License

MIT License. Copyright (c) 2026 Ashay Kushwaha (CypherLabs).

---

**Keywords:** bloom filter, learned index, URL blocklist, streaming algorithms, approximate membership, machine learning, cybersecurity, open source
