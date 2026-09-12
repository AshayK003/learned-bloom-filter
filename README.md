# Streaming Learned Bloom Filters for Dynamic URL Blocklists

**Author:** [Ashay Kushwaha](https://github.com/AshayK003) ([CypherLabs](https://github.com/AshayK003))
**License:** [MIT](LICENSE)
**Citation:** [CITATION.cff](CITATION.cff)

---

## Problem

Static learned Bloom filters collapse once inserts arrive. Live URL blocklists grow daily. FPR decays toward 100% as the backup filter overfills.

## Approach

Hybrid Trigger + Scalable Backup:
- Scalable Bloom Filter as backup (no counter decrements = no false negatives)
- Drift-triggered classifier retraining
- Staggered frozen filters for historical evidence

## Stack

- Python 3.11+
- sklearn (classifier)
- pybloom-live (ScalableBloomFilter)
- URLhaus CSV (daily malicious URLs)

## Layout

```
src/
  features.py      # URL feature extraction
  baseline.py      # Classic + Static LBF baselines
  streaming.py     # Streaming LBF with drift trigger
  harness.py       # Benchmark harness
data/              # URLhaus snapshots (gitignored)
results/           # Frozen comparison tables (gitignored)
```

## Research

Deep research report: [Research.md](RESEARCH.md)
