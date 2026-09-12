# CypherLabs Open Research — Labs 02

Streaming Learned Bloom Filters for URL blocklists.

**Status:** Research phase complete. Implementation starting.

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

See vault: [[01-Projects/Ideas/Labs 02 - Streaming Learned Bloom Filters]]

