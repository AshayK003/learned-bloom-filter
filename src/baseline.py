"""Baseline bloom filter implementations for comparison."""

import math
from typing import Optional

from pybloom_live import BloomFilter, ScalableBloomFilter


class ClassicBloomFilter:
    """Standard Bloom Filter with optimal parameters."""

    def __init__(self, capacity: int, error_rate: float = 0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bf = BloomFilter(capacity=capacity, error_rate=error_rate)
        self._n_items = 0

    def add(self, item: str, label: int = 1) -> None:
        self.bf.add(item)
        self._n_items += 1

    def __contains__(self, item: str) -> bool:
        return item in self.bf

    def query(self, item: str) -> bool:
        return item in self.bf

    @property
    def n_items(self) -> int:
        return self._n_items

    @property
    def size_bytes(self) -> int:
        return len(self.bf.bitarray) // 8

    def current_fpr(self) -> float:
        """Estimate current FPR based on fill ratio."""
        bit_array = self.bf.bitarray
        m = len(bit_array)
        ones = sum(bit_array)
        fill_ratio = ones / m if m > 0 else 0
        k = self.bf.num_slices
        return (1 - math.exp(-k * self._n_items / m)) ** k if m > 0 else 0.0


class ScalableBloomFilterWrapper:
    """Wrapper around pybloom-live ScalableBloomFilter with tracking."""

    def __init__(self, initial_capacity: int = 10000, error_rate: float = 0.01):
        self.bf = ScalableBloomFilter(
            initial_capacity=initial_capacity,
            error_rate=error_rate,
            mode=ScalableBloomFilter.LARGE_SET_GROWTH,
        )
        self._n_items = 0

    def add(self, item: str, label: int = 1) -> None:
        self.bf.add(item)
        self._n_items += 1

    def __contains__(self, item: str) -> bool:
        return item in self.bf

    def query(self, item: str) -> bool:
        return item in self.bf

    @property
    def n_items(self) -> int:
        return self._n_items

    @property
    def size_bytes(self) -> int:
        """Total size of all filters in bytes."""
        return sum(len(f.bitarray) // 8 for f in self.bf.filters)
