"""
test_run.py

Sanity-check runner: exercises NoCache / LRU / LFU against the stub
data stream (test_stub_data.py) and prints their metrics.

Purpose: verify the baseline logic in caches.py behaves correctly --
e.g., hit rate is non-trivial, stale-hit-rate rises after a content
flip, LRU and LFU differ under a workload with recency/frequency skew,
NoCache always reports 0 hit rate / 1.0 normalized cost.

This is NOT the project's benchmarking run. Once the real gym is
available, this file should be replaced by a runner that consumes the
gym's output through the same `process(record)` interface.
"""

from caches import LRUSemanticCache, LFUSemanticCache, NoCacheBaseline
from test_stub_data import make_stub_stream


def run(cache, records):
    for record in records:
        cache.process(record)
    return cache.metrics()


if __name__ == "__main__":
    print("=== Scenario A: capacity (50) > clusters (10) -- no eviction pressure ===")
    print("(Confirms hit/miss/stale-tracking logic only; LRU==LFU is EXPECTED here.)")
    records = make_stub_stream(n_records=2000, n_clusters=10, content_flip_every=400, seed=0)
    caches = {
        "NoCache": NoCacheBaseline(),
        "LRU": LRUSemanticCache(capacity=50, similarity_threshold=0.85),
        "LFU": LFUSemanticCache(capacity=50, similarity_threshold=0.85),
    }
    print(f"{'Method':<10} {'HitRate':>10} {'StaleHit':>10} {'NormCost':>10}")
    print("-" * 42)
    for name, cache in caches.items():
        m = run(cache, records)
        print(f"{name:<10} {m['hit_rate']:>10.4f} {m['stale_hit_rate']:>10.4f} {m['normalized_api_cost']:>10.4f}")

    print("\n=== Scenario B: capacity (5) < clusters (20) -- forces eviction ===")
    print("(Confirms LRU and LFU actually differ once eviction is triggered.)")
    records_b = make_stub_stream(n_records=3000, n_clusters=20, content_flip_every=400, seed=1)
    caches_b = {
        "NoCache": NoCacheBaseline(),
        "LRU": LRUSemanticCache(capacity=5, similarity_threshold=0.85),
        "LFU": LFUSemanticCache(capacity=5, similarity_threshold=0.85),
    }
    print(f"{'Method':<10} {'HitRate':>10} {'StaleHit':>10} {'NormCost':>10}")
    print("-" * 42)
    for name, cache in caches_b.items():
        m = run(cache, records_b)
        print(f"{name:<10} {m['hit_rate']:>10.4f} {m['stale_hit_rate']:>10.4f} {m['normalized_api_cost']:>10.4f}")
