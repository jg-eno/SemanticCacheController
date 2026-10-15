"""
Adapter-focused regression tests for the semantic-cache baselines.

How to run this test file from the repo root:
    python -m unittest src.tests.test_baseline_caches -v

What this checks:
    - workload rows like {"query": ..., "step": ..., "intent_id": ...}
      are accepted by the LRU/LFU cache baselines
    - the cache converts those rows into the internal cache contract
      (embedding + cost + version + step)
    - stale-hit logic still works after the conversion
    - a missing encoder raises a clear ValueError instead of failing silently
"""

import unittest
import numpy as np
from src.baseline_models.caches import (FIFOSemanticCache, LFUSemanticCache, LIFOSemanticCache,
                                        LRUSemanticCache, NoCacheBaseline)
from src.baseline_models.evaluate_workload import CachedEncoder, evaluate_all


class FixedEncoder:
    def encode(self, query):
        return {
            "same-a": np.array([1.0, 0.0]),
            "same-b": np.array([0.99, 0.01]),
        }[query]


class BaselineWorkloadAdapterTests(unittest.TestCase):
    def test_lru_and_lfu_accept_workload_rows(self):
        rows = [
            {"query": "same-a", "intent_id": "intent-a", "step": 0, "true_cost": 2.0},
            {"query": "same-b", "intent_id": "intent-a", "step": 1},
            {"query": "same-a", "intent_id": "intent-b", "step": 2, "true_cost": 3.0},
        ]

        for cache_type in (LRUSemanticCache, LFUSemanticCache):
            with self.subTest(cache=cache_type.__name__):
                cache = cache_type(encoder=FixedEncoder(), default_true_cost=1.0)
                outcomes = [cache.process(row)[0] for row in rows]

                self.assertEqual(outcomes, ["miss", "hit", "hit"])
                # Row 2 hits an entry from a DIFFERENT intent: a wrong answer (false hit),
                # not a stale one. No content_version was supplied, so staleness is unknown.
                self.assertEqual(cache.metrics(), {
                    "n_queries": 3,
                    "hit_rate": 0.6667,
                    "stale_hit_rate": 0.0,
                    "normalized_api_cost": 0.3333,
                    "false_hit_rate": 0.3333,        # 1 incorrect / 3 evaluable requests
                    "recomputation_rate": 0.3333,
                    "evaluable_requests": 3,
                    "correct_reuse_count": 1,
                    "incorrect_reuse_count": 1,
                    "unevaluable_reuse_count": 0,
                })

    def test_workload_rows_require_encoder(self):
        cache = LRUSemanticCache()
        with self.assertRaisesRegex(ValueError, "encoder is required"):
            cache.process({"query": "text", "step": 0})


    def test_unlabeled_rows_are_unevaluable_not_clean(self):
        """WildChat-style rows (intent_id None): hits must be reported as unevaluable."""
        rows = [{"query": "same-a", "intent_id": None, "step": 0},
                {"query": "same-b", "intent_id": None, "step": 1}]
        cache = LRUSemanticCache(encoder=FixedEncoder())
        [cache.process(r) for r in rows]
        m = cache.metrics()
        self.assertEqual(m["unevaluable_reuse_count"], 1)
        self.assertIsNone(m["false_hit_rate"])      # unknown, not 0.0
        self.assertEqual(m["stale_hit_rate"], 0.0)

    def test_explicit_content_version_counts_as_stale(self):
        rows = [{"query": "same-a", "intent_id": "a", "content_version": "v1", "step": 0},
                {"query": "same-b", "intent_id": "a", "content_version": "v2", "step": 1}]
        cache = LRUSemanticCache(encoder=FixedEncoder())
        [cache.process(r) for r in rows]
        m = cache.metrics()
        self.assertEqual(m["stale_hit_rate"], 1.0)  # content changed under a correct hit
        self.assertEqual(m["false_hit_rate"], 0.0)  # ...so it is stale but NOT a false hit

    def test_nocache_has_same_metric_schema(self):
        cache = NoCacheBaseline()
        cache.process({"query": "x", "intent_id": "a", "step": 0})
        lru = LRUSemanticCache(encoder=FixedEncoder())
        lru.process({"query": "same-a", "intent_id": "a", "step": 0})
        self.assertEqual(set(cache.metrics()), set(lru.metrics()))
        self.assertEqual(cache.metrics()["recomputation_rate"], 1.0)


def _rec(vec, step):
    return {"embedding": np.array(vec, dtype=float), "true_cost": 1.0,
            "content_version": 0, "step": step}


class EvictionPolicyTests(unittest.TestCase):
    """Hand-built sequence where LRU and LFU must pick DIFFERENT victims.

    capacity=2, three orthogonal embeddings a, b, c (never similar to each other):
      0:a miss  1:a hit  2:a hit   -> a has 3 uses, last used at step 2
      3:b miss                     -> b has 1 use,  last used at step 3
      4:c miss -> eviction needed:  LRU drops a (oldest use), LFU drops b (fewest uses)
      5:a      -> LRU misses (a is gone), LFU hits (a was kept)
    """

    def run_policy(self, cache_type):
        a, b, c = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        cache = cache_type(capacity=2, similarity_threshold=0.85)
        seq = [a, a, a, b, c, a]
        return [cache.process(_rec(v, i))[0] for i, v in enumerate(seq)]

    def test_lru_evicts_least_recently_used(self):
        self.assertEqual(self.run_policy(LRUSemanticCache),
                         ["miss", "hit", "hit", "miss", "miss", "miss"])

    def test_lfu_evicts_least_frequently_used(self):
        self.assertEqual(self.run_policy(LFUSemanticCache),
                         ["miss", "hit", "hit", "miss", "miss", "hit"])


class InsertionOrderPolicyTests(unittest.TestCase):
    """FIFO / LIFO look only at insertion order. Hand-worked, capacity=2, a/b/c orthogonal.

    Sequence 1: a, b, a, c, b
      c arrives with the cache full ({a, b}, and a was just reused):
        FIFO drops a (oldest insert, reuse is ignored) -> final b HITS
        LRU drops b (a used more recently), LFU drops b (b used less), LIFO drops b (newest) -> final b MISSES
    Sequence 2: a, b, c, a
      c arrives with the cache full ({a, b}):
        LIFO drops b (newest insert) -> final a HITS
        FIFO drops a (oldest), LRU drops a (least recently used) -> final a MISSES
    """

    def outcomes(self, cache_type, seq):
        vecs = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}
        cache = cache_type(capacity=2, similarity_threshold=0.85)
        return [cache.process(_rec(vecs[k], i))[0] for i, k in enumerate(seq)]

    def test_fifo_ignores_reuse(self):
        self.assertEqual(self.outcomes(FIFOSemanticCache, "abacb"),
                         ["miss", "miss", "hit", "miss", "hit"])

    def test_other_policies_drop_b_in_sequence_1(self):
        for cls in (LRUSemanticCache, LFUSemanticCache, LIFOSemanticCache):
            with self.subTest(cache=cls.__name__):
                self.assertEqual(self.outcomes(cls, "abacb"),
                                 ["miss", "miss", "hit", "miss", "miss"])

    def test_lifo_keeps_oldest_entries(self):
        self.assertEqual(self.outcomes(LIFOSemanticCache, "abca"),
                         ["miss", "miss", "miss", "hit"])

    def test_fifo_and_lru_drop_a_in_sequence_2(self):
        for cls in (FIFOSemanticCache, LRUSemanticCache):
            with self.subTest(cache=cls.__name__):
                self.assertEqual(self.outcomes(cls, "abca"),
                                 ["miss", "miss", "miss", "miss"])

    def test_default_capacity_is_32(self):
        self.assertEqual(LRUSemanticCache().capacity, 32)


class EvaluateWorkloadTests(unittest.TestCase):
    def test_each_distinct_query_is_embedded_once_across_all_baselines(self):
        class Counting(FixedEncoder):
            calls = 0
            def encode(self, query):
                Counting.calls += 1
                return super().encode(query)

        rows = [{"query": "same-a", "intent_id": "a", "step": 0},
                {"query": "same-b", "intent_id": "a", "step": 1},
                {"query": "same-a", "intent_id": "a", "step": 2}]
        encoder = CachedEncoder(Counting())
        results = evaluate_all(rows, encoder)

        self.assertEqual(Counting.calls, 2)     # 2 distinct strings, not 3 rows x 2 caches
        self.assertEqual(set(results), {"NoCacheBaseline", "FIFOSemanticCache", "LIFOSemanticCache",
                                        "LRUSemanticCache", "LFUSemanticCache"})
        self.assertEqual(results["LRUSemanticCache"]["hit_rate"], 0.6667)


if __name__ == "__main__":
    unittest.main()