import unittest
import numpy as np

from src.env.semantic_cache_env import FEATURE_NAMES, SemanticCacheEnv
from src.models.semantic_cache import FIFOCache


class DummyEncoder:
    """Deterministic encoder for testing without loading an ML model."""

    def encode(self, query):
        vectors = {
            "A": [1.0, 0.0],
            "B": [0.0, 1.0],
            "C": [-1.0, 0.0],
        }
        return np.array(vectors[query], dtype=np.float32)



def make_workload():
    queries = ["A", "B", "A", "C"]

    return [
        {
            "request_id": str(i),
            "intent_id": query,
            "query": query,
            "topic_id": None,
            "phase_id": 0,
            "step": i,
        }
        for i, query in enumerate(queries)
    ]


    return [
        {
            "request_id": str(i),
            "query": query,
            "intent_id": query,
            "phase_id": "test",
            "step": i,
        }
        for i, query in enumerate(queries)
    ]


class CachePolicyIntegrationTests(unittest.TestCase):

    def test_all_policies_run(self):
        for policy in ("fifo", "lifo", "lru", "lfu"):
            with self.subTest(policy=policy):
                env = SemanticCacheEnv(
                    workload=make_workload(),
                    encoder=DummyEncoder(),
                    capacity=2,
                    threshold=0.85,
                    cache_policy=policy,
                )

                observation, _ = env.reset()

                self.assertEqual(observation.shape, (len(FEATURE_NAMES),))

                for _ in range(4):
                    observation, reward, terminated, truncated, info = env.step(0)

                    self.assertTrue(np.isfinite(reward))
                    self.assertIn("cache_hit", info)

                    if terminated or truncated:
                        break

                self.assertTrue(env.done)

    def test_default_policy_remains_fifo(self):
        env = SemanticCacheEnv(
            workload=make_workload(),
            encoder=DummyEncoder(),
        )

        env.reset()

        self.assertIsInstance(env.cache, FIFOCache)

    def test_skip_admission_recomputes_without_inserting_for_one_step(self):
        env = SemanticCacheEnv(workload=make_workload(), encoder=DummyEncoder(), capacity=4,
                               threshold=0.85)
        env.reset()
        env.step(1)                                  # recompute: inserts "A"
        self.assertEqual(len(env.cache.entries), 1)
        env.skip_admission = True
        _, _, _, _, info = env.step(1)               # recompute "B" without inserting
        self.assertEqual(len(env.cache.entries), 1)
        self.assertEqual(info["recompute_count"], 2)
        self.assertFalse(env.skip_admission)         # the flag lasts one step only
        env.step(1)                                  # the next recompute inserts again
        self.assertEqual(len(env.cache.entries), 2)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            SemanticCacheEnv(
                workload=make_workload(),
                encoder=DummyEncoder(),
                cache_policy="unknown",
            )


if __name__ == "__main__":
    unittest.main()