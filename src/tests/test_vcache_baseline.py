"""Tests for the vCache baseline. They need only numpy (no gymnasium, no model download)."""

import unittest

import numpy as np

from src.baseline_models.caches import LRUSemanticCache
from src.baseline_models.vcache import (EntryModel, VCacheMethod, REUSE, RECOMPUTE,
                                        make_vcache_method)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class FakeEnv:
    """Minimal stand-in for SemanticCacheEnv's cache and label bookkeeping."""

    def __init__(self, workload, embeddings, capacity=64, threshold=0.85):
        self.workload, self.embeddings = workload, embeddings
        self.cache = LRUSemanticCache(capacity=capacity, similarity_threshold=threshold)
        self.cached_intents, self.index = {}, 0
        self.hits = self.false_hits = 0

    def prepare(self):
        row = self.workload[self.index]
        self.candidate, self.similarity = self.cache.lookup(self.embeddings[row["query"]])

    def step(self, action):
        row = self.workload[self.index]
        embedding = self.embeddings[row["query"]]
        if action == REUSE and self.candidate is not None:
            self.cache.record_reuse(self.candidate, row["step"])
            self.hits += 1
            self.false_hits += self.cached_intents[self.candidate] != row["intent_id"]
        else:
            entry, evicted = self.cache.admit(row["query"], embedding, "r",
                                              step=row["step"], intent_id=row["intent_id"])
            if evicted is not None:
                del self.cached_intents[evicted]
            self.cached_intents[entry] = row["intent_id"]
        self.index += 1


def confusable_workload(n, seed=0, cosine=0.90):
    """Intents A and B are confusable (cosine 0.90 > 0.85 cache threshold); C is distinct."""
    rng = np.random.default_rng(seed)
    angle = np.arccos(cosine)
    centers = {"A": unit([1, 0, 0]), "B": unit([np.cos(angle), np.sin(angle), 0]),
               "C": unit([0, 0, 1])}
    workload, embeddings = [], {}
    for i in range(n):
        intent = str(rng.choice(["A", "B", "C"], p=[0.4, 0.4, 0.2]))
        embeddings[f"q{i}"] = unit(centers[intent] + rng.normal(0, 0.02, 3))
        workload.append(dict(query=f"q{i}", intent_id=intent, step=i))
    return workload, embeddings


def run(method, workload, embeddings):
    env = FakeEnv(workload, embeddings)
    for _ in workload:
        env.prepare()
        if method is None:       # static threshold: always reuse a match
            action = REUSE if env.candidate is not None else RECOMPUTE
        else:
            action = method.select_action(None, env)
        env.step(action)
    return env


class EntryModelTests(unittest.TestCase):
    def test_no_observations_explores_with_probability_one_minus_delta(self):
        self.assertAlmostEqual(EntryModel().exploration_probability(0.9, 0.05), 0.95)

    def test_correct_observations_lower_exploration(self):
        model = EntryModel()
        for _ in range(30):
            model.add(0.95, True)
        self.assertLess(model.exploration_probability(0.95, 0.05), 0.2)

    def test_wrong_observations_keep_exploring(self):
        model = EntryModel()
        for _ in range(30):
            model.add(0.90, False)
        self.assertGreater(model.exploration_probability(0.90, 0.05), 0.9)

    def test_probability_is_valid_and_deterministic(self):
        model = EntryModel()
        for s, c in [(0.6, False), (0.8, False), (0.9, True), (0.95, True)]:
            model.add(s, c)
        first = model.exploration_probability(0.85, 0.1)
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(first, 1.0)
        self.assertEqual(first, model.exploration_probability(0.85, 0.1))


class VCacheMethodTests(unittest.TestCase):
    def test_rejects_invalid_delta(self):
        for delta in (0, 1, -0.1, 1.5):
            with self.assertRaises(ValueError):
                VCacheMethod(delta=delta)

    def test_recomputes_without_candidate(self):
        env = FakeEnv([], {})
        env.candidate, env.similarity = None, None
        self.assertEqual(make_vcache_method().select_action(None, env), RECOMPUTE)

    def test_same_seed_gives_same_result(self):
        workload, embeddings = confusable_workload(150)
        a = run(VCacheMethod(seed=3, delta=0.05), workload, embeddings)
        b = run(VCacheMethod(seed=3, delta=0.05), workload, embeddings)
        self.assertEqual((a.hits, a.false_hits), (b.hits, b.false_hits))

    def test_avoids_false_hits_that_a_static_threshold_makes(self):
        workload, embeddings = confusable_workload(400)
        static = run(None, workload, embeddings)
        vcache = run(VCacheMethod(seed=0, delta=0.05), workload, embeddings)
        self.assertGreater(static.false_hits / static.hits, 0.25)
        self.assertLessEqual(vcache.false_hits / max(vcache.hits, 1), 0.05)
        self.assertGreater(vcache.hits, 0.5 * static.hits)    # still reuses answers

    def test_smaller_delta_reuses_less(self):
        workload, embeddings = confusable_workload(400)
        strict = run(VCacheMethod(seed=0, delta=0.01), workload, embeddings)
        loose = run(VCacheMethod(seed=0, delta=0.2), workload, embeddings)
        self.assertLess(strict.hits, loose.hits)

    def test_inserted_duplicate_shares_parent_observations(self):
        workload, embeddings = confusable_workload(1)
        embeddings["q1"] = embeddings["q0"]
        workload.append(dict(query="q1", intent_id=workload[0]["intent_id"], step=1))
        workload.append(dict(query="q1", intent_id=workload[0]["intent_id"], step=2))
        method = VCacheMethod(seed=0, delta=0.05)
        env = FakeEnv(workload, embeddings)
        env.prepare(); env.step(method.select_action(None, env))      # miss: first entry
        env.prepare(); action = method.select_action(None, env)       # explore (no data yet)
        self.assertEqual(action, RECOMPUTE)
        env.step(action)                                              # inserts a duplicate
        env.prepare(); method.select_action(None, env)                # aliases the duplicate
        models = list(method.models.values())
        self.assertGreaterEqual(len(method.models), 2)
        self.assertTrue(all(m is models[0] for m in models))


if __name__ == "__main__":
    unittest.main()