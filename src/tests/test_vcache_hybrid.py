"""Tests for the vCache hybrid method.

Most tests use a stand-in for the official algorithm and need only numpy. One test class
runs the REAL official algorithm and is skipped unless the `vcache` package is importable.
"""

import importlib.util
import types
import unittest

import numpy as np

from src.baseline_models.vcache_hybrid import (INSTALL_HINT, OfficialDecisionMethod, RECOMPUTE,
                                               REUSE, load_official, make_official_decision_method)
from src.baseline_models.caches import LRUSemanticCache


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


class StandInMetadata:
    def __init__(self, embedding_id, response):
        self.embedding_id, self.response = embedding_id, response
        self.observations = []


class ScriptedAlgorithm:
    """Stands in for the official _Algorithm: returns scripted actions, records calls."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.selected, self.observed = [], []

    def select_action(self, similarity_score, metadata):
        self.selected.append((similarity_score, metadata))
        return types.SimpleNamespace(value=self.actions.pop(0))

    def add_observation_to_metadata(self, similarity_score, is_correct, metadata):
        self.observed.append((similarity_score, is_correct, metadata))
        metadata.observations.append((similarity_score, int(is_correct)))


def method_with(actions, seed=0):
    algorithm = ScriptedAlgorithm(actions)
    return OfficialDecisionMethod(seed=seed, delta=0.05, algorithm=algorithm,
                                  metadata_class=StandInMetadata), algorithm


def two_request_env(second_intent):
    workload, embeddings = confusable_workload(1)
    first = workload[0]["intent_id"]
    embeddings["q1"] = embeddings["q0"]
    workload.append(dict(query="q1", intent_id=second_intent or first, step=1))
    return FakeEnv(workload, embeddings), workload


class DecisionMappingTests(unittest.TestCase):
    def test_rejects_invalid_delta(self):
        for delta in (0, 1, -0.1, 1.5):
            with self.assertRaises(ValueError):
                OfficialDecisionMethod(delta=delta, algorithm=object(), metadata_class=object)

    def test_recomputes_without_candidate_and_does_not_call_algorithm(self):
        method, algorithm = method_with([])
        env = FakeEnv([], {})
        env.candidate, env.similarity = None, None
        self.assertEqual(method.select_action(None, env), RECOMPUTE)
        self.assertEqual(algorithm.selected, [])

    def test_exploit_reuses_and_records_nothing(self):
        method, algorithm = method_with(["exploit"])
        env, _ = two_request_env(None)
        env.prepare(); env.step(RECOMPUTE)
        env.prepare()
        self.assertEqual(method.select_action(None, env), REUSE)
        self.assertEqual(algorithm.observed, [])
        self.assertEqual((method.exploited, method.explored), (1, 0))

    def test_explore_recomputes_and_records_a_correct_observation(self):
        method, algorithm = method_with(["explore"])
        env, _ = two_request_env(None)                  # same intent as the cached entry
        env.prepare(); env.step(RECOMPUTE)
        env.prepare()
        self.assertEqual(method.select_action(None, env), RECOMPUTE)
        similarity, correct, _ = algorithm.observed[0]
        self.assertTrue(correct)
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_explore_records_an_incorrect_observation_for_a_different_intent(self):
        method, algorithm = method_with(["explore"])
        env, workload = two_request_env(None)
        workload[1]["intent_id"] = "OTHER"
        env.prepare(); env.step(RECOMPUTE)
        env.prepare()
        method.select_action(None, env)
        self.assertFalse(algorithm.observed[0][1])

    def test_unlabeled_request_records_nothing(self):
        method, algorithm = method_with(["explore"])
        env, workload = two_request_env(None)
        workload[1]["intent_id"] = None
        env.prepare(); env.step(RECOMPUTE)
        env.prepare()
        self.assertEqual(method.select_action(None, env), RECOMPUTE)
        self.assertEqual(algorithm.observed, [])

    def test_entry_keeps_one_metadata_object(self):
        method, algorithm = method_with(["explore", "explore"])
        workload, embeddings = confusable_workload(1)
        for i in (1, 2):
            embeddings[f"q{i}"] = embeddings["q0"]
            workload.append(dict(query=f"q{i}", intent_id=workload[0]["intent_id"], step=i))
        env = FakeEnv(workload, embeddings)
        env.prepare(); env.step(method.select_action(None, env))        # cold miss
        env.prepare(); env.step(method.select_action(None, env))        # explore, correct
        env.prepare(); method.select_action(None, env)                  # duplicate is aliased
        first, second = algorithm.selected[0][1], algorithm.selected[1][1]
        self.assertIs(first, second)                                    # parent and duplicate share
        # one observation from the first explore, one from the second on the shared object
        self.assertEqual(len(first.observations), 2)


class WholeRunTests(unittest.TestCase):
    def test_rejects_a_missing_official_package_with_a_hint(self):
        if importlib.util.find_spec("vcache") is not None:
            self.skipTest("official vCache is installed")
        with self.assertRaises(ImportError) as context:
            make_official_decision_method()
        self.assertIn(INSTALL_HINT, str(context.exception))


@unittest.skipIf(importlib.util.find_spec("vcache") is None, "official vCache is not installed")
class RealOfficialAlgorithmTests(unittest.TestCase):
    def test_same_seed_gives_same_result(self):
        workload, embeddings = confusable_workload(150)
        a = run(make_official_decision_method(seed=3, delta=0.05), workload, embeddings)
        b = run(make_official_decision_method(seed=3, delta=0.05), workload, embeddings)
        self.assertEqual((a.hits, a.false_hits), (b.hits, b.false_hits))

    def test_avoids_false_hits_that_a_static_threshold_makes(self):
        workload, embeddings = confusable_workload(600)
        static = run(None, workload, embeddings)
        hybrid = run(make_official_decision_method(seed=0, delta=0.05), workload, embeddings)
        self.assertGreater(static.false_hits / static.hits, 0.25)
        self.assertLess(hybrid.false_hits / max(hybrid.hits, 1),
                        static.false_hits / static.hits)

    def test_loader_returns_official_classes(self):
        algorithm_class, metadata_class = load_official()
        self.assertEqual(algorithm_class.__name__, "_Algorithm")
        self.assertEqual(metadata_class.__name__, "EmbeddingMetadataObj")


if __name__ == "__main__":
    unittest.main()