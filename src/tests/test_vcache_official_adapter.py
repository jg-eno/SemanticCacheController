"""Tests for the official-vCache adapter's own logic.

They use stand-ins and need only numpy, so they do NOT exercise the official vCache
policy itself; that needs the real package (see BASELINE.md).
"""

import importlib.util
import threading
import time
import types
import unittest

from src.baseline_models.vcache_official import (INSTALL_HINT, OfficialVCacheSystem,
                                                 build_system, run_workload)


class FakeRewards:
    def components(self, status, recomputed):
        return dict(reuse=1.0 if status == "correct" else 0.0,
                    penalty=-2.0 if status == "incorrect" else 0.0,
                    cost=-0.2 if recomputed else 0.0)


class ScriptedSystem:
    """Returns scripted (is_hit, cached_id_set, explored) and records the id_sets it saw."""

    def __init__(self, script):
        self.script, self.seen = list(script), []

    def infer(self, prompt, id_set):
        self.seen.append((prompt, id_set))
        return self.script.pop(0)


def row(query, intent):
    return dict(query=query, intent_id=intent)


class RunWorkloadTests(unittest.TestCase):
    def test_counts_statuses_and_metrics(self):
        rows = [row("a", "X"), row("b", "X"), row("c", "Y"), row("d", "Y")]
        system = ScriptedSystem([
            (False, None, False),   # first request: nothing cached
            (True, 0, False),       # reuse, same intent -> correct
            (True, 0, False),       # reuse, different intent -> incorrect
            (False, None, True),    # explored and recomputed
        ])
        result = run_workload(system, rows, FakeRewards())
        self.assertEqual(result["requests"], 4)
        self.assertEqual(result["evaluable_labeled_requests"], 4)
        self.assertAlmostEqual(result["false_hit_rate"], 0.25)
        self.assertAlmostEqual(result["recomputation_rate"], 0.5)
        self.assertEqual(result["explorations"], 1)
        # not_reused -0.2, correct +1, incorrect -2, not_reused -0.2
        self.assertAlmostEqual(result["mean_environment_reward"], (-0.2 + 1 - 2 - 0.2) / 4)

    def test_same_intent_gets_same_integer_id(self):
        rows = [row("a", "X"), row("b", "Y"), row("c", "X")]
        system = ScriptedSystem([(False, None, False)] * 3)
        run_workload(system, rows, FakeRewards())
        self.assertEqual([i for _, i in system.seen], [0, 1, 0])

    def test_unlabeled_rows_are_unevaluable(self):
        rows = [row("a", None), row("b", None)]
        system = ScriptedSystem([(False, None, False), (True, -1, False)])
        result = run_workload(system, rows, FakeRewards())
        self.assertEqual(result["evaluable_labeled_requests"], 0)
        self.assertIsNone(result["false_hit_rate"])
        self.assertEqual([i for _, i in system.seen], [-1, -1])

    def test_empty_workload_is_rejected(self):
        with self.assertRaises(ValueError):
            run_workload(ScriptedSystem([]), [], FakeRewards())


class FakeQueue:
    unfinished_tasks = 0


class FakeVCache:
    """Reports an explore on every call and returns a real-looking nearest neighbour."""

    def infer_with_cache_info(self, prompt, system_prompt, id_set):
        nearest = types.SimpleNamespace(embedding_id=3, id_set=id_set)
        return False, "response", None, nearest


class WaitingTests(unittest.TestCase):
    def make(self, timeout=2.0):
        queue = FakeQueue()
        policy = types.SimpleNamespace(callback_queue=queue, shutdown=lambda: None)
        state = {"puts": 0, "lock": threading.Lock()}
        return OfficialVCacheSystem(FakeVCache(), policy, state, timeout), queue, state

    def test_waits_for_background_update_to_finish(self):
        system, queue, state = self.make()
        finished = []

        def background():
            time.sleep(0.05)
            with state["lock"]:
                state["puts"] += 1
                queue.unfinished_tasks = 1
            time.sleep(0.05)
            queue.unfinished_tasks = 0
            finished.append(True)

        threading.Thread(target=background).start()
        hit, cached, explored = system.infer("q", 0)
        self.assertEqual((hit, cached, explored), (False, None, True))
        self.assertTrue(finished)          # infer() returned only after the update completed

    def test_times_out_when_update_never_arrives(self):
        system, _, _ = self.make(timeout=0.05)
        with self.assertRaises(TimeoutError):
            system.infer("q", 0)


class ImportTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("vcache") is not None, "official vCache is installed")
    def test_missing_package_gives_install_hint(self):
        with self.assertRaises(ImportError) as context:
            build_system(encoder=None, delta=0.05)
        self.assertIn(INSTALL_HINT, str(context.exception))


if __name__ == "__main__":
    unittest.main()