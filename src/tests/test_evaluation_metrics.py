
import unittest

from src.evaluation_metrics import summarize


class EvaluationMetricsTests(unittest.TestCase):

    def setUp(self):
        # Three cache candidates are available.
        # Two are reused correctly/incorrectly/unevaluably;
        # the fourth request recomputes instead.
        self.records = [
            {
                "cache_hit": True,
                "effective_action": 0,
                "reuse_status": "correct",
            },
            {
                "cache_hit": True,
                "effective_action": 0,
                "reuse_status": "incorrect",
            },
            {
                "cache_hit": True,
                "effective_action": 0,
                "reuse_status": "unevaluable",
            },
            {
                "cache_hit": False,
                "effective_action": 1,
                "reuse_status": "not_reused",
            },
        ]

    def test_hit_and_correctness_metrics(self):
        result = summarize(self.records)

        self.assertEqual(result["query_count"], 4)
        self.assertAlmostEqual(result["semantic_hit_rate"], 0.75)
        self.assertAlmostEqual(result["served_hit_rate"], 0.75)
        self.assertAlmostEqual(
            result["quality_adjusted_hit_rate"], 0.25
        )
        self.assertAlmostEqual(result["hit_precision"], 1 / 3)
        self.assertAlmostEqual(result["false_hit_rate"], 1 / 3)
        self.assertAlmostEqual(
            result["unevaluable_hit_rate"], 1 / 3
        )

    def test_timing_is_excluded_by_default(self):
        result = summarize(
            self.records,
            [10, 20, 30, 40],
            [1, 2, 3, 4],
        )

        self.assertNotIn("latency_ms_mean", result)
        self.assertNotIn("decision_latency_ms_mean", result)

    def test_timing_metrics_when_enabled(self):
        result = summarize(
            self.records,
            [10, 20, 30, 40],
            [1, 2, 3, 4],
            include_timing=True,
        )

        self.assertAlmostEqual(result["latency_ms_mean"], 25)
        self.assertEqual(result["latency_ms_p50"], 20)
        self.assertEqual(result["latency_ms_p95"], 40)
        self.assertEqual(result["latency_ms_p99"], 40)

        self.assertAlmostEqual(
            result["decision_latency_ms_mean"], 2.5
        )
        self.assertEqual(result["decision_latency_ms_p50"], 2)
        self.assertEqual(result["decision_latency_ms_p95"], 4)
        self.assertEqual(result["decision_latency_ms_p99"], 4)

    def test_no_served_hits(self):
        records = [
            {
                "cache_hit": False,
                "effective_action": 1,
                "reuse_status": "not_reused",
            }
        ]

        result = summarize(records)

        self.assertEqual(result["served_hit_rate"], 0)
        self.assertIsNone(result["hit_precision"])
        self.assertIsNone(result["false_hit_rate"])

    def test_empty_input(self):
        self.assertEqual(summarize([]), {"query_count": 0})


if __name__ == "__main__":
    unittest.main()

