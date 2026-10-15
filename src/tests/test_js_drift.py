import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from src.models.query_drift import ProjectionHistogramDrift, js_divergence
from src.semantic_cache import SemanticCacheEnv
from src.tests.test_query_drift import Encoder, workload
from src.evaluate import compare, drift_rule


class JSDriftTests(unittest.TestCase):
    def test_divergence_math_and_sparse_bins(self):
        self.assertEqual(js_divergence([1, 0], [1, 0]), 0)
        self.assertEqual(js_divergence([1, 0], [0, 1]), 1)
        expected = 0.31127812445913283
        self.assertAlmostEqual(js_divergence([1, 0], [0.5, 0.5]), expected)
        self.assertAlmostEqual(js_divergence([5, 0], [1, 1]), expected)
        self.assertAlmostEqual(js_divergence([1, 1], [5, 0]), expected)
        for p, q in [([0, 0], [1, 0]), ([-1, 2], [1, 0]), ([np.nan, 0], [1, 0]), ([1], [1, 0])]:
            with self.assertRaises(ValueError): js_divergence(p, q)

    def test_stable_changed_and_reordered_windows(self):
        detector = ProjectionHistogramDrift(window=2)
        a, b = np.array([1, 0]), np.array([-1, 0])
        history = [a, a, a]
        self.assertEqual(detector.score(history, a), 0)
        self.assertEqual(len(history), 3)
        self.assertEqual(detector.score([a, a, b], b), 1)
        self.assertEqual(detector.score([a, b, b], a), 0)  # Same distribution, different ordering.
        score = detector.score([a, a, a], b)
        self.assertAlmostEqual(score, 0.31127812445913283)
        self.assertTrue(0 <= score <= 1)

    def test_warmup_configuration_and_reproducibility(self):
        detector = ProjectionHistogramDrift(window=2, seed=17)
        for history in ([], [[1, 0]], [[1, 0], [-1, 0]]):
            self.assertEqual(detector.score(history, [-1, 0]), 0)
        other = ProjectionHistogramDrift(window=2, seed=17)
        self.assertEqual(detector.score([[1, 0]] * 3, [-1, 0]), other.score([[1, 0]] * 3, [-1, 0]))
        np.testing.assert_equal(detector.directions, other.directions)
        different = ProjectionHistogramDrift(window=2, seed=18)
        different.score([], [1, 0])
        self.assertFalse(np.array_equal(detector.directions, different.directions))
        with self.assertRaises(ValueError): detector.score([], [1, 0, 0])
        for kwargs in [dict(window=0), dict(projections=0), dict(bins=1), dict(seed=-1), dict(window=True)]:
            with self.assertRaises(ValueError): ProjectionHistogramDrift(**kwargs)

    def test_causal_observations_reset_and_labels(self):
        trajectories = []
        for labels, last in [(('A',) * 5, 'a'), (('B',) * 5, 'c')]:
            env = SemanticCacheEnv(workload(('a', 'a', 'c', 'c', last), labels), Encoder(), drift_window=2)
            obs, _ = env.reset(seed=4)
            observations = [obs.copy()]
            for _ in range(3):
                obs, _, _, _, _ = env.step(1)
                observations.append(obs.copy())
            self.assertEqual(obs[0], 1)  # Current 'c' completes the changed second window.
            np.testing.assert_equal([v[0] for v in observations[:3]], [0, 0, 0])
            self.assertTrue(env.observation_space.contains(obs))
            _, _, _, _, info = env.step(1)
            self.assertEqual(info['query_drift_features']['query_embedding_shift'], 1)
            trajectories.append(observations)
            reset_obs, _ = env.reset(seed=4)
            self.assertEqual(reset_obs[0], 0)
            self.assertEqual(len(env.history), 0)
        np.testing.assert_equal(trajectories[0], trajectories[1])  # Labels and future query cannot change prefix.

    def test_window_one_gym_policy_and_reports(self):
        rows = workload(('a', 'a', 'c', 'c'), ('A', 'A', 'C', 'C'))
        check_env(SemanticCacheEnv(rows, Encoder(), drift_window=1), skip_render_check=True)
        report = compare(rows, Encoder(), drift_window=1)
        self.assertEqual(report, compare(rows, Encoder(), drift_window=1))
        self.assertEqual(report['always_reuse']['phase_transitions'][0]['mean_shift_after'], 0.5)
        self.assertEqual(drift_rule(np.array([0.05, 0.03, 0, 0, 1])), 0)
        self.assertEqual(drift_rule(np.array([0.2, 0.03, 0, 0, 1])), 1)
        self.assertEqual(drift_rule(np.array([0.2, 0.03, 0, 0, 1]), shift_threshold=0.3), 0)


if __name__ == '__main__':
    unittest.main()
