import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from src.semantic_cache import SemanticCacheEnv, FIFOCache, Rewards, FEATURE_NAMES
from src.models.semantic_cache import cosine
from src.datasets.workload import generate, primary_rows, validate, wildchat_rows
from src.evaluate import compare, cacheeval, drift_rule


class Encoder:
    def encode(self, query):
        return {'a': [1, 0], 'paraphrase': [0.99, 0.1], 'b': [0, 1], 'c': [-1, 0],
                'collision': [1, 0]}[query]


def workload(queries=('a', 'paraphrase', 'b'), labels=('A', 'A', 'B')):
    return [dict(request_id=str(i), intent_id=label, query=q, topic_id=None,
                 phase_id=i // 2, step=i) for i, (q, label) in enumerate(zip(queries, labels))]


class Tests(unittest.TestCase):
    def test_workload(self):
        source = primary_rows([{'id': i, 'prompt': q, 'ID_Set': label} for i, (q, label) in
                              enumerate([('a', 'A'), ('paraphrase', 'A'), ('b', 'B'), ('c', 'C')])])
        a, b = generate(source, seed=4), generate(source, seed=4)
        self.assertEqual(a, b)
        self.assertNotEqual(a, generate(source, seed=5))
        distributions = validate(a, True, True)
        self.assertNotEqual(distributions[0], distributions[1])
        self.assertEqual([r['phase_id'] for r in a], sorted(r['phase_id'] for r in a))
        self.assertEqual({r['intent_id'] for r in a}, {'A', 'B', 'C'})
        self.assertTrue(any(r['query'] == 'paraphrase' for r in a))
        self.assertEqual(len({r['request_id'] for r in a}), len(a))
        self.assertNotIn('C', {r['intent_id'] for r in a if r['phase_id'] < 2})
        with self.assertRaises(ValueError): primary_rows([{'prompt': 'a'}])
        with self.assertRaises(ValueError): primary_rows([{'id': 0, 'prompt': 'a', 'ID_Set': None}])
        with self.assertRaises(ValueError): generate(source, [{'A': 2}, {'A': 2}])

    def test_retrieval_and_fifo(self):
        cache = FIFOCache(2)
        cache.admit('a', [1, 0], 'response')
        self.assertEqual(cache.lookup([0.99, 0.1])[0].response, 'response')
        self.assertIsNone(cache.lookup([0, 1])[0])
        self.assertAlmostEqual(cosine([2, 0], [1, 0]), 1)
        boundary = FIFOCache(1, 1)
        boundary.admit('a', [1, 0], 'r')
        self.assertIsNotNone(boundary.lookup([1, 0])[0])
        self.assertIsNone(boundary.lookup([0.99, 0.1])[0])
        cache.admit('b', [0, 1], 'b')
        cache.lookup([1, 0])
        cache.admit('c', [-1, 0], 'c')
        self.assertEqual([e.query for e in cache.entries], ['b', 'c'])
        self.assertEqual([e.response for e in cache.entries], ['b', 'c'])

    def test_observations_actions_rewards(self):
        env = SemanticCacheEnv(workload(), Encoder(), capacity=2)
        obs, _ = env.reset(seed=3)
        self.assertEqual(FEATURE_NAMES, ('query_embedding_shift', 'query_novelty', 'recent_cache_miss_rate',
                                      'recent_reuse_rate', 'cache_occupancy', 'budget_remaining'))
        np.testing.assert_equal(obs, [0, 1, 0, 0, 0, 1])
        obs, reward, end, trunc, info = env.step(0)
        self.assertTrue(info['forced_recomputation'])
        self.assertEqual(info['effective_action'], 1)
        self.assertEqual(reward, -0.2)
        self.assertEqual(info['reward_components']['recompute_cost'], -0.2)
        self.assertEqual(obs.shape, (6,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(float(obs[0]), 0)  # Full drift windows are not yet available.
        self.assertEqual(obs[2], 1)
        self.assertEqual(obs[4], 0.5)
        _, reward, _, _, info = env.step(0)
        self.assertEqual(info['reuse_status'], 'correct')
        self.assertEqual(reward, 1)
        self.assertEqual(len(env.cache.entries), 1)
        obs, _, end, trunc, _ = env.step(1)
        self.assertTrue(end)
        self.assertFalse(trunc)
        np.testing.assert_equal(obs, np.zeros(6))
        with self.assertRaises(RuntimeError): env.step(0)
        env.reset(seed=3)
        self.assertEqual(env.recompute_count, 0)

    def test_labels_never_change_decisions(self):
        observations, infos = [], []
        for labels in [('A', 'A'), ('A', 'B'), (None, None)]:
            env = SemanticCacheEnv(workload(('a', 'collision'), labels), Encoder(), drift_window=1)
            trajectory = [env.reset()[0]]
            trajectory.append(env.step(1)[0])
            obs, _, _, _, info = env.step(0)
            trajectory.append(obs)
            observations.append(trajectory)
            infos.append(info)
        for obs in observations[1:]: np.testing.assert_equal(obs, observations[0])
        self.assertEqual([i['reuse_status'] for i in infos], ['correct', 'incorrect', 'unevaluable'])
        self.assertTrue(all(i['effective_action'] == 0 for i in infos))
        self.assertEqual(infos[1]['reward_components']['false_hit_penalty'], -2)

    def test_admission_and_provider(self):
        def provider(row):
            self.assertNotIn('intent_id', row)
            return row['query']
        env = SemanticCacheEnv(workload(('a', 'a'), ('A', 'A')), Encoder(), capacity=1, response_provider=provider)
        env.reset()
        env.step(1)
        _, _, _, _, info = env.step(1)
        self.assertTrue(info['cache_hit'])
        self.assertEqual(env.cached_intents[env.cache.entries[0]], 'A')
        self.assertEqual(set(env.cached_intents), set(env.cache.entries))
        self.assertFalse(hasattr(env.cache.entries[0], 'intent_id'))
        self.assertFalse(hasattr(env.cache.entries[0], 'source_request_id'))
        self.assertEqual(env.recompute_count, 2)

    def test_invalid_configuration(self):
        for capacity in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError): FIFOCache(capacity)
        with self.assertRaises(ValueError): FIFOCache(threshold=float('nan'))
        with self.assertRaises(ValueError): SemanticCacheEnv(workload(), Encoder(), window=0)
        with self.assertRaises(ValueError): Rewards(recompute_cost=-1)

    def test_gym_and_evaluation(self):
        check_env(SemanticCacheEnv(workload(), Encoder()), skip_render_check=True)
        self.assertEqual(compare(workload(), Encoder()), compare(workload(), Encoder()))
        report = compare(workload(), Encoder())
        self.assertEqual(report['always_recompute']['recompute_count'], 3)
        self.assertEqual(report['always_reuse']['correct_reuse_count'], 1)
        self.assertEqual(report['always_reuse']['forced_recomputation_count'], 0)
        self.assertEqual(report['always_reuse']['cache_hit_rate'], 1/3)
        self.assertEqual(len(report['always_reuse']['phase_transitions']), 1)
        self.assertEqual(drift_rule(np.array([0.5, 0, 0, 0, 1, 1])), 1)

    def test_additional_data(self):
        rows = wildchat_rows([dict(timestamp=1, conversation_id='x', conversation=[dict(role='user', content='a')])])
        self.assertIsNone(rows[0]['intent_id'])
        result = cacheeval([dict(query_a='a', query_b='collision', binary_label='MISS')], Encoder())
        self.assertEqual(result['false_hit_rate'], 1)


if __name__ == '__main__':
    unittest.main()
