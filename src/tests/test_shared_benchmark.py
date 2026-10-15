import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from gymnasium.utils.env_checker import check_env

from src.benchmark import CORE_METRICS, benchmark, load_json_object
from src.datasets.workload import write
from src.env.semantic_cache_env import SemanticCacheEnv
from src.env.factory import make_env
from src.models.semantic_cache import FIFOCache
from src.rl.config import Config
from src.rl.train import train


class Encoder:
    def encode(self, query):
        return {'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]}[query]


def rows(queries=('a', 'b', 'a', 'c', 'a')):
    return [dict(request_id=str(i), query=query, intent_id=query,
                 topic_id=None, phase_id=i // 3, step=i)
            for i, query in enumerate(queries)]


def factory(**kwargs):
    return SemanticCacheEnv(rows(), Encoder(), capacity=2, **kwargs)


def split_factory(split='train'):
    data = rows()
    for row in data:
        row['request_id'] = f'{split}-{row["request_id"]}'
    return SemanticCacheEnv(data, Encoder(), capacity=2)


def reuse_method(seed=0):
    return lambda observation: 0


def recompute_method(seed=0):
    return lambda observation: 1


class SharedBenchmarkTests(unittest.TestCase):
    def test_environment_contract_and_fifo(self):
        check_env(factory(), skip_render_check=True)
        env = factory()
        env.reset()
        self.assertIsInstance(env.cache, FIFOCache)
        _, _, _, _, info = env.step(1)
        self.assertTrue(info['labeled_request'])
        self.assertEqual(set(env.cached_intents), set(env.cache.entries))
        self.assertEqual(set(vars(env.cache.entries[0])), {'query', 'embedding', 'response'})
        report = benchmark(factory, {}, {'reuse': reuse_method, 'recompute': recompute_method})
        self.assertEqual(set(report['results']), {'reuse', 'recompute'})
        reuse = report['results']['reuse']
        self.assertEqual(set(reuse), set(CORE_METRICS))
        self.assertEqual(reuse['requests'], 5)
        self.assertEqual(reuse['evaluable_labeled_requests'], 5)
        self.assertEqual(reuse['false_hit_rate'], 0)
        self.assertEqual(reuse['recomputation_rate'], 4 / 5)
        self.assertEqual(report['results']['recompute']['recomputation_rate'], 1)

    def test_trace_mismatch_is_rejected(self):
        count = 0

        def changing_factory(**kwargs):
            nonlocal count
            count += 1
            data = rows(('a', 'b'))
            data[0]['request_id'] = f'run{count}-0'
            return SemanticCacheEnv(data, Encoder(), **kwargs)

        with self.assertRaisesRegex(ValueError, 'same request trace'):
            benchmark(changing_factory, {}, {'a': recompute_method, 'b': recompute_method})

    def test_method_factories_are_loaded_from_json(self):
        methods = load_json_object('examples/methods.json')
        report = benchmark(factory, {}, methods)
        self.assertEqual(set(report['results']), set(methods))
        self.assertEqual(set(report['results']['always_reuse']), set(CORE_METRICS))
        with self.assertRaisesRegex(ValueError, 'factory and settings'):
            benchmark(factory, {}, {'invalid': {'factory': 'src.evaluate:always_reuse'}})

    def test_false_hit_denominator(self):
        from src.benchmark import summarize
        statuses = ('correct', 'incorrect', 'unevaluable', 'not_reused')
        records = [dict(status=status, evaluable=status != 'unevaluable',
                        reward=0, recomputed=status == 'not_reused')
                   for status in statuses]
        result = summarize(records)
        self.assertEqual(set(result), set(CORE_METRICS))
        self.assertEqual(result['false_hit_rate'], 1 / 3)
        self.assertEqual(result['recomputation_rate'], 1 / 4)

    def test_repository_factory_and_ppo_adapter(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'workload.jsonl'
            write(path, rows())
            with patch('src.env.factory.SentenceEncoder', return_value=Encoder()):
                env = make_env(str(path), capacity=2)
                env.reset()
                self.assertIsInstance(env.cache, FIFOCache)

            config = Config(total_steps=16, rollout_length=8, minibatch_size=4,
                            epochs=1, eval_frequency=8, checkpoint_frequency=8,
                            log_frequency=8, min_labeled_steps=1,
                            false_limit=1, recompute_limit=1)
            output = Path(directory) / 'training'
            train(split_factory, {'split': 'train'}, {'split': 'validation'}, output, config)
            rollout = json.loads((output / 'metrics.jsonl').read_text().splitlines()[0])
            self.assertIn('quality_adjusted_hit_rate', rollout)
            self.assertIn('reuse_precision', rollout)
            methods = {'ppo': {'factory': 'src.method_adapters:ppo_checkpoint',
                               'settings': {'checkpoint': str(output / 'best.pt')}}}
            report = benchmark(split_factory, {'split': 'heldout'}, methods)
            self.assertEqual(report['results']['ppo']['requests'], 5)


if __name__ == '__main__':
    unittest.main()
