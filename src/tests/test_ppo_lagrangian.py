"""Offline public-contract fixtures; no repository environment/model downloads."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from src.rl.config import Config, seed_all, observation
from src.rl.policy import ActorCritics, PPO, Multipliers, combined_advantage, clipped_objective
from src.rl.rollout import gae, signals, Collector
from src.rl.train import train, feasible
from src.rl.evaluate import evaluate_episode, compare


class TinyCacheEnv(gym.Env):
    def __init__(self, split='train', labels=True, emit_flag=True, length=8):
        self.split, self.labels, self.emit_flag, self.length = split, labels, emit_flag, length
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, (5,), np.float32)
        self.index = 0

    def obs(self):
        shifted = self.index >= self.length//2
        return np.array([float(shifted), float(self.index % 4 == 0), 0.25, 0.5, 0.5], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self.obs(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        hit = self.index % 4 != 0
        effective = int(action == 1 or not hit)
        status = 'not_reused' if effective else 'unevaluable' if not self.labels else 'incorrect' if self.index >= self.length//2 else 'correct'
        utility = 1.0 if status == 'correct' else -2.0 if status == 'incorrect' else 0.0
        components = dict(reuse_utility=utility, recompute_cost=-0.2 if effective else 0.0)
        info = dict(effective_action=effective, requested_action=int(action), reuse_status=status,
                    reward_components=components, cache_hit=hit, cache_occupancy=0.5,
                    forced_recomputation=action == 0 and not hit,
                    request_id=f'{self.split}-{self.index}', phase_id=self.index//4,
                    intent_id='evaluation-only-secret')
        if self.emit_flag:
            info['labeled_request'] = self.labels
        self.index += 1
        return self.obs(), sum(components.values()), self.index == self.length, False, info


def factory(**kwargs):
    return TinyCacheEnv(**kwargs)


def reuse(obs):
    return 0 if obs[1] < 1 else 1


def recompute(obs):
    return 1


def drift(obs):
    return 0 if obs[1] < 1 and obs[0] <= 0.1 else 1


def short_config(**kwargs):
    return Config(total_steps=64, rollout_length=16, minibatch_size=8, epochs=2, hidden_size=16,
                  eval_frequency=32, checkpoint_frequency=32, log_frequency=16,
                  min_labeled_steps=1, false_limit=0.5, recompute_limit=1, **kwargs)


class PPOTests(unittest.TestCase):
    def test_actor_shape_distribution_actions_and_labels(self):
        seed_all(9)
        model = ActorCritics(8)
        dist = model.distribution(torch.zeros(5, 5))
        self.assertEqual(dist.probs.shape, (5, 2))
        torch.testing.assert_close(dist.probs.sum(-1), torch.ones(5))
        self.assertEqual(model.values(torch.zeros(5, 5)).shape, (5, 3))
        obs = np.zeros(5, np.float32)
        for _ in range(20): self.assertIn(model.action(obs)[0], (0, 1))
        with self.assertRaises(ValueError): observation(np.zeros(7, np.float32))
        with self.assertRaises(ValueError): observation(np.zeros(5, np.float64))
        with self.assertRaises(ValueError): model.distribution(torch.zeros(2, 6))
        env = TinyCacheEnv()
        batch, _, _ = Collector(env, PPO(short_config()), short_config()).collect(4)
        self.assertEqual(batch['observations'].shape, (4, 5))
        self.assertNotIn('intent_id', batch)

    def test_separate_returns_and_boundaries(self):
        a, returns = gae([1, 2, 10], [0, 0, 0], [0, 0, 0], [False, True, True], [False, True, True], 1, 1)
        np.testing.assert_equal(a, [3, 2, 10])
        np.testing.assert_equal(returns, a)
        _, false_returns = gae([0, 1], [0, 0], [0, 0], [False, True], [False, True], 1, 1)
        _, cost_returns = gae([1, 0], [0, 0], [0, 0], [False, True], [False, True], 1, 1)
        np.testing.assert_equal(false_returns, [1, 1])
        np.testing.assert_equal(cost_returns, [1, 0])
        # A truncation bootstraps its own final observation, not the next episode.
        a, _ = gae([1, 99], [0, 0], [5, 0], [False, True], [True, True], 0.9, 1)
        self.assertAlmostEqual(a[0], 5.5)
        a, _ = gae([0, 0, 1], [0, 0, 0], [0, 0, 0], [False]*3, [False]*3, 1, 1, [True, False, True])
        np.testing.assert_equal(a, [0, 0, 1])

    def test_lagrangian_and_clipping(self):
        advantages = torch.tensor([[4., 1., 2.], [1., 3., 0.]])
        torch.testing.assert_close(combined_advantage(advantages, [2, 0.5]), torch.tensor([1., -5.]))
        result = clipped_objective(torch.tensor([2., 0.5]), torch.tensor([1., -1.]), 0.2)
        torch.testing.assert_close(result, torch.tensor([1.2, -0.8]))

    def test_dual_rates_projection_and_disabled_constraints(self):
        cfg = short_config(false_dual_lr=1, recompute_dual_lr=1, max_false_multiplier=0.1)
        dual = Multipliers(cfg)
        dual.update(8, 10, 10, 10)
        self.assertAlmostEqual(dual.values[0], 0.1)
        dual.update(0, 10, 0, 10)
        np.testing.assert_equal(dual.values, [0, 0])
        cfg = replace(cfg, false_limit=0.1, recompute_limit=0.2, max_false_multiplier=100)
        a, b = Multipliers(cfg), Multipliers(cfg)
        a.update(2, 10, 5, 10)
        b.update(20, 100, 50, 100)
        np.testing.assert_equal(a.values, b.values)
        self.assertGreater(a.values[0], 0)
        self.assertGreater(a.values[1], 0)
        disabled = Multipliers(replace(cfg, false_constraint=False, recompute_constraint=False))
        disabled.update(10, 10, 10, 10)
        np.testing.assert_equal(disabled.values, [0, 0])

    def test_effective_cost_and_reward_no_double_penalty(self):
        env = TinyCacheEnv()
        env.reset()
        _, reward, _, _, info = env.step(0)
        signal = signals(reward, info, short_config())
        self.assertEqual(signal['recompute_cost'], 1)
        self.assertTrue(signal['forced'])
        self.assertEqual(signal['utility'], 0)
        self.assertEqual(signals(reward, info, replace(short_config(), recompute_constraint=False))['utility'], -0.2)
        env.index = 1
        _, reward, _, _, info = env.step(0)
        self.assertEqual(signals(reward, info, short_config())['utility'], 1)
        env.index = 5
        _, reward, _, _, info = env.step(0)
        self.assertEqual(signals(reward, info, short_config())['false_cost'], 1)
        self.assertEqual(signals(reward, info, short_config())['utility'], -2)
        info['reward_components'] = {'unknown': 1}
        with self.assertRaisesRegex(ValueError, 'recompute_cost'): signals(reward, info, short_config())

    def test_unknown_costs_and_missing_metadata_failures(self):
        env = TinyCacheEnv(labels=False)
        env.reset()
        env.index = 1
        _, reward, _, _, info = env.step(0)
        signal = signals(reward, info, short_config())
        self.assertIsNone(signal['false_cost'])
        self.assertEqual(signal['status'], 'unevaluable')
        self.assertFalse(signal['evaluable'])
        env = TinyCacheEnv(labels=False)
        with self.assertRaisesRegex(ValueError, 'no valid labeled'):
            Collector(env, PPO(short_config()), short_config()).collect(8)
        env = TinyCacheEnv(emit_flag=False)
        env.reset()
        _, reward, _, _, info = env.step(1)
        with self.assertRaisesRegex(ValueError, 'labeled_request'): signals(reward, info, short_config())
        self.assertTrue(signals(reward, info, short_config(), {'train-0': 'A'})['evaluable'])

    def test_training_checkpoint_reproducibility_and_reports(self):
        cfg = short_config()
        seed_all(cfg.seed)
        initial = [p.detach().clone() for p in PPO(cfg).model.parameters()]
        with tempfile.TemporaryDirectory() as tmp:
            one, meta = train(factory, {'split': 'train'}, {'split': 'validation'}, Path(tmp)/'one', cfg)
            two, _ = train(factory, {'split': 'train'}, {'split': 'validation'}, Path(tmp)/'two', cfg)
            for first, second in zip(one.model.parameters(), two.model.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            loaded, loaded_meta = PPO.load(Path(tmp)/'one/final.pt')
            self.assertEqual(loaded_meta['steps'], 64)
            np.testing.assert_equal(loaded.multipliers.values, one.multipliers.values)
            obs = np.zeros(5, np.float32)
            self.assertEqual(loaded.model.action(obs, True)[0], one.model.action(obs, True)[0])
            baseline = {'always_reuse': reuse, 'always_recompute': recompute, 'drift_rule': drift}
            reports = compare(factory, {'split': 'held_out'}, loaded, baseline, [0, 1])
            again = compare(factory, {'split': 'held_out'}, one, baseline, [0, 1])
            self.assertEqual(reports, again)
            self.assertEqual(reports['always_reuse']['runs'][0]['overall']['false_hit_rate'], 3/8)
            self.assertEqual(reports['always_recompute']['runs'][0]['overall']['recomputation_rate'], 1)
            self.assertEqual(len(reports['ppo_lagrangian']['runs'][0]['transitions']), 1)
            self.assertTrue((Path(tmp)/'one/step_32.pt').exists())
            self.assertTrue((Path(tmp)/'one/best.pt').exists())
            self.assertEqual(len(meta['multiplier_history']), 4)
            records = [json.loads(line) for line in (Path(tmp)/'one/metrics.jsonl').read_text().splitlines()]
            self.assertTrue(any('reward_critic_loss' in row for row in records))
            costs = [json.loads(line) for line in (Path(tmp)/'one/costs.jsonl').read_text().splitlines()]
            self.assertEqual(len(costs), cfg.total_steps)
            self.assertTrue(all('false_cost' in row and 'recompute_cost' in row for row in costs))
            self.assertTrue(all(np.isfinite(parameter.detach().numpy()).all() for parameter in loaded.model.parameters()))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial, loaded.model.parameters())))

    def test_insufficient_evidence_selection_and_config(self):
        report = evaluate_episode(factory, {'split': 'validation'}, reuse, replace(short_config(), min_labeled_steps=100))
        self.assertIsNone(report['overall']['false_constraint_satisfied'])
        self.assertFalse(feasible(report, short_config()))
        for cfg in [replace(short_config(), total_steps=0), replace(short_config(), minibatch_size=32),
                    replace(short_config(), gamma=1.1), replace(short_config(), actor_lr=float('nan'))]:
            with self.assertRaises(ValueError): cfg.validate()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'distinct'): train(factory, {'split': 'x'}, {'split': 'x'}, tmp, short_config())


if __name__ == '__main__':
    unittest.main()
