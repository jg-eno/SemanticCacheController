"""Post-action signals and separate GAE; no evaluation metadata enters the actor."""
import numpy as np
import torch
from .config import observation


def signals(reward, info, config, labels=None):
    required = ('effective_action', 'reuse_status', 'reward_components')
    if any(key not in info for key in required):
        raise ValueError('Environment info must expose effective_action, reuse_status, reward_components')
    effective = info['effective_action']
    status = info['reuse_status']
    if effective not in (0, 1) or status not in ('correct', 'incorrect', 'unevaluable', 'not_reused'):
        raise ValueError('Invalid effective action or reuse status')
    if (effective == 1) != (status == 'not_reused'):
        raise ValueError('Reuse status must agree with effective action')
    components = info['reward_components']
    if not isinstance(components, dict) or 'recompute_cost' not in components:
        raise ValueError('Reward components must identify the signed recompute_cost')
    values = list(components.values())
    if not values or not all(np.isfinite(v) for v in values) or not np.isclose(sum(values), reward):
        raise ValueError('Finite reward components must sum to the environment reward')
    penalty = components['recompute_cost']
    if penalty > 0 or (effective == 0 and penalty != 0):
        raise ValueError('recompute_cost must be a signed penalty on actual recomputation only')
    utility = float(reward - penalty if config.recompute_constraint else reward)
    labeled = info.get('labeled_request')
    if labeled is None and labels is not None:
        request_id = str(info.get('request_id', ''))
        if request_id not in labels:
            raise ValueError('Request missing from offline label mapping')
        labeled = labels[request_id] is not None
    if labeled is None and status in ('correct', 'incorrect'):
        labeled = True
    if labeled is None and config.false_constraint:
        raise ValueError('False-hit constraint requires info[labeled_request] or an offline request-ID label mapping')
    if labeled is not None and not isinstance(labeled, (bool, np.bool_)):
        raise ValueError('labeled_request must be boolean')
    if status in ('correct', 'incorrect') and labeled is False:
        raise ValueError('Evaluable reuse requires a labeled incoming request')
    evaluable = bool(labeled) and status != 'unevaluable'
    false_cost = float(status == 'incorrect') if evaluable else None
    return dict(utility=utility, environment_reward=float(reward), false_cost=false_cost,
                recompute_cost=float(effective == 1), labeled_request=bool(labeled), evaluable=evaluable,
                status=status, forced=bool(info.get('forced_recomputation', False)),
                request_id=info.get('request_id'), requested_action=info.get('requested_action'),
                effective_action=int(effective), reward_components=dict(components))


def gae(signals, values, next_values, terminated, ended, gamma, lam, valid=None):
    """Bootstrap truncations, but never propagate GAE through reset boundaries."""
    signals, values, next_values = map(lambda x: np.asarray(x, dtype=np.float64), (signals, values, next_values))
    valid = np.ones(len(signals), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    advantages = np.zeros(len(signals), dtype=np.float64)
    running = 0.0
    for t in reversed(range(len(signals))):
        if not valid[t]:
            running = 0.0
            continue
        next_valid = t+1 == len(signals) or bool(ended[t]) or valid[t+1]
        bootstrap = not terminated[t] and next_valid
        delta = signals[t] + gamma * next_values[t] * bootstrap - values[t]
        continuation = not ended[t] and next_valid
        running = delta + gamma * lam * continuation * running
        advantages[t] = running
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


class Collector:
    def __init__(self, env, agent, config, labels=None):
        self.env, self.agent, self.config, self.labels = env, agent, config, labels
        self.obs, _ = env.reset(seed=config.seed)
        self.obs = observation(self.obs)
        self.episode_steps, self.episode_reward, self.episode_number = 0, 0.0, 0

    def collect(self, length):
        records, episodes = [], []
        for _ in range(length):
            action, logp, values = self.agent.model.action(self.obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs = observation(next_obs)
            signal = signals(reward, info, self.config, self.labels)
            ended = terminated or truncated
            with torch.no_grad():
                next_values = np.zeros(3) if terminated else self.agent.model.values(torch.from_numpy(next_obs)).numpy()
            records.append((self.obs, action, logp, values, next_values, terminated, ended, signal))
            self.episode_steps += 1
            self.episode_reward += signal['utility']
            if self.episode_steps >= self.config.max_episode_steps and not ended:
                raise ValueError('Environment episode exceeded max_episode_steps')
            if ended:
                episodes.append(dict(episode=self.episode_number, reward=self.episode_reward, steps=self.episode_steps))
                self.episode_number += 1
                self.episode_steps, self.episode_reward = 0, 0.0
                self.obs, _ = self.env.reset(seed=self.config.seed + self.episode_number)
                self.obs = observation(self.obs)
            else:
                self.obs = next_obs
        metadata = [r[7] for r in records]
        evaluable = np.array([s['evaluable'] for s in metadata])
        if self.config.false_constraint and not evaluable.any():
            raise ValueError('False-hit constraint enabled but rollout has no valid labeled cost signal')
        costs = np.array([[s['utility'], s['false_cost'] if s['false_cost'] is not None else 0,
                           s['recompute_cost']] for s in metadata], dtype=np.float32)
        # Unknown false costs are storage placeholders only: excluded from GAE/loss/rate denominators.
        values, next_values = np.stack([r[3] for r in records]), np.stack([r[4] for r in records])
        masks = np.ones_like(costs, dtype=bool)
        masks[:, 1] = evaluable
        advantages, returns = [], []
        for head in range(3):
            gamma = self.config.gamma if head == 0 else self.config.cost_gamma
            lam = self.config.gae_lambda if head == 0 else self.config.cost_gae_lambda
            a, ret = gae(costs[:, head], values[:, head], next_values[:, head],
                         [r[5] for r in records], [r[6] for r in records], gamma, lam, masks[:, head])
            advantages.append(a)
            returns.append(ret)
        batch = dict(observations=np.stack([r[0] for r in records]), actions=np.array([r[1] for r in records]),
                     logp=np.array([r[2] for r in records]), masks=masks,
                     advantages=np.stack(advantages, axis=1), returns=np.stack(returns, axis=1))
        return batch, metadata, episodes
