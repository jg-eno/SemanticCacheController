"""Run configuration and public environment-factory contract."""
from dataclasses import dataclass, asdict
import importlib
import json
import math
import random
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

FEATURE_NAMES = ('query_embedding_shift', 'query_novelty', 'recent_cache_miss_rate',
                 'recent_reuse_rate', 'cache_occupancy', 'budget_remaining')


@dataclass
class Config:
    seed: int = 0
    total_steps: int = 10000
    rollout_length: int = 256
    epochs: int = 4
    minibatch_size: int = 64
    hidden_size: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    cost_gamma: float = 0.99
    cost_gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    false_limit: float = 0.01
    recompute_limit: float = 0.5
    false_dual_lr: float = 0.05
    recompute_dual_lr: float = 0.05
    initial_false_multiplier: float = 0.0
    initial_recompute_multiplier: float = 0.0
    max_false_multiplier: float = 100.0
    max_recompute_multiplier: float = 100.0
    false_constraint: bool = True
    recompute_constraint: bool = True
    min_labeled_steps: int = 32
    min_labeled_fraction: float = 1.0
    eval_frequency: int = 1024
    checkpoint_frequency: int = 1024
    log_frequency: int = 256
    max_episode_steps: int = 100000

    def validate(self):
        for name in ('seed', 'total_steps', 'rollout_length', 'epochs', 'minibatch_size', 'hidden_size',
                     'min_labeled_steps', 'eval_frequency', 'checkpoint_frequency', 'log_frequency', 'max_episode_steps'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == 'seed' else 1):
                raise ValueError(f'{name} must be a valid integer')
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if self.minibatch_size > self.rollout_length:
            raise ValueError('minibatch_size must not exceed rollout_length')
        for name in ('gamma', 'gae_lambda', 'cost_gamma', 'cost_gae_lambda', 'false_limit',
                     'recompute_limit', 'min_labeled_fraction'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        for name in ('actor_lr', 'critic_lr', 'max_grad_norm', 'max_false_multiplier', 'max_recompute_multiplier'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('entropy_coef', 'value_coef', 'false_dual_lr', 'recompute_dual_lr'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        if not 0 < self.clip < 1:
            raise ValueError('clip must be in (0,1)')
        for kind in ('false', 'recompute'):
            if not 0 <= getattr(self, f'initial_{kind}_multiplier') <= getattr(self, f'max_{kind}_multiplier'):
                raise ValueError('Initial multiplier must be within its projection bounds')
        return self


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def observation(value):
    array = np.asarray(value)
    if array.shape != (6,) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('Actor input must be a finite float32 observation of shape (6,)')
    return array.copy()


def load_factory(spec):
    module, separator, name = spec.partition(':')
    if not separator:
        raise ValueError('Factory must have format module.path:callable')
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise ValueError('Factory must be callable')
    return factory


def make_env(factory, settings, seed):
    if isinstance(factory, str) and factory.startswith('gym:'):
        env = gym.make(factory[4:], **settings)
    else:
        env = (load_factory(factory) if isinstance(factory, str) else factory)(**settings)
    if not isinstance(env.action_space, gym.spaces.Discrete) or env.action_space.n != 2 or env.action_space.start != 0:
        raise ValueError('Environment must preserve Discrete(2) actions 0 and 1')
    if env.observation_space.shape != (6,) or env.observation_space.dtype != np.float32:
        raise ValueError('Environment must preserve the six-dimensional float32 observation')
    env.action_space.seed(seed)
    return env


def settings_file(path):
    settings = json.loads(Path(path).read_text())
    if not isinstance(settings, dict):
        raise ValueError('Environment settings must be a JSON object')
    return settings
