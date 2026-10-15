from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ..models.semantic_cache import FIFOCache, SentenceEncoder, unit
from ..baseline_models.caches import (
    FIFOSemanticCache,
    LIFOSemanticCache,
    LRUSemanticCache,
    LFUSemanticCache,
)
from ..datasets.workload import validate
from ..models.query_drift import ProjectionHistogramDrift

FEATURE_NAMES = ('query_embedding_shift', 'query_novelty', 'recent_cache_miss_rate',
                 'recent_reuse_rate', 'cache_occupancy', 'budget_remaining')
REUSE, RECOMPUTE = 0, 1


@dataclass(frozen=True)
class Rewards:
    correct_reuse: float = 1.0
    incorrect_reuse_penalty: float = 2.0
    recompute_cost: float = 0.2
    unevaluable_reuse: float = 0.0

    def __post_init__(self):
        values = (self.correct_reuse, self.incorrect_reuse_penalty, self.recompute_cost, self.unevaluable_reuse)
        if not all(np.isfinite(v) for v in values) or any(v < 0 for v in values[:3]):
            raise ValueError('Reward parameters must be finite; rewards, penalties and cost nonnegative')

    def components(self, status, recomputed):
        return dict(reuse_utility=self.correct_reuse if status == 'correct' else
                    self.unevaluable_reuse if status == 'unevaluable' else 0.0,
                    false_hit_penalty=-self.incorrect_reuse_penalty if status == 'incorrect' else 0.0,
                    recompute_cost=-self.recompute_cost if recomputed else 0.0)


class SemanticCacheEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, workload, encoder=None, capacity=32, threshold=0.85, window=16,
                 rewards=None, response_provider=None, drift_window=16,
                 projection_count=8, histogram_bins=8, projection_seed=0,
                 cache_policy='fifo'):

        validate(workload)
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError('window must be a positive integer')
        cache_classes = {
            'fifo': FIFOCache,
            'lifo': LIFOSemanticCache,
            'lru': LRUSemanticCache,
            'lfu': LFUSemanticCache,
        }

        if cache_policy not in cache_classes:
            raise ValueError(
                f"Unknown cache_policy {cache_policy!r}. "
                f"Choose from {list(cache_classes)}."
            )

        # Retain the original cache's capacity and threshold validation.
        FIFOCache(capacity, threshold)

        self.cache_class = cache_classes[cache_policy]
        self.cache_policy = cache_policy
        self.drift_config = dict(window=drift_window, projections=projection_count,
                                 bins=histogram_bins, seed=projection_seed)
        ProjectionHistogramDrift(**self.drift_config)
        self.workload = deepcopy(workload)
        self.encoder = encoder if encoder is not None else SentenceEncoder()
        self.capacity, self.threshold, self.window = capacity, threshold, window
        self.rewards = rewards if rewards is not None else Rewards()
        # Request-specific placeholders do not consult evaluation labels.
        self.response_provider = response_provider or (lambda row: 'reference:' + row['request_id'])
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, shape=(6,), dtype=np.float32)
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('No reset options are supported')
        if self.cache_policy == 'fifo':
            self.cache = FIFOCache(self.capacity, self.threshold)
        else:
            self.cache = self.cache_class(
                capacity=self.capacity,
                similarity_threshold=self.threshold,
            )

        self.cached_intents = {}
        self.drift_detector = ProjectionHistogramDrift(**self.drift_config)
        self.history = deque(maxlen=2 * self.drift_detector.window - 1)
        self.misses, self.reuses = deque(maxlen=self.window), deque(maxlen=self.window)
        self.index, self.reuse_count, self.recompute_count = 0, 0, 0
        self.done = False
        self._prepare()
        return self.observation.copy(), {}

    def _prepare(self):
        self.embedding = unit(self.encoder.encode(self.workload[self.index]['query']))
        self.candidate, self.similarity = self.cache.lookup(self.embedding)
        shift = self.drift_detector.score(self.history, self.embedding)
        novelty = (1 - self.similarity) / 2 if self.candidate is not None else 1
        self.observation = np.asarray([shift, novelty, np.mean(self.misses) if self.misses else 0,
                                     np.mean(self.reuses) if self.reuses else 0,
                                     len(self.cache.entries) / self.capacity, 1], dtype=np.float32)
        self.observation = np.clip(self.observation, 0, 1)

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode ended or reset has not been called')
        if not self.action_space.contains(action):
            raise ValueError('action must be 0 or 1')
        row = self.workload[self.index]
        hit = self.candidate is not None
        reuse = action == REUSE and hit
        forced = action == REUSE and not hit
        status = 'not_reused'
        drift = dict(zip(FEATURE_NAMES, map(float, self.observation)))
        if reuse:
            response = self.candidate.response
            if self.cache_policy != 'fifo':
                self.cache.record_reuse(self.candidate, row['step'])
            incoming = row['intent_id']
            cached = self.cached_intents[self.candidate]
            status = 'unevaluable' if incoming is None or cached is None else 'correct' if incoming == cached else 'incorrect'
            self.reuse_count += 1
        else:
            response = self.response_provider({k: v for k, v in row.items() if k != 'intent_id'})
            
            if self.cache_policy == 'fifo':
                entry, evicted = self.cache.admit(
                    row['query'], self.embedding, response
                )
            else:
                entry, evicted = self.cache.admit(
                    row['query'],
                    self.embedding,
                    response,
                    step=row['step'],
                    intent_id=row['intent_id'],
                )

            if evicted is not None:
                del self.cached_intents[evicted]
            self.cached_intents[entry] = row['intent_id']
            self.recompute_count += 1
        components = self.rewards.components(status, not reuse)
        self.history.append(self.embedding.copy())
        self.misses.append(not hit)
        self.reuses.append(reuse)
        info = dict(requested_action=int(action), effective_action=REUSE if reuse else RECOMPUTE,
                    cache_hit=hit, cache_miss=not hit, forced_recomputation=forced, reuse_status=status,
                    reuse_count=self.reuse_count, recompute_count=self.recompute_count,
                    reward_components=components, cache_occupancy=len(self.cache.entries) / self.capacity,
                    query_drift_features=drift, response=response, request_id=row['request_id'],
                    phase_id=row['phase_id'], step=row['step'],
                    labeled_request=row['intent_id'] is not None)
        self.index += 1
        self.done = self.index == len(self.workload)
        if self.done:
            self.observation = np.zeros(6, dtype=np.float32)
        else:
            self._prepare()
        return self.observation.copy(), float(sum(components.values())), self.done, False, info
