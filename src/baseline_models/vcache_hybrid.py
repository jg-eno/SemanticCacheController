"""vCache hybrid: the authors' decision logic driven by this environment's cache.

This is a HYBRID, not the official system end to end. It imports the official, unmodified
`_Algorithm` from `vcache.vcache_policy.strategies.verified` (the code that fits the
per-entry logistic model, computes the exploration probability tau and draws the
explore/exploit coin) and uses it to choose REUSE or RECOMPUTE inside `SemanticCacheEnv`.
No official code is copied here (its licence is CC BY-NC-ND 3.0).

What is official
- observation storage : one official `EmbeddingMetadataObj` per cached entry
                        (starts with the official seed points (0, 0) and (1, 1))
- decision            : `_Algorithm.select_action(similarity, metadata)`
                        (explores until an entry has 6 observations, then uses tau)
- learning            : `_Algorithm.add_observation_to_metadata(similarity, correct, metadata)`
- the explore/exploit coin is the `random` module's `uniform`, seeded here

What is this repository's (the differences from the official system)
- cache               : SemanticCacheEnv's cache (capacity, eviction policy, and the
                        environment's similarity threshold gate the nearest neighbour).
                        The official system has none of these: it searches all entries,
                        has no global threshold, and never evicts.
- similarity          : the environment's cosine similarity, not the HNSW score
- "LLM call"          : simulated from intent labels (the cached answer is correct when
                        incoming and cached intent labels match); labels are read only
                        on the explore path
- updates             : applied immediately, where the official system queues them in a
                        background thread. The explore decision for the next request sees
                        the same state as the adapter in `vcache_official.py` after it
                        waits for the queue.
- insertion           : like the official policy, no entry is added after an explore that
                        found the cached answer correct (the method sets
                        `env.skip_admission` for that step; the environment honours it).
                        Without this, every exploration inserted a near-duplicate, the
                        small cache churned, entries were evicted before they collected 6
                        observations, and the method stayed in exploration. For environments
                        without `skip_admission`, the inserted duplicate shares its parent's
                        metadata instead.

Requirements: the official package installed (Linux, Python 3.10-3.12; see BASELINE.md,
"Running the official vCache policy"). Action convention: REUSE = 0, RECOMPUTE = 1.
"""

import random

import numpy as np

REUSE, RECOMPUTE = 0, 1

INSTALL_HINT = (
    "The official vCache package is not importable. Install it from "
    "https://github.com/vcache-project/vCache on Linux with Python 3.10-3.12 "
    "(see BASELINE.md, 'Running the official vCache policy')."
)


def load_official():
    """Return (algorithm_class, metadata_class) from the installed official package."""
    try:
        from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
            EmbeddingMetadataObj,
        )
        from vcache.vcache_policy.strategies.verified import _Algorithm
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    return _Algorithm, EmbeddingMetadataObj


def _is_exploit(action):
    return getattr(action, "value", action) == "exploit"


class OfficialDecisionMethod:
    """Benchmark method: official vCache decision logic over the environment's cache."""

    def __init__(self, seed=0, delta=0.02, algorithm=None, metadata_class=None):
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if algorithm is None or metadata_class is None:
            algorithm_class, metadata_class = load_official()
            algorithm = algorithm_class(delta=delta)
        self.delta = delta
        self.algorithm = algorithm
        self.metadata_class = metadata_class
        random.seed(seed)                # the official coin is random.uniform (global state)
        self.metadata = {}               # cached entry object -> official metadata object
        self._alias_of = None            # metadata the next inserted entry should share
        self.explored = 0                # diagnostics
        self.exploited = 0

    @staticmethod
    def _newest_entry(cache):
        entries = cache.entries
        if isinstance(entries, dict):    # baseline_models caches
            return next(reversed(entries.values()), None)
        return entries[-1] if entries else None   # models.semantic_cache.FIFOCache (list)

    def _metadata_for(self, entry):
        if entry not in self.metadata:
            self.metadata[entry] = self.metadata_class(embedding_id=len(self.metadata), response="")
        return self.metadata[entry]

    def select_action(self, observation, env):
        # After a correct explore the environment inserts a duplicate of its parent; it
        # shares the parent's observations, as the official policy adds no entry there.
        if self._alias_of is not None:
            newest = self._newest_entry(env.cache)
            if newest is not None and newest not in self.metadata:
                self.metadata[newest] = self._alias_of
            self._alias_of = None
        entry = env.candidate
        if entry is None:
            return RECOMPUTE             # no nearest neighbour above the cache threshold
        similarity = float(np.clip(env.similarity, 0.0, 1.0))
        metadata = self._metadata_for(entry)
        action = self.algorithm.select_action(similarity_score=similarity, metadata=metadata)
        if _is_exploit(action):
            self.exploited += 1
            return REUSE
        # Explore: the LLM is called and its answer compared with the cached one.
        self.explored += 1
        incoming = env.workload[env.index]["intent_id"]
        cached = env.cached_intents.get(entry)
        if incoming is not None and cached is not None:
            correct = incoming == cached
            self.algorithm.add_observation_to_metadata(
                similarity_score=similarity, is_correct=correct, metadata=metadata)
            if correct:
                if hasattr(env, "skip_admission"):
                    # Like the official policy: no entry is added after a correct explore.
                    env.skip_admission = True
                else:
                    self._alias_of = metadata   # environments that always insert
        return RECOMPUTE


def make_official_decision_method(seed=0, delta=0.02):
    return OfficialDecisionMethod(seed=seed, delta=delta)