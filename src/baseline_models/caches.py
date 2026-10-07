"""
caches.py

Semantic-cache baselines: fixed-threshold similarity lookup + LRU or LFU
eviction. These are the "zero learning, zero drift-awareness" floor that
any RL controller needs to beat.

SCOPE NOTE: this module owns only the caching *policy* logic. It makes
no assumption about how query records are generated -- that is the
gym's responsibility (built separately). See decision_log.md, entry D1,
for the exact interface contract this module expects from whatever
produces query records.

Design notes (see decision_log.md for the reasoning/citations behind
each choice, referenced by ID below):
- Cosine similarity is used as the match metric (D2), against a fixed
  threshold (D3, default 0.85).
- The similarity-matching step is IDENTICAL across both baselines and
  will be reused, unchanged, for the RL controller's fixed-threshold
  ablation -- only the eviction policy differs. This isolates "does
  learning help?" from "did we just change the matching machinery?"
- On a hit, we record whether the served entry was STALE at serve time
  (its cached content_version != the query's current ground-truth
  content_version, supplied by the caller). This is only computable
  because ground truth is controllable in a simulation -- the same
  caveat flagged for the CMAB / CacheSense / FreshCache papers.
- Eviction tie-breaking is deterministic (D5).
- Metric definitions mirror CacheSense's Table 1 exactly (D6).
- NoCache is included as the trivial upper bound on cost / lower bound
  on staleness, matching the convention used in all four reviewed papers.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np


@dataclass
class CacheEntry:
    embedding: np.ndarray
    cached_content_version: object   # opaque id; only compared for equality, D1
    cost_at_cache_time: float
    last_used_step: int = 0     # for LRU
    hit_count: int = 0          # for LFU


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class SemanticCacheBaseline:
    """
    Shared logic for fixed-threshold semantic caching. Subclasses only
    need to implement `_choose_eviction_victim`.
    """

    def __init__(self, capacity: int = 50, similarity_threshold: float = 0.85):
        self.capacity = capacity
        self.threshold = similarity_threshold
        self.entries: Dict[int, CacheEntry] = {}   # key: arbitrary slot id
        self._next_key = 0

        # running stats
        self.n_queries = 0
        self.n_hits = 0
        self.n_stale_hits = 0
        self.total_cost = 0.0        # cost actually incurred (LLM calls)
        self.total_cost_nocache = 0.0  # cost if every query hit the LLM (normalization baseline)

    # ---- to be overridden ----------------------------------------------

    def _choose_eviction_victim(self) -> int:
        raise NotImplementedError

    def _on_hit_update(self, key: int, step: int):
        raise NotImplementedError

    # ---- shared lookup / insert logic -----------------------------------

    def _find_best_match(self, query_emb: np.ndarray):
        best_key, best_sim = None, -1.0
        for key, entry in self.entries.items():
            sim = cosine_sim(query_emb, entry.embedding)
            if sim > best_sim:
                best_key, best_sim = key, sim
        return best_key, best_sim

    def _insert(self, query_emb, content_version, cost, step):
        if len(self.entries) >= self.capacity:
            victim_key = self._choose_eviction_victim()
            del self.entries[victim_key]
        key = self._next_key
        self._next_key += 1
        self.entries[key] = CacheEntry(
            embedding=query_emb,
            cached_content_version=content_version,
            cost_at_cache_time=cost,
            last_used_step=step,
            hit_count=1,
        )

    # ---- main step --------------------------------------------------------

    def process(self, record: dict):
        """
        record: a dict describing one incoming query, with REQUIRED keys:
            - "embedding": np.ndarray, the query's embedding vector
            - "true_cost": float, cost of serving this query fresh (a miss)
            - "content_version": opaque id, the query's CURRENT ground-truth
                  content version (used only to check staleness of a hit)
            - "step": int, a monotonically increasing counter (used for LRU)

        This is the full interface contract this module expects from
        whatever generates the query stream (the gym, built separately).
        See decision_log.md, entry D1.
        """
        self.n_queries += 1
        self.total_cost_nocache += record["true_cost"]

        best_key, best_sim = self._find_best_match(record["embedding"])

        if best_key is not None and best_sim >= self.threshold:
            # HIT -- served from cache, no LLM cost incurred
            self.n_hits += 1
            entry = self.entries[best_key]
            if entry.cached_content_version != record["content_version"]:
                self.n_stale_hits += 1
            self._on_hit_update(best_key, record["step"])
            return "hit", best_sim

        # MISS -- simulate an LLM call, pay the cost, cache the fresh result
        self.total_cost += record["true_cost"]
        self._insert(
            query_emb=record["embedding"],
            content_version=record["content_version"],
            cost=record["true_cost"],
            step=record["step"],
        )
        return "miss", best_sim

    # ---- reporting ----------------------------------------------------------

    def metrics(self) -> dict:
        hit_rate = self.n_hits / self.n_queries if self.n_queries else 0.0
        stale_hit_rate = self.n_stale_hits / self.n_hits if self.n_hits else 0.0
        normalized_cost = self.total_cost / self.total_cost_nocache if self.total_cost_nocache else 0.0
        return {
            "n_queries": self.n_queries,
            "hit_rate": round(hit_rate, 4),
            "stale_hit_rate": round(stale_hit_rate, 4),
            "normalized_api_cost": round(normalized_cost, 4),
        }


class LRUSemanticCache(SemanticCacheBaseline):
    """Fixed-threshold semantic lookup + Least-Recently-Used eviction."""

    def _choose_eviction_victim(self) -> int:
        # smallest last_used_step = least recently used.
        # Ties broken by dict insertion order (D5): min() over a dict
        # returns the first minimal element encountered during iteration,
        # and dicts preserve insertion order, so the earliest-inserted
        # tied entry is evicted -- deterministic, no extra randomness.
        return min(self.entries, key=lambda k: self.entries[k].last_used_step)

    def _on_hit_update(self, key: int, step: int):
        self.entries[key].last_used_step = step


class LFUSemanticCache(SemanticCacheBaseline):
    """Fixed-threshold semantic lookup + Least-Frequently-Used eviction."""

    def _choose_eviction_victim(self) -> int:
        # smallest hit_count = least frequently used. Same deterministic
        # tie-break rule as LRU (D5).
        return min(self.entries, key=lambda k: self.entries[k].hit_count)

    def _on_hit_update(self, key: int, step: int):
        self.entries[key].hit_count += 1
        self.entries[key].last_used_step = step  # kept only in case a future
        # LFU+recency hybrid baseline is added; unused by plain LFU logic.


class NoCacheBaseline:
    """Trivial baseline: every query hits the LLM. Upper bound on cost, floor on staleness."""

    def __init__(self):
        self.n_queries = 0
        self.total_cost = 0.0
        self.total_cost_nocache = 0.0

    def process(self, record: dict):
        self.n_queries += 1
        self.total_cost += record["true_cost"]
        self.total_cost_nocache += record["true_cost"]
        return "miss", None

    def metrics(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "hit_rate": 0.0,
            "stale_hit_rate": 0.0,
            "normalized_api_cost": 1.0,
        }
