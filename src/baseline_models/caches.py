"""
caches.py

Deterministic semantic-cache baselines for a fixed-threshold lookup policy
with FIFO, LIFO, LRU or LFU eviction. These are intentionally simple, zero-learning
references used to compare against a learned controller.

This module owns only the cache-policy logic. It does not assume a single
input format. Instead, it accepts either:

1) a direct cache record with an embedding, a fresh-request cost,
   a version label, and a monotonic step counter; or
2) a project workload row such as {"query": ..., "step": ..., "intent_id": ...}
   which is converted into the direct form by an injected encoder.

The internal contract is deliberately small and stable:
    - embedding: the query vector used for cosine similarity
    - true_cost: the cost of serving the request fresh on a miss
    - content_version: a version/id used to detect stale hits
    - step: a monotonic counter used for recency/frequency bookkeeping

Design notes:
- Cosine similarity is used as the match metric against a fixed threshold
  (default 0.85).
- The lookup step is shared by both baselines; only the eviction policy changes.
- On a hit, the cache checks whether the cached content version differs from
  the incoming request's current version and increments stale-hit stats if so.
- Eviction tie-breaking is deterministic.
- Metrics report hit rate, stale-hit rate, and normalized API cost.
- NoCache is the trivial upper bound on cost and lower bound on staleness.

Integration note for later metric code:
- A future evaluation-metrics module can iterate through workload rows and call
  cache.process(row) for each request.
- The returned status ("hit" / "miss") and the cache.metrics() summary are the
  canonical outputs to aggregate into richer reports.
- For per-request evaluation, each row can be annotated with the cache result,
  the request's true cost, and whether it was stale; these become the inputs for
  report generation without changing the baseline policy logic.
- In other words, the baseline cache should remain the data-producing layer,
  while a separate metrics file should remain the presentation/aggregation layer.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np



@dataclass(eq=False)
class CacheEntry:
    embedding: np.ndarray
    cached_content_version: object
    cost_at_cache_time: float
    last_used_step: int = 0
    hit_count: int = 0
    intent_id: object = None
    query: str = ""
    response: object = None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class SemanticCacheBaseline:
    """
    Shared logic for fixed-threshold semantic caching. Subclasses only
    need to implement `_choose_eviction_victim`.
    """

    def __init__(self, capacity: int = 32, similarity_threshold: float = 0.85,
                 encoder=None, default_true_cost: float = 1.0):
        self.capacity = capacity
        self.threshold = similarity_threshold
        self.encoder = encoder
        self.default_true_cost = default_true_cost
        self.entries: Dict[int, CacheEntry] = {}   # key: arbitrary slot id
        self._next_key = 0

        # running stats
        self.n_queries = 0
        self.n_hits = 0
        self.n_stale_hits = 0
        # Reuse correctness, using the same vocabulary as src/evaluate.py and src/env:
        #   correct / incorrect / unevaluable (either side has no intent label).
        self.n_correct_hits = 0
        self.n_incorrect_hits = 0
        self.n_unevaluable_hits = 0
        self.n_evaluable = 0         # requests with a label whose outcome is judgeable (src/rl/rollout.py:41)
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

    def _insert(self, query_emb, content_version, cost, step, intent_id=None):
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
            intent_id=intent_id,
        )

    # ---- main step --------------------------------------------------------

    def process(self, record: dict):
        """
        Accepts either a direct cache record or a workload row.

        Direct record example:
            {
                "embedding": np.ndarray,
                "true_cost": float,
                "content_version": object,
                "step": int,
            }

        Workload-row example:
            {
                "query": "What is the capital of France?",
                "step": 12,
                "intent_id": "example-capital",
                "true_cost": 1.0,         # optional
                "content_version": "v2"   # optional
            }

        When the input lacks an "embedding", the cache converts the query string
        to an embedding using the injected encoder and fills in the required
        internal fields. If "true_cost" is missing it falls back to
        default_true_cost. "content_version" is NOT inferred from "intent_id":
        a missing version means staleness is unknown (never counted as stale),
        while "intent_id" is used only to judge whether a hit was correct.

        Future metrics integration:
            - call cache.process(row) for each request in a workload
            - record the returned status and the request metadata
            - aggregate per-step or per-phase outcomes using the cache.metrics()
              summary as the source of truth for hit-rate and normalized-cost
              reporting
        """
        if "embedding" not in record:
            if "query" not in record or "step" not in record:
                raise ValueError('Workload rows require query and step fields')
            if self.encoder is None:
                raise ValueError('An encoder is required to process workload rows')
            record = {
                "embedding": self.encoder.encode(record["query"]),
                "true_cost": record.get("true_cost", self.default_true_cost),
                # Deliberately NOT defaulted to intent_id. An intent label says WHICH
                # question this is; a content version says whether its ANSWER changed.
                # Aliasing them makes wrong-answer hits look like stale hits and makes
                # unlabeled (intent_id=None) rows look perfectly fresh.
                "content_version": record.get("content_version"),
                "intent_id": record.get("intent_id"),
                "step": record["step"],
            }

        intent = record.get("intent_id")          # optional on direct records
        version = record.get("content_version")   # optional; None = staleness unknown

        self.n_queries += 1
        self.total_cost_nocache += record["true_cost"]

        best_key, best_sim = self._find_best_match(record["embedding"])

        if best_key is not None and best_sim >= self.threshold:
            # HIT -- served from cache, no LLM cost incurred
            self.n_hits += 1
            entry = self.entries[best_key]
            # Stale = the underlying content changed. Only judgeable when both versions are known.
            if (entry.cached_content_version is not None and version is not None
                    and entry.cached_content_version != version):
                self.n_stale_hits += 1
            # Correct / incorrect / unevaluable: identical rule to src/env/semantic_cache_env.py.
            if intent is None or entry.intent_id is None:
                self.n_unevaluable_hits += 1
            else:
                self.n_evaluable += 1
                if intent == entry.intent_id:
                    self.n_correct_hits += 1
                else:
                    self.n_incorrect_hits += 1
            self._on_hit_update(best_key, record["step"])
            return "hit", best_sim

        # MISS -- simulate an LLM call, pay the cost, cache the fresh result
        self.total_cost += record["true_cost"]
        if intent is not None:
            self.n_evaluable += 1   # a labeled, recomputed request is judgeable (never a false hit)
        self._insert(
            query_emb=record["embedding"],
            content_version=version,
            cost=record["true_cost"],
            step=record["step"],
            intent_id=intent,
        )
        return "miss", best_sim

    
    def lookup(self, embedding):
        """Find the best matching entry without updating usage statistics."""
        best_key, best_sim = self._find_best_match(embedding)

        if best_key is None or best_sim < self.threshold:
            return None, None

        return self.entries[best_key], best_sim

    
    def admit(self, query, embedding, response, step=0, intent_id=None):
        """Insert a fresh response and return both the new and evicted entries."""
        embedding = np.asarray(embedding, dtype=np.float64)

        evicted = None
        if len(self.entries) >= self.capacity:
            victim_key = self._choose_eviction_victim()
            evicted = self.entries.pop(victim_key)

        key = self._next_key
        self._next_key += 1

        entry = CacheEntry(
            embedding=embedding,
            cached_content_version=None,
            cost_at_cache_time=self.default_true_cost,
            last_used_step=step,
            hit_count=1,  # same as _insert(): the insertion counts as the first use
            intent_id=intent_id,
            query=query,
            response=response,
        )
        self.entries[key] = entry

        return entry, evicted


    def record_reuse(self, entry, step):
        """Update recency/frequency after the environment actually reuses an entry."""
        for key, cached_entry in self.entries.items():
            if cached_entry is entry:
                self._on_hit_update(key, step)
                return

        raise ValueError("Cannot record reuse: entry is no longer in the cache")


    # ---- reporting ----------------------------------------------------------

    def metrics(self) -> dict:
        hit_rate = self.n_hits / self.n_queries if self.n_queries else 0.0
        stale_hit_rate = self.n_stale_hits / self.n_hits if self.n_hits else 0.0
        normalized_cost = self.total_cost / self.total_cost_nocache if self.total_cost_nocache else 0.0
        n = self.n_queries
        return {
            "n_queries": n,
            "hit_rate": round(hit_rate, 4),
            # Staleness (content changed). Per HIT. 0.0 when no content versions are supplied.
            "stale_hit_rate": round(stale_hit_rate, 4),
            "normalized_api_cost": round(normalized_cost, 4),
            # Same names and denominators as src/rl/evaluate.py, so baselines and the
            # learned policy can sit in one table. false_hit_rate is per EVALUABLE REQUEST
            # (not per hit) and is None when no labeled request could be judged.
            "false_hit_rate": (round(self.n_incorrect_hits / self.n_evaluable, 4)
                               if self.n_evaluable else None),
            "recomputation_rate": round((n - self.n_hits) / n, 4) if n else 0.0,
            "evaluable_requests": self.n_evaluable,
            "correct_reuse_count": self.n_correct_hits,
            "incorrect_reuse_count": self.n_incorrect_hits,
            "unevaluable_reuse_count": self.n_unevaluable_hits,
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


class FIFOSemanticCache(SemanticCacheBaseline):
    """Fixed-threshold semantic lookup + First-In-First-Out eviction.

    Evicts the entry that has been in the cache the longest, no matter how often
    or how recently it was used (hits do not change its position). Entry keys are
    handed out in insertion order, so the smallest key is the oldest entry.
    """

    def _choose_eviction_victim(self) -> int:
        return min(self.entries)

    def _on_hit_update(self, key: int, step: int):
        pass  # FIFO ignores usage; only insertion order matters


class LIFOSemanticCache(SemanticCacheBaseline):
    """Fixed-threshold semantic lookup + Last-In-First-Out eviction.

    Evicts the entry inserted most recently. Once the cache is full, the first
    (capacity - 1) entries are never evicted and only the newest slot churns.
    Included as a deliberately weak reference point.
    """

    def _choose_eviction_victim(self) -> int:
        return max(self.entries)

    def _on_hit_update(self, key: int, step: int):
        pass  # LIFO ignores usage; only insertion order matters


class NoCacheBaseline:
    """Trivial baseline: every query hits the LLM. Upper bound on cost, floor on staleness."""

    def __init__(self, default_true_cost: float = 1.0):
        self.default_true_cost = default_true_cost
        self.n_queries = 0
        self.n_evaluable = 0
        self.total_cost = 0.0
        self.total_cost_nocache = 0.0

    def process(self, record: dict):
        true_cost = record.get("true_cost", self.default_true_cost)
        self.n_queries += 1
        if record.get("intent_id") is not None:
            self.n_evaluable += 1
        self.total_cost += true_cost
        self.total_cost_nocache += true_cost
        return "miss", None

    def metrics(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "hit_rate": 0.0,
            "stale_hit_rate": 0.0,
            "normalized_api_cost": 1.0,
            "false_hit_rate": 0.0 if self.n_evaluable else None,
            "recomputation_rate": 1.0 if self.n_queries else 0.0,
            "evaluable_requests": self.n_evaluable,
            "correct_reuse_count": 0,
            "incorrect_reuse_count": 0,
            "unevaluable_reuse_count": 0,
        }