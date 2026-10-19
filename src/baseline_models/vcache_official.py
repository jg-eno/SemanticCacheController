"""Run the authors' own vCache policy on a SemanticCacheController workload.

This is NOT a re-implementation. It plugs this repository's workload, encoder and
labels into the official `vcache` package (https://github.com/vcache-project/vCache,
`VerifiedDecisionPolicy`) through the package's public extension points and counts
the outcomes in this repository's metric format. The official code is imported
unmodified; none of it is copied here (its licence is CC BY-NC-ND 3.0).

How the pieces map
- embedding engine : this repo's SentenceEncoder (vectors are normalised)
- inference engine : a placeholder; no LLM is called
- similarity check : two answers are "the same" when their intent labels match
                     (same rule as the official benchmark's id_set comparison;
                     intent labels are a proxy for answer correctness)
- cache            : the official cache (HNSW search over ALL entries, no global
                     similarity threshold, no eviction), not SemanticCacheEnv's cache

Because the official policy owns its cache, this cannot run inside SemanticCacheEnv's
step loop; the numbers are comparable with the benchmark's metrics but come from a
different cache (no threshold gate, no capacity limit). The official policy also
updates its observations in background threads; `OfficialVCacheSystem` waits for each
update to finish before the next request so a seeded run is repeatable.

Requirements: the official package must be installed (Linux, Python 3.10-3.12; see
BASELINE.md, "Running the official vCache policy"), plus this repo's own dependencies.

    python -m src.baseline_models.vcache_official \
        --workload src/datasets/example_workload.jsonl --deltas 0.01 0.02 0.05
"""

import argparse
import json
import random
import threading
import time
from pathlib import Path

import numpy as np

INSTALL_HINT = (
    "The official vCache package is not importable. Install it from "
    "https://github.com/vcache-project/vCache on Linux with Python 3.10-3.12 "
    "(see BASELINE.md, 'Running the official vCache policy')."
)


class OfficialVCacheSystem:
    """The official VCache object plus the waiting needed for deterministic runs."""

    def __init__(self, vcache, policy, state, wait_timeout=60.0):
        self.vcache = vcache
        self.policy = policy
        self.state = state            # {"puts": int, "lock": Lock}: queue updates enqueued so far
        self.wait_timeout = wait_timeout
        self.expected_updates = 0

    def infer(self, prompt, id_set):
        """Return (is_hit, cached_id_set_or_None, explored) for one request."""
        is_hit, _response, _metadata, nearest = self.vcache.infer_with_cache_info(prompt, None, id_set)
        # An explore is a recompute whose nearest neighbour exists; the policy then
        # schedules exactly one background observation update.
        explored = (not is_hit) and nearest is not None and nearest.embedding_id != -1
        if explored:
            self.expected_updates += 1
        self._wait_for_updates()
        return bool(is_hit), (nearest.id_set if is_hit else None), explored

    def _wait_for_updates(self):
        queue = self.policy.callback_queue
        deadline = time.monotonic() + self.wait_timeout
        while True:
            with self.state["lock"]:
                enqueued = self.state["puts"]
            if enqueued >= self.expected_updates and queue.unfinished_tasks == 0:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"vCache background updates did not finish within {self.wait_timeout}s "
                    f"({enqueued}/{self.expected_updates} queued, {queue.unfinished_tasks} unfinished)"
                )
            time.sleep(0.0005)

    def close(self):
        self.policy.shutdown()


def build_system(encoder, delta, wait_timeout=60.0):
    """Create the official VCache with this repo's encoder and intent-label checks."""
    try:
        from vcache.config import VCacheConfig
        from vcache.inference_engine.inference_engine import InferenceEngine
        from vcache.main import VCache
        from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
        from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import (
            InMemoryEmbeddingMetadataStorage,
        )
        from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import HNSWLibVectorDB
        from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import NoEvictionPolicy
        from vcache.vcache_core.similarity_evaluator.similarity_evaluator import SimilarityEvaluator
        from vcache.vcache_policy.strategies.verified import VerifiedDecisionPolicy
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error

    class RepoEmbeddingEngine(EmbeddingEngine):
        def get_embedding(self, text):
            vector = np.asarray(encoder.encode(text), dtype=np.float64)
            return (vector / np.linalg.norm(vector)).tolist()

    class PlaceholderInferenceEngine(InferenceEngine):
        def create(self, prompt, system_prompt=None):
            return "answer:" + prompt

    class IntentLabelEvaluator(SimilarityEvaluator):
        def answers_similar(self, a, b, id_set_a=None, id_set_b=None):
            return id_set_a == id_set_b

    policy = VerifiedDecisionPolicy(delta=delta)
    config = VCacheConfig(
        inference_engine=PlaceholderInferenceEngine(),
        embedding_engine=RepoEmbeddingEngine(),
        vector_db=HNSWLibVectorDB(),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        eviction_policy=NoEvictionPolicy(),
        similarity_evaluator=IntentLabelEvaluator(),
    )
    vcache = VCache(config=config, policy=policy)

    # Count queued background updates so runs can wait for them (instrumentation only).
    state = {"puts": 0, "lock": threading.Lock()}
    queue = policy.callback_queue
    original_put = queue.put

    def counting_put(item, *args, **kwargs):
        original_put(item, *args, **kwargs)
        if item is not None:
            with state["lock"]:
                state["puts"] += 1

    queue.put = counting_put
    return OfficialVCacheSystem(vcache, policy, state, wait_timeout)


def run_workload(system, rows, rewards):
    """Send workload rows through `system` and return this repo's benchmark metrics."""
    id_sets = {}
    evaluable = incorrect = recomputed = explorations = 0
    total_reward = 0.0
    for row in rows:
        intent = row.get("intent_id")
        id_set = -1 if intent is None else id_sets.setdefault(str(intent), len(id_sets))
        hit, cached_id_set, explored = system.infer(row["query"], id_set)
        explorations += bool(explored)
        if not hit:
            status = "not_reused"
            recomputed += 1
        elif intent is None or cached_id_set in (None, -1):
            status = "unevaluable"
        else:
            status = "correct" if cached_id_set == id_set else "incorrect"
        evaluable += intent is not None and status != "unevaluable"
        incorrect += status == "incorrect"
        total_reward += sum(rewards.components(status, not hit).values())
    n = len(rows)
    if n == 0:
        raise ValueError("Cannot evaluate an empty workload")
    return dict(requests=n, evaluable_labeled_requests=evaluable,
                false_hit_rate=incorrect / evaluable if evaluable else None,
                recomputation_rate=recomputed / n, mean_environment_reward=total_reward / n,
                explorations=explorations)


def evaluate(rows, encoder, delta, rewards, seed=0, wait_timeout=60.0):
    random.seed(seed)        # the official policy draws its explore/exploit coin from `random`
    np.random.seed(seed)
    system = build_system(encoder, delta, wait_timeout)
    try:
        return run_workload(system, rows, rewards)
    finally:
        system.close()


def main():
    parser = argparse.ArgumentParser(description="Run the official vCache policy on a workload.")
    parser.add_argument("--workload", required=True, help="Workload JSONL (same format as the benchmark)")
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--output", default="benchmark/results_vcache_official.json")
    args = parser.parse_args()

    from src.baseline_models.evaluate_workload import CachedEncoder
    from src.datasets.workload import read
    from src.env.semantic_cache_env import Rewards
    from src.models.semantic_cache import SentenceEncoder

    rows = read(args.workload)
    encoder = CachedEncoder(SentenceEncoder(args.model, args.model_revision))
    rewards = Rewards()
    results = {f"official_vCache_delta_{delta}": evaluate(rows, encoder, delta, rewards, args.seed)
               for delta in args.deltas}
    report = dict(seed=args.seed, settings=dict(workload=args.workload, model=args.model,
                                                model_revision=args.model_revision),
                  results=results,
                  limitation="Official vCache policy with its own cache (no similarity gate, no eviction); "
                             "intent labels are a proxy for answer correctness.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()