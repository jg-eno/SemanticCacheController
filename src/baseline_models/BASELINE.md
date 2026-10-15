# Semantic Cache Baselines

This folder contains deterministic cache-policy baselines using fixed-threshold similarity lookup:

- `NoCacheBaseline` — always serves a fresh result and pays the full cost.
- `FIFOSemanticCache` — evicts the oldest inserted entry.
- `LIFOSemanticCache` — evicts the newest inserted entry.
- `LRUSemanticCache` — evicts the least recently used cached entry.
- `LFUSemanticCache` — evicts the least frequently used cached entry.

These classes provide standalone lookup, reuse-quality, eviction, and summary-metric simulation. They do not call an LLM. They are separate from the shared environment benchmark: `src.benchmark` evaluates observation-to-action methods in `SemanticCacheEnv`, whose cache currently uses FIFO eviction. Its methods JSON cannot select these baseline cache classes as eviction policies.

## Folder contents

- `caches.py` — shared cache logic and the baseline policies.
- `evaluate_workload.py` — evaluates all five cache baselines on workload JSONL rows.
- `test_run.py` and `test_stub_data.py` — optional synthetic smoke-test runner and data generator; not benchmark evidence.
- `BASELINE.md` — this guide.

## Internal cache contract

The cache logic accepts two shapes of input records.

### 1) Direct cache record

This is the internal baseline format used by the cache logic itself:

```python
{
    "embedding": np.ndarray,
    "true_cost": float,
    "content_version": object,
    "step": int,
}
```

Here:
- `embedding` is the query embedding vector used in cosine similarity checks
- `true_cost` is the fresh-serving cost on a miss
- `content_version` tracks whether the cached answer is stale
- `step` is a monotonically increasing counter used by LRU/LFU bookkeeping

### 2) Project workload row

The project dataset rows are not in this exact internal shape. They look like:

```python
{
    "request_id": "s0-r0",
    "intent_id": "example-capital",
    "query": "Name France's capital.",
    "topic_id": None,
    "phase_id": 0,
    "step": 0,
}
```

The cache adapter converts this to the internal record format. It encodes `query`, uses `true_cost` or the configured default, and preserves the optional `content_version` and `intent_id` fields. A missing `content_version` stays unknown; it is not inferred from `intent_id` because intent labels identify query equivalence, not answer freshness.

This lets the same cache logic consume either synthetic records or project workload rows without changing the eviction behavior.

## How the baseline works

For every query, the cache performs the following steps:

1. Calculate cosine similarity between the query embedding and each cached embedding.
2. Treat the query as a cache hit when the best similarity is at least the configured threshold.
3. Compare the cached content version with the current content version.
4. Count a hit as stale when those versions differ.
5. On a miss, pay the query's `true_cost` and insert the result into the cache.
6. Evict an entry according to the selected policy when the cache is full.

The default similarity threshold is `0.85`, and the default capacity is `32` entries. `evaluate_workload.py` uses the same defaults. `NoCacheBaseline` does not use either value.

### Cache policies

#### `NoCacheBaseline`

This policy always misses. It is useful as a reference because every query incurs the full cost and no cache entry is reused.

#### `LRUSemanticCache`

This policy uses the most recent query step for each cached entry. When the cache is full, it evicts the entry with the smallest `last_used_step`.

#### `LFUSemanticCache`

This policy counts cache hits for each entry. When the cache is full, it evicts the entry with the smallest `hit_count`.

If multiple entries tie on the eviction criterion, the earliest inserted entry is evicted. This deterministic tie-breaker makes repeated runs stable.

## Metrics

`metrics()` returns:

- `n_queries` — number of processed records.
- `hit_rate` — cache hits divided by total queries.
- `stale_hit_rate` — known stale hits divided by all cache hits. Without explicit content versions, staleness cannot be determined and no stale hits are countezd; a reported `0.0` does not establish that answers were fresh.
- `normalized_api_cost` — actual cost divided by the cost of processing every query without a cache.
- `false_hit_rate` — incorrect reuse divided by evaluable labeled requests. A `null` value means no request could be evaluated.
- `recomputation_rate` — requests recomputed divided by all requests.
- `evaluable_requests`, `correct_reuse_count`, `incorrect_reuse_count`, and `unevaluable_reuse_count` — reuse-quality counts.

A stale hit is a cache hit whose cached content version differs from the current query content version. It is not counted as a miss.

## How to run the real validation path

From the repository root, use the repository virtual environment to evaluate these cache classes on workload rows:

```powershell
.\venv\Scripts\Activate.ps1
python -m src.baseline_models.evaluate_workload src/datasets/example_workload.jsonl --held-out src/datasets/example_held_out.jsonl
python -m unittest src.tests.test_baseline_caches -v
```

The evaluator prints metrics for `NoCacheBaseline`, FIFO, LIFO, LRU, and LFU on the workload and optional held-out workload. The default sentence encoder may download its model on first use. `python -m src.evaluate` evaluates action policies in the Gymnasium environment; it does not run these five cache classes. The shared benchmark in `src.benchmark` also evaluates action methods, and currently runs them against the environment's FIFO cache. Use that benchmark for action-policy comparisons, not as a direct LRU/LFU comparison.

## How to run the synthetic stub sanity check

The stub data is optional and only intended as a quick smoke test of the standalone cache logic. It is not the workload evaluator or shared benchmark.

From the repository root:

```powershell
python src/baseline_models/test_run.py
```

The runner executes two scenarios:

1. **Scenario A:** capacity `50`, fewer than `50` semantic clusters. No eviction occurs.
2. **Scenario B:** capacity `5`, more semantic clusters. Eviction occurs and LRU and LFU can produce different results.

The runner prints hit rate, stale-hit rate, and normalized API cost for `NoCache`, `LRU`, and `LFU`.

### Use the cache classes in another script

Import the classes from the baseline folder or use the repository package path:

```python
from src.baseline_models.caches import LRUSemanticCache, LFUSemanticCache, NoCacheBaseline

cache = LRUSemanticCache(capacity=50, similarity_threshold=0.85, encoder=encoder)
result = cache.process(record)
print(result)
print(cache.metrics())
```

The `record` may be either:
- a direct cache record with `embedding`, `true_cost`, `content_version`, `step`, or
- a project workload row with `query`, `step`, and optional metadata.

The stub generator can still be used as a reference for synthetic sanity checks:

```python
from src.baseline_models.test_stub_data import make_stub_stream

records = make_stub_stream(
    n_records=2000,
    n_clusters=10,
    content_flip_every=400,
    seed=0,
)
```

## Expected behavior

- `NoCacheBaseline` should always report a hit rate of `0.0` and normalized API cost of `1.0`.
- With no eviction pressure, LRU and LFU should have similar metrics because they do not need to choose between entries.
- Under eviction pressure, LRU and LFU should differ because they select different victims.
- Stale-hit rate should increase when cached content versions become outdated.

The synthetic values are only sanity-check output. They are not benchmark results and should not be used as evidence about a real embedding model, workload, or API cost model.

## Input boundary

The standalone cache classes accept direct cache records or project workload rows through `process(record)`. They do not consume Gymnasium observations or actions. The shared benchmark uses the environment's own cache implementation, so comparing these eviction policies within that benchmark requires making the environment's eviction policy configurable and running each policy on the same workload and settings.
