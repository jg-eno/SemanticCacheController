# Semantic Cache Baselines

This folder contains the deterministic baseline policies used to evaluate a semantic-cache controller. The repository compares three policies using the same fixed-threshold similarity lookup:

- `NoCacheBaseline` — always serves a fresh result and pays the full cost.
- `LRUSemanticCache` — evicts the least recently used cached entry.
- `LFUSemanticCache` — evicts the least frequently used cached entry.

The baseline is intentionally limited to lookup, freshness checking, eviction, and summary metrics. It does not generate embeddings, call an LLM, or provide an end-to-end drift environment.

## Folder contents

- `caches.py` — shared cache logic and the baseline policies.
- `test_run.py` — old synthetic smoke test using the stub data generator.
- `test_stub_data.py` — creates a simple synthetic query stream for quick sanity checks.
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

The baseline adapter converts this to the internal record format by doing the following:

```python
record = {
    "embedding": encoder.encode(record["query"]),
    "true_cost": record.get("true_cost", default_true_cost),
    "content_version": record.get("content_version", record.get("intent_id")),
    "step": record["step"],
}
```

This lets the same cache logic consume either synthetic records or project data without changing LRU/LFU behavior.

## How the baseline works

For every query, the cache performs the following steps:

1. Calculate cosine similarity between the query embedding and each cached embedding.
2. Treat the query as a cache hit when the best similarity is at least the configured threshold.
3. Compare the cached content version with the current content version.
4. Count a hit as stale when those versions differ.
5. On a miss, pay the query's `true_cost` and insert the result into the cache.
6. Evict an entry according to the selected policy when the cache is full.

The default similarity threshold is `0.85`, and the default capacity is `50` entries. The default `NoCacheBaseline` does not use either value.

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
- `stale_hit_rate` — stale hits divided by all cache hits.
- `normalized_api_cost` — actual cost divided by the cost of processing every query without a cache.

A stale hit is a cache hit whose cached content version differs from the current query content version. It is not counted as a miss.

## How to run the real validation path

Use the repository venv and run the actual evaluation entry points from the repo root:

```powershell
.\venv\Scripts\Activate.ps1
python -m unittest discover -s src/tests -v
python -m src.evaluate src/datasets/example_workload.jsonl --held-out src/datasets/example_held_out.jsonl
```

These are the authoritative checks for the project. They use real workload rows, not the stub stream.

## How to run the synthetic stub sanity check

The stub data is optional and only intended as a quick smoke test. It is not the main validation path for the project.

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

## Replace the synthetic stream

To evaluate a real environment, provide records through the same `process(record)` interface. The cache code does not require knowledge of how those records were generated. A real implementation can replace the stub generator while keeping the existing cache classes unchanged.
