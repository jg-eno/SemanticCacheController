# Semantic Cache Baselines

This folder contains a small, reproducible baseline for semantic caching. It compares three policies using the same fixed-threshold similarity lookup:

- `NoCacheBaseline` — always serves a fresh result and pays the full cost.
- `LRUSemanticCache` — evicts the least recently used cached entry.
- `LFUSemanticCache` — evicts the least frequently used cached entry.

The baseline is intentionally limited to cache lookup, freshness checking, eviction, and metrics. It does not generate embeddings, call an LLM, or provide the full drift environment.

## Folder contents

- `caches.py` — shared cache logic and the three baseline policies.
- `test_run.py` — runs the two synthetic scenarios and prints metrics.
- `test_stub_data.py` — creates the synthetic query stream used by the runner.
- `BASELINE.md` — this guide.

## How the baseline works

Each query record must contain:

```python
{
    "embedding": np.ndarray,
    "true_cost": float,
    "content_version": object,
    "step": int,
}
```

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

## Run the baseline

### Prerequisites

Use Python 3.13 or another Python version supported by your environment. The baseline requires NumPy.

From the repository root, create and activate a virtual environment if desired:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy
```

### Run the sanity-check scenarios

From the repository root:

```powershell
python src/baseline_models/test_run.py
```

The runner executes two scenarios:

1. **Scenario A:** capacity `50`, fewer than `50` semantic clusters. No eviction occurs.
2. **Scenario B:** capacity `5`, more semantic clusters. Eviction occurs and LRU and LFU can produce different results.

The runner prints hit rate, stale-hit rate, and normalized API cost for `NoCache`, `LRU`, and `LFU`.

### Run the tests directly

You can also run the synthetic data generator and runner through Python from the baseline folder:

```powershell
cd src/baseline_models
python test_run.py
```

The first command is recommended when running from the repository root.

### Use the cache classes in another script

Import the classes from the baseline folder after adding that folder to your Python path, or run the script from the folder:

```python
from caches import LRUSemanticCache, LFUSemanticCache, NoCacheBaseline

cache = LRUSemanticCache(capacity=50, similarity_threshold=0.85)
result = cache.process(record)
print(result)
print(cache.metrics())
```

The `record` must use the required fields described above. The current synthetic generator can be used as a reference:

```python
from test_stub_data import make_stub_stream

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

To evaluate a real environment, provide records through the same `process(record)` interface. The cache code does not require knowledge of how those records were generated. A real implementation can replace `test_stub_data.py` while keeping the existing cache classes unchanged.