# Environment and benchmark walkthrough

## How a request moves through the environment

1. **Load and reset.** The environment reads a chronological workload. On reset, it creates an empty FIFO cache and clears its recent-history counters. The benchmark gives each strategy a fresh environment and the same request trace.
2. **Encode and look up.** The current query is embedded and normalized. The cache searches for its most similar entry by cosine similarity. An entry is a candidate only when its similarity meets the configured retrieval threshold. The cache holds a query, embedding, and response; intent labels stay outside cache entries.
3. **Build the observation.** The strategy receives five float32 values, in this order: `query_embedding_shift`, `query_novelty`, `recent_cache_miss_rate`, `recent_reuse_rate`, and `cache_occupancy`. Novelty is `(1 - similarity) / 2` when there is a candidate, or `1` when there is none.
4. **Choose an action.** The strategy returns `0` to request reuse or `1` to recompute. Reuse succeeds only if lookup found a candidate. Requesting reuse on a miss forces recomputation.
5. **Serve and update.** Successful reuse serves the cached response. Recomputation obtains a response, admits it to the cache, and evicts the oldest entry if the cache is full. Recent miss and reuse history is updated before the next observation.
6. **Score the outcome.** For successful reuse, the environment compares the incoming intent label with the cached entry's label: `correct`, `incorrect`, or `unevaluable` if either label is missing. Recomputation has status `not_reused`. The environment returns a reward, its components, and step information. Intent labels are for offline assessment; the strategy does not receive them in its observation.

The environment and cache behavior are implemented in [`src/env/semantic_cache_env.py`](src/env/semantic_cache_env.py) and [`src/models/semantic_cache.py`](src/models/semantic_cache.py).

## What is modular

The **reuse-versus-recompute strategy** is pluggable. Each entry in `examples/methods.json` names a `module:factory` and settings. The factory receives `seed` plus those settings and returns a function that accepts one observation and returns action `0` or `1`. `src.benchmark` loads each factory, runs its strategy in a fresh environment, and checks that the request IDs and phases match across runs. You can add a strategy without editing the benchmark.

The **cache storage and eviction policy** is currently fixed: `SemanticCacheEnv` constructs `FIFOCache` directly. Testing LRU or LFU as the underlying cache requires changing or adapting the environment. Changing a decision strategy alone does not replace FIFO.

## What drift contributes

`query_embedding_shift` compares two adjacent windows of query embeddings using fixed random projections, histograms, and Jensen–Shannon divergence. It is `0` until two complete windows are available. It uses query embeddings, not intent labels.

The supplied `drift_rule` requests reuse only when a candidate exists, novelty is at most `0.075`, and drift is at most `0.1`; otherwise it recomputes. A high drift value can therefore prevent a potentially incorrect reuse, but may increase recomputation. It cannot improve the current request's raw lookup hit: lookup happens before the strategy acts. Its decisions can change later cache contents through admissions and evictions.

The repository does not yet establish that drift improves correct reuse or the overall quality–cost tradeoff. To test its contribution, compare the drift rule with an otherwise identical novelty-only rule on the same held-out trace. Check incorrect reuse, correct reuse, recomputation, and reward together. [`EXPERIMENTS.md`](EXPERIMENTS.md) lists feature ablation as a planned experiment.

## Metrics in the shared benchmark

The current shared benchmark reports exactly five metrics **per strategy**:

| Metric | Calculation | Interpretation |
| --- | --- | --- |
| `requests` | Number of processed requests | Trace size; should match across strategies. |
| `evaluable_labeled_requests` | Labeled requests whose outcome is evaluable | Denominator for false-hit rate. A labeled recomputation counts; a reuse with a missing incoming or cached label does not. |
| `false_hit_rate` | Incorrect reuses / evaluable labeled requests | Fraction of evaluable requests served from an entry with a different intent label. `null` when the denominator is zero. This is not the fraction of cache hits that were wrong. |
| `recomputation_rate` | Actual recomputations / all requests | Includes forced recomputation when reuse was requested but no candidate existed. |
| `mean_environment_reward` | Sum of simulated rewards / all requests | Average under the configured correct-reuse reward and incorrect-reuse and recomputation penalties. |

The metric calculations are in [`src/benchmark.py`](src/benchmark.py). The shared benchmark does not currently report raw cache hit rate, serving latency, or monetary cost. Its results are from an offline simulation; intent labels are a proxy for answer correctness.

Run it from the repository root:

```sh
python -m src.benchmark --settings examples/benchmark.json --methods examples/methods.json
```

The JSON report is saved to `benchmark/results.json` by default. Use `--output path/to/report.json` for a different destination.
