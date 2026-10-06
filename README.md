# Semantic Cache Controller

# Evaluation Strategies

## 1. Cache Effectiveness

-   **Semantic Hit Rate:** cache hits / total queries.
-   **Cost-Weighted Hit Rate:** gives more weight to hits that save more
    tokens or latency.

## 2. Correctness of Reuse

-   **Precision, Recall and F1:** evaluate whether cached responses are
    actually reusable.
-   **False-Hit / Error Rate:** measure incorrect cached answers.
-   **Error-Bound Satisfaction:** verify that observed error stays below
    the target limit.
-   **P-CHR AUC:** evaluate precision across different cache-utilization
    levels.
-   **Answer Quality:** validate sampled cache hits using labeled query
    pairs or an LLM-as-judge.

## 3. Latency and Cost

-   Measure **mean, P50, P95 and P99 end-to-end latency**.
-   Report cache lookup latency separately.
-   Measure **cost per query, tokens saved and savings per hit**.

## 4. Freshness and Data Drift

-   **Stale-Hit Rate:** fraction of cache hits returning outdated
    answers.
-   **Invalidation Precision, Recall and F1.**
-   **Average Staleness Penalty.**

## 5. Query and Data Drift Adaptation

Evaluate performance before and after drift using: - Windowed hit rate,
precision and stale-hit rate. - Performance drop at drift. - **Recovery
Time:** queries/episodes needed to return close to pre-drift
performance. - **Threshold Trajectory:** how the learned threshold
changes with drift.

## 6. RL Performance

-   Cumulative and average reward.
-   Convergence speed.
-   Regret.
-   Cache switching count.
-   Training time and stability.
-   Run multiple seeds and report **mean ± 95% confidence interval**.

## 7. Controller Overhead

-   RL decision latency per query.
-   Memory footprint of the RL policy and cache.
-   **Net Benefit = latency/cost savings − controller overhead.**

## 8. Baseline Comparison

Compare the proposed controller with: - No Cache - FIFO - LRU - LFU -
ARC - Static/best tuned threshold - GPTCache-style semantic caching -
vCache-style learned threshold

Also include RL ablations: - **RL threshold only** - **RL eviction
only** - **No freshness signal**

## 9. Sensitivity and Robustness

Evaluate different: - Cache sizes - Similarity thresholds - Drift speeds
and magnitudes - Query workloads - Embedding models - Random seeds

## 10. Recommended Core Results

  Objective             Main Metric
  --------------------- ---------------------------------
  Cache effectiveness   Semantic Hit Rate
  Correctness           Precision, False-Hit Rate
  Latency               P50 / P95 / P99
  Cost                  Cost per Query
  Freshness             Stale-Hit Rate
  Drift adaptation      Recovery Time
  RL quality            Cumulative Reward / Convergence
  Overhead              Decision Latency / Net Benefit

### Main Evaluation Goal

The RL controller should **increase useful cache reuse and reduce
latency/cost while keeping incorrect and stale responses within an
acceptable limit**, especially under query and data drift.
