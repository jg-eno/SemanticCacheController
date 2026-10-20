# CUCB-SC + Reverse Greedy (offline semantic caching baseline)

Offline-learning baseline from

> X. Liu, B. Atalar, X. Dai, et al., *Semantic Caching for Low-Cost LLM Serving:
> From Offline Learning to Online Adaptation*, arXiv:2508.07675.

Implemented: Algorithm 1 (Reverse_Greedy) and Algorithm 2 (CUCB-SC). Not implemented: the
online algorithm CLCB-SC-LS (Algorithm 3).

| Paper | Code |
|---|---|
| `d(q1,q2) = ||e(q1)-e(q2)||_2`, min-max normalised to [0,1] | `cache.distance_matrix` |
| `d(q,M)`, `s(q,M)`, decision rule (Eq. 1), loss (Eq. 2) | `cache.py` |
| Algorithm 1 Reverse_Greedy | `reverse_greedy.reverse_greedy` |
| Algorithm 2 steps 3-7: `N`, `N_c`, `p_hat`, `c_hat`, UCB `c_bar`, Reverse_Greedy | `cucb_sc.py` |
| Embeddings `e(q)`: all-MiniLM-L6-v2 | `embed.py` |

## Standard benchmark run

```
python src/baseline_models/cucb_sc_reverse_greedy/evaluate_semantic_cache_workload.py
```

- Dataset: `jg-eno/SemanticCacheWorkload` at revision `0be7e5981d640493849471113ebbf4706d1f88c2`,
  committed in `data/semantic_cache_workload/` (downloaded at that revision only if missing).
- Learning: training split only (300 requests, 300 distinct queries, 100 `origin_id`s). Cache
  candidates, arrival frequencies, costs and the distance normalisation constant come from it.
- Evaluation: test split only (51 requests, 17 `origin_id`s), ordered exactly like
  `src/datasets/prepare_semantic_cache_workload.py` (seed 0, `origin_id -> intent_id`, `s0-r{step}`).
- Cost: uniform `C_t = 0.2`, the benchmark's `recompute_cost`. It is not a measured serving cost
  (the dataset has none).
- Replay: the selected cache is fixed. Reuse `s(q_t, M)` iff `d(q_t, M) < 0.2` (paper rule (1));
  a reuse is correct iff the cached query's `origin_id` equals the request's.
- Metrics: the shared benchmark's `requests`, `evaluable_labeled_requests`, `false_hit_rate`,
  `recomputation_rate`, `mean_environment_reward` (+1.0 correct, -2.0 incorrect, -0.2 recompute),
  plus the paper loss `l(M; p_test, 0.2, d)` separately.
- Output: `benchmark/results_cucb_sc.json` (includes the selected cache and per-request outcomes).
  Embeddings are cached in `data/semantic_cache_workload/embeddings_{train,test}.npz` (git-ignored).

Result (k = 32): 51 requests, 51 evaluable, false_hit_rate 0.0, recomputation_rate 1.0,
mean_environment_reward -0.2, 0 hits, paper loss 0.2.

## Protocol difference: not like-for-like with `SemanticCacheEnv` methods

The benchmark environment is online: its cache starts empty on the test sequence and admits every
recomputed answer, so paraphrases inside the test sequence can hit each other. CUCB-SC + Reverse Greedy
is offline: the cache is selected from the training split before the test sequence and never changes,
and reuse uses the paper's distance rule rather than the environment's cosine threshold. The training
and test splits share no query and no `origin_id`, so any reuse would be wrong-intent; with k=32 no test
query is within distance 0.2 of the cache (minimum 0.2286), so the method recomputes everything.
With one cost observation per query, `c_bar ≈ 3.05` for every query, so the UCB bonus is uniform and
Reverse Greedy effectively minimises `sum_q p_hat(q) d(q, M)`.

Implementation choices the paper does not specify: Reverse Greedy ties are broken by smallest query id;
`delta = 0.05`; k = 32 matches the benchmark's default capacity.
