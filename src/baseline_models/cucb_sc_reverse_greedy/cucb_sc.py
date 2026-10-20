"""Algorithm 2 (CUCB-SC) from the paper: offline learning for semantic caching.

Offline dataset D = {(M_t, q_t, C_t)}_{t=1..n}:
  q_t  = the query that arrived at step t
  C_t  = the realised serving cost C_t(q_t) if the LLM was called, or
         "empty" (None here) if the query was served from the cache M_t.
  M_t  = the logging system's cache at step t. It only explains *why* C_t may be
         missing; Algorithm 2's formulas never use it, so our log format does
         not need to store it.

Algorithm 2 steps -> code
  Step 3: N(q)   = sum_t 1{q = q_t}                       -> `count_arrivals`
          N_c(q) = sum_t 1{q = q_t and C_t != empty}       -> `count_cost_observations`
  Step 4: p_hat(q) = N(q) / n                              -> `estimate_arrival_probs`
          c_hat(q) = sum of observed C_t for q / N_c(q)    -> `estimate_costs`
  Step 5: c_bar(q) = c_hat(q) + sqrt( log(6 m n / delta) / (2 N_c(q)) )
                                                           -> `ucb_costs`
  Step 7: A_hat = Reverse_Greedy(Q, p_hat, c_bar, d, k)    -> `cucb_sc`
  Step 8: call the LLM to get a(q) for q in A_hat. Not done here: we only
          select *which* queries to cache; response generation is outside
          the scope of this baseline.
"""

import math
import warnings

import numpy as np

from reverse_greedy import reverse_greedy


def count_arrivals(query_ids, m):
    return np.bincount(query_ids, minlength=m).astype(float)


def count_cost_observations(query_ids, costs, m):
    observed = np.array([cost is not None for cost in costs])
    return np.bincount(query_ids[observed], minlength=m).astype(float)


def estimate_arrival_probs(query_ids, m):
    return count_arrivals(query_ids, m) / len(query_ids)


def estimate_costs(query_ids, costs, m):
    """Empirical mean of observed costs; NaN where N_c(q) = 0."""
    totals = np.zeros(m)
    for q, cost in zip(query_ids, costs):
        if cost is not None:
            totals[q] += cost
    n_c = count_cost_observations(query_ids, costs, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        return totals / n_c


def ucb_costs(c_hat, n_c, m, n, delta):
    """Pessimistic (upper-confidence) cost: c_hat + sqrt(log(6mn/delta) / 2N_c).

    If a query's cost was never observed (N_c(q) = 0) the bonus is unbounded,
    so c_bar(q) = +inf. Theorem 2 assumes nu(q) > 0 and enough data so this
    does not happen; we warn instead of inventing a value.
    """
    bonus = np.full(m, np.inf)
    seen = n_c > 0
    bonus[seen] = np.sqrt(math.log(6 * m * n / delta) / (2 * n_c[seen]))
    c_bar = np.where(seen, c_hat + bonus, np.inf)
    if not seen.all():
        warnings.warn(f"{(~seen).sum()} queries have no observed cost; c_bar = inf for them")
    return c_bar


def cucb_sc(query_ids, costs, dist, k, delta):
    """Run Algorithm 2 and return the selected cache plus all estimates."""
    query_ids = np.asarray(query_ids)
    m, n = dist.shape[0], len(query_ids)

    n_c = count_cost_observations(query_ids, costs, m)
    p_hat = estimate_arrival_probs(query_ids, m)
    c_hat = estimate_costs(query_ids, costs, m)
    c_bar = ucb_costs(c_hat, n_c, m, n, delta)

    cache = reverse_greedy(p_hat, c_bar, dist, k)
    return {
        "cache": cache,
        "N": count_arrivals(query_ids, m),
        "N_c": n_c,
        "p_hat": p_hat,
        "c_hat": c_hat,
        "c_bar": c_bar,
    }
