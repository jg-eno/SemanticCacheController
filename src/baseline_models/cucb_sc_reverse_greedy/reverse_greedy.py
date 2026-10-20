"""Algorithm 1 (Reverse_Greedy) from the paper, plus a brute-force reference.

Algorithm 1: Removing Least Beneficial Cache Items
  Input: queries Q, probabilities p, serving costs c, distance d, cache size k
  M_0 <- Q
  for i = 1 .. m-k:
      compute l(M_{i-1} \\ {q}) for all q in M_{i-1}
      q_i = argmin_{q in M_{i-1}} l(M_{i-1} \\ {q})
      M_i <- M_{i-1} \\ {q_i}
  return M_{m-k}

Ties in the argmin are broken by the smallest query id so results are
deterministic (the paper does not specify a tie-breaking rule).
"""

from itertools import combinations
from math import comb

from cache import loss


def reverse_greedy(p, c, dist, k):
    m = len(p)
    if not 1 <= k <= m:
        raise ValueError(f"cache size k must be in [1, {m}], got {k}")

    cache = list(range(m))                         # M_0 <- Q
    for _ in range(m - k):                         # for i = 1 .. m-k
        # l(M_{i-1} \ {q}) for every candidate q still in the cache
        losses = [loss([u for u in cache if u != q], p, c, dist) for q in cache]
        q_i = cache[losses.index(min(losses))]     # argmin (first index on ties)
        cache.remove(q_i)                          # M_i <- M_{i-1} \ {q_i}
    return sorted(cache)


def brute_force(p, c, dist, k, max_subsets=2_000_000):
    """Exact M* (Eq. 3) by enumerating all size-k caches.

    Size exactly k is sufficient because the loss is non-increasing in M
    (Lemma 2). Used only as a reference, as in the paper's Fig. 2(a).
    """
    m = len(p)
    if comb(m, k) > max_subsets:
        return None
    best = min(combinations(range(m), k), key=lambda s: loss(s, p, c, dist))
    return sorted(best)
