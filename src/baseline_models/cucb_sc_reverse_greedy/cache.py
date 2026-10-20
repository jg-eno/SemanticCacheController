"""Semantic cache model from Section II of the paper (arXiv:2508.07675).

Paper-to-code mapping
---------------------
- Queries Q = {q_1..q_m}            -> integer ids 0..m-1
- Embeddings e(q) in R^{d_e}        -> numpy array `embeddings` of shape (m, d_e)
- Distance d(q1,q2)=||e(q1)-e(q2)||_2 (paper default, Sec. II-A)
                                    -> `distance_matrix`
  Sec. VI also min-max normalises d into [0,1]; see `normalize`.
- d(q, M) = min_{u in M} d(q, u)    -> `distance_to_set`
- s(q, M) = argmin_{u in M} d(q, u) -> `nearest_cached`
- Decision rule (Eq. 1): serve from cache iff d(q, M) < c(q), else call the LLM
                                    -> `serve_from_cache`
- Loss (Eq. 2): l(M; p, c, d) = sum_q p(q) * min{c(q), d(q, M)}
                                    -> `loss`
Mismatch scale gamma is set to 1, as in the paper.
"""

import numpy as np


def distance_matrix(embeddings, normalize=True):
    """Pairwise Euclidean distances d(q_i, q_j).

    If `normalize` is True, distances are divided by the largest pairwise
    distance. Because d(q, q) = 0 is the minimum, this is exactly min-max
    normalisation to [0, 1] as described in the paper's experimental setup.
    """
    diff = embeddings[:, None, :] - embeddings[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    if normalize and dist.max() > 0:
        dist = dist / dist.max()
    return dist


def distance_to_set(dist, cache):
    """d(q, M) for every query q. d(q, empty set) = +inf (nothing to reuse)."""
    cache = list(cache)
    if not cache:
        return np.full(dist.shape[0], np.inf)
    return dist[:, cache].min(axis=1)


def nearest_cached(dist, cache):
    """s(q, M): the cached query closest to each q."""
    cache = list(cache)
    return np.array(cache)[dist[:, cache].argmin(axis=1)]


def serve_from_cache(dist, cache, cost):
    """Eq. (1): True where reusing s(q, M) is cheaper than calling the LLM."""
    return distance_to_set(dist, cache) < cost


def loss(cache, p, c, dist):
    """Eq. (2): expected per-query cost of cache M under the optimal decision rule."""
    return float(np.sum(p * np.minimum(c, distance_to_set(dist, cache))))
