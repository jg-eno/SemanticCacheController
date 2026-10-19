"""vCache baseline: verified semantic caching with per-entry learned thresholds.

Re-implementation of the decision policy from Schroeder et al., "vCache: Verified
Semantic Prompt Caching" (arXiv 2502.03771), written for this repository's
Gymnasium environment. It is NOT the authors' code.

Idea (paper, Sec. 4): instead of one global similarity threshold, every cached
entry y keeps its own observations O(y) = {(similarity, was_reuse_correct)} from
earlier requests whose nearest neighbour was y. From them it fits a sigmoid
    P(correct | s) = 1 / (1 + exp(-gamma * (s - t)))
and, for a new request with similarity s to y, computes an exploration
probability tau from the user error bound delta:
    tau = min over eps of ((1 - delta) - a) / (1 - a),   a = (1 - eps) * P(s; t'(eps), gamma)
where t'(eps) is a pessimistic (high) threshold from a (1 - eps) confidence bound.
With probability tau the request is explored (recomputed, the answer compared with
the cached one, and an observation recorded); otherwise the cached answer is reused.

Choices made here (the paper leaves them to the implementation):
- gamma and t are fitted by a deterministic grid search over binary cross-entropy.
- t'(eps) is the (1 - eps) quantile of a grid posterior over t with a uniform prior
  and gamma fixed at its point estimate (the paper's experiments also put the
  confidence interval on t only).
- The "LLM call" on the explore path is simulated from intent labels: the cached
  answer counts as correct when the incoming and cached intent labels match.
  The label is read only on the explore path, never on the exploit path.
- The environment always inserts a recomputed answer; the paper inserts it only
  when the cached answer was wrong. After a correct explore, the entry the
  environment inserts therefore shares its parent's observations instead of
  starting empty (see VCacheMethod.select_action).

Action convention of the environment: REUSE = 0, RECOMPUTE = 1.
"""

import numpy as np

REUSE, RECOMPUTE = 0, 1

_T_GRID = np.linspace(0.0, 1.0, 201)
_GAMMA_GRID = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
_EPS_GRID = np.linspace(0.01, 0.99, 50)


def _sigmoid(s, t, gamma):
    return 1.0 / (1.0 + np.exp(-gamma * (s - t)))


def _log_likelihood_over_t(sims, labels, gamma):
    """Log-likelihood of the observations for every t on the grid (gamma fixed)."""
    z = gamma * (sims[:, None] - _T_GRID[None, :])          # (n, |t|)
    log_p = -np.logaddexp(0.0, -z)                          # log sigmoid(z)
    log_not_p = -np.logaddexp(0.0, z)                       # log (1 - sigmoid(z))
    c = labels[:, None]
    return (c * log_p + (1.0 - c) * log_not_p).sum(axis=0)


class EntryModel:
    """Observations and sigmoid model of one cached entry."""

    def __init__(self):
        self.sims = []
        self.labels = []

    def add(self, similarity, correct):
        self.sims.append(float(similarity))
        self.labels.append(1.0 if correct else 0.0)

    def fit(self):
        """Return (t_hat, gamma_hat) minimising binary cross-entropy; None if no data."""
        if not self.sims:
            return None
        sims, labels = np.array(self.sims), np.array(self.labels)
        best = None
        for gamma in _GAMMA_GRID:
            ll = _log_likelihood_over_t(sims, labels, gamma)
            i = int(np.argmax(ll))      # first maximum: lowest t on ties, deterministic
            if best is None or ll[i] > best[0] + 1e-12:
                best = (ll[i], _T_GRID[i], gamma)
        return best[1], best[2]

    def exploration_probability(self, similarity, delta):
        """tau: probability of recomputing (exploring) instead of reusing."""
        fitted = self.fit()
        if fitted is None:
            return 1.0 - delta           # nothing known: explore (paper's formula with a = 0)
        _, gamma = fitted
        sims, labels = np.array(self.sims), np.array(self.labels)
        ll = _log_likelihood_over_t(sims, labels, gamma)
        posterior = np.exp(ll - ll.max())
        cdf = np.cumsum(posterior) / posterior.sum()
        best = 1.0
        for eps in _EPS_GRID:
            t_prime = _T_GRID[min(int(np.searchsorted(cdf, 1.0 - eps)), len(_T_GRID) - 1)]
            a = (1.0 - eps) * _sigmoid(similarity, t_prime, gamma)
            tau = ((1.0 - delta) - a) / (1.0 - a) if a < 1.0 else 0.0
            best = min(best, tau)
        return float(np.clip(best, 0.0, 1.0))


class VCacheMethod:
    """Benchmark method: reuse a cached answer unless vCache decides to explore."""

    def __init__(self, seed=0, delta=0.02):
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.models = {}                 # cached entry object -> EntryModel
        self._alias_of = None            # model the next inserted entry should share
        self.explored = 0                # diagnostics
        self.exploited = 0

    @staticmethod
    def _newest_entry(cache):
        entries = cache.entries
        if isinstance(entries, dict):    # baseline_models caches
            return next(reversed(entries.values()), None)
        return entries[-1] if entries else None   # models.semantic_cache.FIFOCache (list)

    def select_action(self, observation, env):
        # The environment inserts every recomputed answer. After a correct explore that
        # entry duplicates its parent, so it shares the parent's observations, as in the
        # paper, where a correct explore adds no new entry.
        if self._alias_of is not None:
            newest = self._newest_entry(env.cache)
            if newest is not None and newest not in self.models:
                self.models[newest] = self._alias_of
            self._alias_of = None
        entry = env.candidate
        if entry is None:
            return RECOMPUTE             # no nearest neighbour above the cache threshold
        similarity = float(np.clip(env.similarity, 0.0, 1.0))
        model = self.models.setdefault(entry, EntryModel())
        tau = model.exploration_probability(similarity, self.delta)
        if self.rng.uniform() > tau:
            self.exploited += 1
            return REUSE
        # Explore: the LLM is called and its answer compared with the cached one.
        self.explored += 1
        incoming = env.workload[env.index]["intent_id"]
        cached = env.cached_intents.get(entry)
        if incoming is not None and cached is not None:
            correct = incoming == cached
            model.add(similarity, correct)
            if correct:
                self._alias_of = model
        return RECOMPUTE


def make_vcache_method(seed=0, delta=0.02):
    return VCacheMethod(seed=seed, delta=delta)