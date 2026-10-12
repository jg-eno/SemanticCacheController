"""Label-free window drift from fixed random projection histograms."""
import numpy as np
from .semantic_cache import unit


def js_divergence(p, q):
    """Base-2 Jensen–Shannon divergence, not its square-root distance."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if (p.ndim != 1 or p.shape != q.shape or not p.size or
            not np.isfinite(p).all() or not np.isfinite(q).all() or
            np.any(p < 0) or np.any(q < 0) or p.sum() <= 0 or q.sum() <= 0):
        raise ValueError('Histograms must be finite, nonnegative, equal-size vectors with positive mass')
    p, q = p / p.sum(), q / q.sum()
    midpoint = (p + q) / 2
    divergence = 0.0
    for distribution in (p, q):
        mask = distribution > 0
        divergence += 0.5 * np.sum(distribution[mask] * np.log2(distribution[mask] / midpoint[mask]))
    return float(np.clip(divergence, 0, 1))


class ProjectionHistogramDrift:
    def __init__(self, window=16, projections=8, bins=8, seed=0):
        for name, value, minimum in [('window', window, 1), ('projections', projections, 1), ('bins', bins, 2)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError('projection seed must be a nonnegative integer')
        self.window, self.projections, self.bins, self.seed = window, projections, bins, seed
        self.directions = None
        self.edges = np.linspace(-3, 3, bins + 1)

    def _project(self, embeddings):
        vectors = np.stack([unit(e) for e in embeddings])
        dimension = vectors.shape[1]
        if self.directions is None:
            directions = np.random.default_rng(self.seed).normal(size=(self.projections, dimension))
            self.directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        if self.directions.shape[1] != dimension:
            raise ValueError('Embedding dimensions must remain fixed')
        # Dimension scaling avoids collapsing high-dimensional vectors near zero.
        return np.clip(vectors @ self.directions.T * np.sqrt(dimension), -3, 3)

    def score(self, history, current):
        """Compare adjacent full windows ending at current; never mutate history."""
        recent = list(history)[-(2 * self.window - 1):]
        projected = self._project(recent + [current])
        if len(projected) < 2 * self.window:
            return 0.0
        previous, incoming = projected[:self.window], projected[self.window:]
        scores = []
        for axis in range(self.projections):
            p = np.histogram(previous[:, axis], bins=self.edges)[0]
            q = np.histogram(incoming[:, axis], bins=self.edges)[0]
            scores.append(js_divergence(p, q))
        return float(np.mean(scores))
