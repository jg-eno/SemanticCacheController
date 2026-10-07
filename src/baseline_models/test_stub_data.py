"""
test_stub_data.py

*** THIS IS TEST SCAFFOLDING ONLY -- NOT THE PROJECT'S GYM. ***

The real query-stream / drift environment ("the gym") is being built
separately by a teammate. This file exists purely so that `caches.py`
(the LRU/LFU/vCache/MAB baseline logic) can be exercised and sanity-
checked in isolation, without waiting on the real gym to be ready.

No parameter in this file is tuned for realism, and no result produced
using this file should ever be reported as a project result -- see
decision_log.md, entries D1 and D7.

Once the real gym exists, `caches.py` should be pointed at its output
instead; nothing in `caches.py` needs to change, since it only depends
on the record interface documented in `SemanticCacheBaseline.process`'s
docstring (embedding, true_cost, content_version, step).
"""

import numpy as np


def make_stub_stream(n_records: int = 2000, n_clusters: int = 10,
                      embedding_dim: int = 8, content_flip_every: int = 400,
                      seed: int = 0):
    """
    Yields a list of query records with a *very* simple, arbitrary
    structure: n_clusters fixed random directions, queries are noisy
    draws from a randomly-chosen cluster, and one cluster's content
    version flips periodically (just enough to exercise the
    stale-hit-tracking code path). This is deliberately simplistic --
    it is a unit-test fixture, not an experimental environment.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, embedding_dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    content_versions = [0] * n_clusters

    records = []
    for step in range(n_records):
        if content_flip_every and step > 0 and step % content_flip_every == 0:
            flip_idx = rng.integers(0, n_clusters)
            content_versions[flip_idx] += 1

        cluster = rng.integers(0, n_clusters)
        emb = centers[cluster] + rng.normal(0, 0.05, size=embedding_dim)
        emb /= np.linalg.norm(emb)
        cost = float(rng.uniform(0.1, 0.9))

        records.append({
            "embedding": emb,
            "true_cost": cost,
            "content_version": content_versions[cluster],
            "step": step,
        })
    return records
