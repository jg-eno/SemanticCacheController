"""Evaluate NoCache/FIFO/LIFO/LRU/LFU baselines on project workload rows.

Example:
    python -m src.baseline_models.evaluate_workload 
    src/datasets/example_workload.jsonl  
    --held-out src/datasets/example_held_out.jsonl
"""

import argparse
import json

from src.baseline_models.caches import (FIFOSemanticCache, LFUSemanticCache, LIFOSemanticCache,
                                        LRUSemanticCache, NoCacheBaseline)
from src.datasets.workload import read
from src.models.semantic_cache import SentenceEncoder


class CachedEncoder:
    """Embed each distinct query string once, however many baselines/workloads reuse it.

    The baselines are deterministic and never mutate embeddings, so sharing the
    same array across caches is safe. Without this, every row is embedded once
    per baseline per workload (LRU and LFU each re-encode the same text).
    """

    def __init__(self, encoder):
        self.encoder, self.memo, self.calls = encoder, {}, 0

    def encode(self, query):
        if query not in self.memo:
            self.calls += 1
            self.memo[query] = self.encoder.encode(query)
        return self.memo[query]


BASELINES = (NoCacheBaseline, FIFOSemanticCache, LIFOSemanticCache, LRUSemanticCache, LFUSemanticCache)


def evaluate_all(rows, encoder, capacity=32, threshold=0.85):
    """Run every baseline on `rows`; returns {class_name: metrics} (no duplicated labels)."""
    return {cls.__name__: evaluate_cache_rows(cls, rows, encoder, capacity, threshold)["metrics"]
            for cls in BASELINES}


def evaluate_cache_rows(cache_cls, rows, encoder, capacity=32, threshold=0.85):
    """Process workload rows in-order and return summary metrics."""
    if cache_cls is NoCacheBaseline:
        cache = cache_cls()
    else:
        cache = cache_cls(capacity=capacity, similarity_threshold=threshold, encoder=encoder)
    for row in rows:
        cache.process(row)
    return {
        "cache": cache_cls.__name__,
        "metrics": cache.metrics(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate semantic-cache baseline policies against JSONL workload rows.'
    )
    parser.add_argument('workload', help='Path to the workload JSONL file')
    parser.add_argument('--held-out', help='Optional second workload JSONL file for validation')
    parser.add_argument('--capacity', type=int, default=32, help='Semantic cache capacity')
    parser.add_argument('--threshold', type=float, default=0.85, help='Cosine similarity threshold')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='SentenceTransformer model name')
    parser.add_argument('--model-revision', default=None, help='Optional model revision')
    args = parser.parse_args()

    encoder = CachedEncoder(SentenceEncoder(args.model, args.model_revision))
    rows = read(args.workload)
    results = {
        # Settings that determine the numbers below; keep them next to the results.
        "config": {
            "workload": args.workload, "held_out": args.held_out,
            "capacity": args.capacity, "threshold": args.threshold,
            "model": args.model, "model_revision": args.model_revision,
            "default_true_cost": 1.0,   # workload rows carry no true_cost, so every miss costs 1.0
        },
        "workload": evaluate_all(rows, encoder, args.capacity, args.threshold),
    }
    if args.held_out:
        results["held_out"] = evaluate_all(read(args.held_out), encoder, args.capacity, args.threshold)

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()