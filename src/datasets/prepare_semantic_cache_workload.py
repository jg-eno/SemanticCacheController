"""Turn the jg-eno/SemanticCacheWorkload Hugging Face dataset into a benchmark workload.

The dataset has columns `query` and `origin_id` (queries that are paraphrases of one
another share an `origin_id`). This writes the repository's workload JSONL, with
`origin_id` as `intent_id`, after a seeded shuffle, so request order carries no phase
structure. Shuffling matters: in the dataset's stored order, paraphrases of one prompt
sit next to each other, which favours recency-based eviction.

    python -m src.datasets.prepare_semantic_cache_workload \
        --split train --seed 0 --output src/datasets/semantic_cache_train_shuffled.jsonl

Needs the `datasets` package and network access to Hugging Face.
"""

import argparse
import random

from src.datasets.workload import validate, write

DATASET = "jg-eno/SemanticCacheWorkload"


def convert(rows, seed=0, shuffle=True):
    """Map dataset rows to workload rows; deterministic for a given seed."""
    rows = list(rows)
    for row in rows:
        if "query" not in row or "origin_id" not in row:
            raise ValueError("Expected dataset columns 'query' and 'origin_id'")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError("Queries must be nonempty strings")
        if row["origin_id"] is None:
            raise ValueError("Every row needs an origin_id")
    if shuffle:
        random.Random(seed).shuffle(rows)
    workload = [dict(request_id=f"s{seed}-r{step}", intent_id=str(row["origin_id"]),
                     query=row["query"], topic_id=None, phase_id=0, step=step)
                for step, row in enumerate(rows)]
    validate(workload, require_labels=True)
    return workload


def summarize(workload):
    ids = [row["intent_id"] for row in workload]
    return dict(requests=len(ids), distinct_intents=len(set(ids)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true", help="keep the dataset's stored order")
    parser.add_argument("--revision", help="dataset commit revision for reproducible downloads")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from datasets import load_dataset
    rows = load_dataset(DATASET, split=args.split, revision=args.revision)
    workload = convert(rows, seed=args.seed, shuffle=not args.no_shuffle)
    write(args.output, workload)
    print(args.output, summarize(workload))


if __name__ == "__main__":
    main()