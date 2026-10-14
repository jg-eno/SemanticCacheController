
"""Deterministic policies and simulation-only reports."""

import argparse
from collections import Counter
import json
import time

import numpy as np

from .env.semantic_cache_env import (
    SemanticCacheEnv,
    REUSE,
    RECOMPUTE,
    FEATURE_NAMES,
)
from .models.semantic_cache import SentenceEncoder, cosine
from .datasets.workload import read, generate
from .evaluation_metrics import summarize


def always_reuse(observation):
    """Reuse when the cache has a candidate."""
    return REUSE if observation[1] < 1 else RECOMPUTE


def always_recompute(observation):
    """Always recompute the response."""
    return RECOMPUTE


def drift_rule(
    observation,
    shift_threshold=0.1,
    novelty_threshold=0.075,
):
    """Recompute when drift or novelty exceeds the thresholds."""
    return (
        REUSE
        if observation[1] < 1
        and observation[0] <= shift_threshold
        and observation[1] <= novelty_threshold
        else RECOMPUTE
    )


POLICIES = {
    "always_reuse": always_reuse,
    "always_recompute": always_recompute,
    "drift_rule": drift_rule,
}


def evaluate(
    workload,
    encoder,
    policy,
    seed=0,
    transition_window=3,
    include_timing=False,
    **config,
):
    """Evaluate a policy on a workload.

    Timing statistics are excluded by default to keep reports
    reproducible. Set include_timing=True to include wall-clock metrics.
    """
    if transition_window < 1:
        raise ValueError("transition_window must be positive")

    env = SemanticCacheEnv(
        workload,
        encoder=encoder,
        **config,
    )

    obs, _ = env.reset(seed=seed)

    records = []
    rewards = []
    end_to_end_latencies_ms = []
    decision_latencies_ms = []

    while True:
        query_start = time.perf_counter()

        decision_start = time.perf_counter()
        action = policy(obs.copy())
        decision_ms = (
            time.perf_counter() - decision_start
        ) * 1000

        obs, reward, ended, truncated, info = env.step(action)

        end_to_end_ms = (
            time.perf_counter() - query_start
        ) * 1000

        records.append(info)
        rewards.append(reward)
        decision_latencies_ms.append(decision_ms)
        end_to_end_latencies_ms.append(end_to_end_ms)

        if ended or truncated:
            break

    n = len(records)

    if n == 0:
        raise ValueError("Cannot evaluate an empty workload")

    statuses = Counter(
        record["reuse_status"] for record in records
    )

    # Summarize phase transitions using the configured window.
    transitions = []

    for i in range(1, n):
        if records[i]["phase_id"] != records[i - 1]["phase_id"]:
            before = records[
                max(0, i - transition_window):i
            ]
            after = records[
                i:min(n, i + transition_window)
            ]

            transitions.append({
                "step": i,
                "phase_id": records[i]["phase_id"],
                "mean_shift_before": float(np.mean([
                    record["query_drift_features"][
                        FEATURE_NAMES[0]
                    ]
                    for record in before
                ])),
                "mean_shift_after": float(np.mean([
                    record["query_drift_features"][
                        FEATURE_NAMES[0]
                    ]
                    for record in after
                ])),
            })

    report = {
        "total_reward": float(sum(rewards)),
        "mean_reward": float(sum(rewards) / n),

        "cache_hit_rate": float(
            sum(
                record["cache_hit"]
                for record in records
            ) / n
        ),
        "cache_miss_rate": float(
            sum(
                record["cache_miss"]
                for record in records
            ) / n
        ),

        "correct_reuse_count": statuses["correct"],
        "incorrect_reuse_count": statuses["incorrect"],
        "unevaluable_reuse_count": statuses["unevaluable"],

        "reuse_count": env.reuse_count,
        "recompute_count": env.recompute_count,

        "forced_recomputation_count": sum(
            record["forced_recomputation"]
            for record in records
        ),

        "average_cache_occupancy": float(np.mean([
            record["cache_occupancy"]
            for record in records
        ])),

        "phase_transitions": transitions,
    }

    report["evaluation_metrics"] = summarize(
        records,
        end_to_end_latencies_ms,
        decision_latencies_ms,
        include_timing=include_timing,
    )

    return report


def compare(
    workload,
    encoder,
    seed=0,
    include_timing=False,
    **config,
):
    """Compare the three deterministic policies."""
    return {
        name: evaluate(
            workload,
            encoder,
            policy,
            seed=seed,
            include_timing=include_timing,
            **config,
        )
        for name, policy in POLICIES.items()
    }


def cacheeval(rows, encoder, threshold=0.85):
    """Evaluate labeled text pairs in the separate CacheEval benchmark."""
    counts = Counter()

    for row in rows:
        if row.get("binary_label") not in {"HIT", "MISS"}:
            raise ValueError(
                "CacheEval requires public binary HIT/MISS labels"
            )

        hit = (
            cosine(
                encoder.encode(row["query_a"]),
                encoder.encode(row["query_b"]),
            )
            >= threshold
        )

        expected = row["binary_label"] == "HIT"

        counts[
            "tp"
            if hit and expected
            else "fp"
            if hit
            else "fn"
            if expected
            else "tn"
        ] += 1

    negatives = counts["fp"] + counts["tn"]

    return dict(
        counts,
        false_hit_rate=(
            counts["fp"] / negatives if negatives else None
        ),
        limitation=(
            "Text-only retrieval: context, tools and system prompts "
            "are not encoded; binary labels retain those distinctions."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic semantic cache policies."
    )
    parser.add_argument(
        "--workload",
        help="Path to a workload file. Uses a generated workload if omitted.",
    )
    parser.add_argument(
        "--include-timing",
        action="store_true",
        help="Include wall-clock latency metrics in evaluation reports.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for workload generation and environment reset.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for saving the JSON evaluation report.",
    )
    parser.add_argument(
        "--cacheeval",
        action="store_true",
        help="Run the separate labeled CacheEval benchmark.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for CacheEval.",
    )

    args = parser.parse_args()

    encoder = SentenceEncoder()

    if args.cacheeval:
        if not args.workload:
            parser.error("--cacheeval requires --workload")

        rows = read(args.workload)
        report = cacheeval(
            rows,
            encoder,
            threshold=args.threshold,
        )
    else:
        workload = (
            read(args.workload)
            if args.workload
            else generate(seed=args.seed)
        )

        report = compare(
            workload,
            encoder,
            seed=args.seed,
            include_timing=args.include_timing,
        )

    output = json.dumps(report, indent=2, allow_nan=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(output + "\n")
    else:
        print(output)
