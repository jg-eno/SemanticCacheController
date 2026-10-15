"""Shared evaluation metrics for semantic-cache experiments."""

import math
import statistics


def similarity_threshold_rule(observation, similarity_threshold=0.90):
    """Reuse only when a cache candidate meets a policy similarity threshold.

    The environment encodes a candidate's cosine similarity as novelty
    ``(1 - similarity) / 2``. Novelty is exactly 1 when no candidate exists.
    The policy threshold is independent of the cache's retrieval threshold.
    """
    if not math.isfinite(similarity_threshold) or not -1 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (-1, 1]")
    novelty = float(observation[1])
    similarity = 1.0 - 2.0 * novelty
    return 0 if novelty < 1.0 and similarity >= similarity_threshold else 1


def reuse_metrics(records):
    """Summarize status-bearing records with explicit quality denominators."""
    n = len(records)
    if not n:
        raise ValueError("Cannot summarize empty reuse records")

    correct = sum(record["status"] == "correct" for record in records)
    incorrect = sum(record["status"] == "incorrect" for record in records)
    served = sum(record["status"] != "not_reused" for record in records)
    evaluable_served = correct + incorrect
    return {
        "served_hit_rate": served / n,
        "quality_adjusted_hit_rate": correct / n,
        "reuse_precision": (
            correct / evaluable_served if evaluable_served else None
        ),
        "incorrect_share_of_evaluable_reuse": (
            incorrect / evaluable_served if evaluable_served else None
        ),
        "unevaluable_share_of_served_hits": (
            (served - evaluable_served) / served if served else None
        ),
    }


def similarity_threshold_rule(observation, similarity_threshold=0.90):
    """Reuse only when the candidate cosine similarity meets a threshold.

    The environment encodes candidate similarity as novelty ``(1 -
    similarity) / 2`` and uses novelty 1 when no candidate is available.
    """
    if (
        not math.isfinite(similarity_threshold)
        or not -1 < similarity_threshold <= 1
    ):
        raise ValueError("similarity_threshold must be in (-1, 1]")

    novelty = float(observation[1])
    similarity = 1.0 - 2.0 * novelty
    return 0 if novelty < 1.0 and similarity >= similarity_threshold else 1


def percentile(values, percent):
    """Calculate a nearest-rank percentile."""
    if not values:
        return None

    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percent / 100 * len(ordered)) - 1)
    return ordered[index]


def summarize(
    records,
    end_to_end_latencies_ms=None,
    decision_latencies_ms=None,
    include_timing=False,
):
    """Summarize deterministic evaluation records, optionally including timing."""
    n = len(records)
    if not n:
        return {"query_count": 0}

    end_to_end_latencies_ms = end_to_end_latencies_ms or []
    decision_latencies_ms = decision_latencies_ms or []
    served_hits = [
        record for record in records
        if record.get("effective_action") == 0
    ]
    correct = [
        record for record in served_hits
        if record.get("reuse_status") == "correct"
    ]
    incorrect = [
        record for record in served_hits
        if record.get("reuse_status") == "incorrect"
    ]
    unevaluable = [
        record for record in served_hits
        if record.get("reuse_status") == "unevaluable"
    ]

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    metrics = {
        "query_count": n,
        "semantic_hit_rate": sum(
            bool(record.get("cache_hit")) for record in records
        ) / n,
        "served_hit_rate": len(served_hits) / n,
        "quality_adjusted_hit_rate": len(correct) / n,
        "hit_precision": ratio(len(correct), len(served_hits)),
        "false_hit_rate": ratio(len(incorrect), len(served_hits)),
        "unevaluable_hit_rate": ratio(len(unevaluable), len(served_hits)),
        # These require instrumentation not available in this simulation.
        "stale_hit_rate": None,
        "cost_per_query": None,
        "no_cache_cost_per_query": None,
        "relative_cost": None,
        "cost_savings": None,
        "controller_memory_bytes": None,
        "net_benefit_after_controller": None,
        "limitations": [
            "Latency measures local policy and environment execution time.",
            "Intent-label correctness is not independent response validation.",
            "Staleness requires freshness ground truth.",
            "Cost and net benefit require an explicit cost model.",
            "Controller memory requires dedicated memory measurement.",
        ],
    }
    if include_timing:
        total_latency = list(end_to_end_latencies_ms)
        decision_latency = list(decision_latencies_ms)
        metrics.update({
            "latency_ms_mean": (
                statistics.mean(total_latency) if total_latency else None
            ),
            "latency_ms_p50": percentile(total_latency, 50),
            "latency_ms_p95": percentile(total_latency, 95),
            "latency_ms_p99": percentile(total_latency, 99),
            "decision_latency_ms_mean": (
                statistics.mean(decision_latency) if decision_latency else None
            ),
            "decision_latency_ms_p50": percentile(decision_latency, 50),
            "decision_latency_ms_p95": percentile(decision_latency, 95),
            "decision_latency_ms_p99": percentile(decision_latency, 99),
        })

    return metrics
