
"""Evaluation metrics for semantic cache experiments."""

import math
import statistics


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
    """Summarize evaluation metrics, optionally including timings."""
    n = len(records)

    if n == 0:
        return {"query_count": 0}

    end_to_end_latencies_ms = end_to_end_latencies_ms or []
    decision_latencies_ms = decision_latencies_ms or []

    served_hits = [
        r for r in records
        if r.get("effective_action") == 0
    ]

    correct = [
        r for r in served_hits
        if r.get("reuse_status") == "correct"
    ]

    incorrect = [
        r for r in served_hits
        if r.get("reuse_status") == "incorrect"
    ]

    unevaluable = [
        r for r in served_hits
        if r.get("reuse_status") == "unevaluable"
    ]

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    metrics = {
        "query_count": n,
        "semantic_hit_rate": sum(
            bool(r.get("cache_hit")) for r in records
        ) / n,
        "served_hit_rate": len(served_hits) / n,
        "quality_adjusted_hit_rate": len(correct) / n,
        "hit_precision": ratio(len(correct), len(served_hits)),
        "false_hit_rate": ratio(len(incorrect), len(served_hits)),
        "unevaluable_hit_rate": ratio(
            len(unevaluable), len(served_hits)
        ),

        # Additional instrumentation is required for these metrics.
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
                statistics.mean(total_latency)
                if total_latency else None
            ),
            "latency_ms_p50": percentile(total_latency, 50),
            "latency_ms_p95": percentile(total_latency, 95),
            "latency_ms_p99": percentile(total_latency, 99),

            "decision_latency_ms_mean": (
                statistics.mean(decision_latency)
                if decision_latency else None
            ),
            "decision_latency_ms_p50": percentile(
                decision_latency, 50
            ),
            "decision_latency_ms_p95": percentile(
                decision_latency, 95
            ),
            "decision_latency_ms_p99": percentile(
                decision_latency, 99
            ),
        })

    return metrics
