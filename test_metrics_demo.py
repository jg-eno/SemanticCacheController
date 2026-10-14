
from src.evaluation_metrics import summarize

records = [
    {"cache_hit": True, "effective_action": 0, "reuse_status": "correct"},
    {"cache_hit": True, "effective_action": 0, "reuse_status": "incorrect"},
    {"cache_hit": True, "effective_action": 0, "reuse_status": "unevaluable"},
    {"cache_hit": False, "effective_action": 1, "reuse_status": "not_reused"},
]

end_to_end_latencies = [10, 20, 30, 40]
decision_latencies = [1, 2, 3, 4]

metrics = summarize(
    records,
    end_to_end_latencies,
    decision_latencies,
    include_timing=True,
)

keys = [
    "semantic_hit_rate",
    "served_hit_rate",
    "quality_adjusted_hit_rate",
    "hit_precision",
    "false_hit_rate",
    "latency_ms_mean",
    "latency_ms_p95",
]

for key in keys:
    value = metrics[key]
    if value is not None and "rate" in key or key == "hit_precision":
        print(f"{key}: {value:.4f} ({value * 100:.2f}%)")
    else:
        print(f"{key}: {value}")

