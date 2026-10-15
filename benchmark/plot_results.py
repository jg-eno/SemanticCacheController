
import matplotlib.pyplot as plt
from pathlib import Path

# Hardcoded benchmark results

eviction_results = {
    "FIFO": {
        "false_hit_rate": 0.0,
        "recomputation_rate": 0.0416667,
        "mean_environment_reward": 0.95
    },
    "LIFO": {
        "false_hit_rate": 0.0,
        "recomputation_rate": 0.0416667,
        "mean_environment_reward": 0.95
    },
    "LRU": {
        "false_hit_rate": 0.0,
        "recomputation_rate": 0.0416667,
        "mean_environment_reward": 0.95
},
    "LFU": {
        "false_hit_rate": 0.0,
        "recomputation_rate": 0.0416667,
        "mean_environment_reward": 0.95
    }
}

held_out_results = {
    "always_reuse": {
        "false_hit_rate": 0.0,
        "recomputation_rate": 0.0416667,
        "mean_environment_reward": 0.95
    },
    "always_recompute": {
        "false_hit_rate": 0.0,
        "recomputation_rate": 1.0,
        "mean_environment_reward": -0.2
    },
    "drift_rule": {
        "false_hit_rate": 0.0,
        "recomputation_rate": 0.28125,
        "mean_environment_reward": 0.6625
    }
}

# Graph settings
metrics = [
    ("false_hit_rate", "False Hit Rate"),
    ("recomputation_rate", "Recomputation Rate"),
    ("mean_environment_reward", "Mean Environment Reward")
]

output_dir = Path("benchmark/graphs")
output_dir.mkdir(parents=True, exist_ok=True)


def plot_results(results, title, filename):
    methods = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=15)

    for ax, (metric, label) in zip(axes, metrics):
        values = [results[method][metric] for method in methods]

        bars = ax.bar(methods, values)

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

    fig.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filepath}")


# Generate both graphs
plot_results(
    eviction_results,
    "Eviction Policy Comparison",
    "eviction_policy_comparison.png"
)

plot_results(
    held_out_results,
    "Held-out Baseline Comparison",
    "baseline_comparison.png"
)

print("All graphs generated successfully!")
