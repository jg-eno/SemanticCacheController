"""Plot every benchmark report named results*.json under benchmark/."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BENCHMARK_DIR / "graphs"
METRICS = (
    ("false_hit_rate", "False-hit rate", "rate"),
    ("recomputation_rate", "Recomputation rate", "rate"),
    ("mean_environment_reward", "Mean environment reward", "reward"),
)


def load_report(path):
    """Load a benchmark report, rejecting JSON files that are not result reports."""
    report = json.loads(path.read_text(encoding="utf-8"))
    results = report.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("missing nonempty 'results' object")

    valid_results = {}
    for method, values in results.items():
        if not isinstance(values, dict):
            continue
        metrics = {}
        for key, _, _ in METRICS:
            value = values.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                metrics[key] = value
        if metrics:
            valid_results[method] = metrics

    if not valid_results:
        raise ValueError("no supported metrics found in results")
    return report, valid_results


def report_settings(report):
    """Return a compact title suffix from the environment settings."""
    settings = report.get("settings")
    if not isinstance(settings, dict):
        method_settings = report.get("method_env_settings", {})
        settings = next(iter(method_settings.values()), {}) if isinstance(method_settings, dict) else {}

    parts = []
    workload = settings.get("workload_path")
    if workload:
        parts.append(f"workload: {Path(workload).name}")
    for key in ("capacity", "threshold", "window"):
        if key in settings:
            parts.append(f"{key}: {settings[key]}")
    if "seed" in report:
        parts.append(f"seed: {report['seed']}")
    return " | ".join(parts)


def plot_report(source, report, results, output_path):
    methods = list(results)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 5.5))
    title = source.name
    settings = report_settings(report)
    if settings:
        title = f"{title}\n{settings}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.get_cmap("tab10")
    for axis, (metric, label, kind) in zip(axes, METRICS):
        available = [(method, results[method][metric]) for method in methods
                     if metric in results[method]]
        if not available:
            axis.set_visible(False)
            continue

        names, values = zip(*available)
        plotted = [value * 100 if kind == "rate" else value for value in values]
        bars = axis.bar(names, plotted, color=[colors(i % 10) for i in range(len(names))], width=0.72)
        axis.set_title(label)
        axis.set_ylabel("Percent" if kind == "rate" else "Reward")
        axis.tick_params(axis="x", rotation=30, labelsize=9)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)

        if kind == "rate":
            axis.set_ylim(0, 112)
            axis.set_yticks(range(0, 101, 20), labels=[f"{tick}%" for tick in range(0, 101, 20)])
        else:
            low = min(0.0, min(plotted))
            high = max(0.0, max(plotted))
            padding = max((high - low) * 0.12, 0.05)
            axis.set_ylim(low - padding, high + padding)
            if low < 0 < high:
                axis.axhline(0, color="#596863", linewidth=0.8)

        for bar, value in zip(bars, values):
            label_text = f"{value * 100:.1f}%" if kind == "rate" else f"{value:.3f}"
            height = bar.get_height()
            axis.annotate(
                label_text,
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4 if height >= 0 else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )

    fig.text(
        0.5, 0.015,
        "Offline simulation; intent labels are a proxy for answer correctness.",
        ha="center", fontsize=9, color="#596863",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def result_files():
    """Find benchmark outputs without picking up settings or methods JSON files."""
    return sorted(BENCHMARK_DIR.rglob("results*.json"))


def main():
    files = result_files()
    if not files:
        raise SystemExit(f"No results*.json files found under {BENCHMARK_DIR}")

    generated = 0
    for source in files:
        try:
            report, results = load_report(source)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"Skipping {source.relative_to(BENCHMARK_DIR)}: {error}")
            continue

        relative_stem = source.relative_to(BENCHMARK_DIR).with_suffix("").as_posix()
        filename = relative_stem.replace("/", "__") + "_comparison.png"
        output_path = OUTPUT_DIR / filename
        plot_report(source, report, results, output_path)
        print(f"Saved: {output_path.relative_to(BENCHMARK_DIR)}")
        generated += 1

    if not generated:
        raise SystemExit("No valid benchmark result reports were found.")
    print(f"Generated {generated} graph(s) from {len(files)} result file(s).")


if __name__ == "__main__":
    main()