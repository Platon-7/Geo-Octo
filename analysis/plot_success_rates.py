"""
Generate Plot 1 (evaluation bar chart) from JSON or inline dictionaries.

Expected JSON format:
{
  "libero_object": {
    "Octo (baseline)": 0.42,
    "VGGT concat": 0.37,
    "VGGT replace": 0.28,
    "Pointmap": 0.33
  },
  "libero_spatial": {
    "Octo (baseline)": 0.35,
    "VGGT concat": 0.31,
    "VGGT replace": 0.21,
    "Pointmap": 0.25
  }
}

Usage:
    python analysis/plot_success_rates.py \
        --results results.json \
        --output plot_success_rates.png
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str | None) -> Dict[str, Dict[str, float]]:
    if path is None:
        raise ValueError("Please provide --results or edit the inline fallback dictionary.")
    with open(path, "r") as f:
        return json.load(f)


def plot(results: Dict[str, Dict[str, float]], output: str | None = None) -> None:
    suites = list(results.keys())
    model_names: List[str] = sorted({model for suite in suites for model in results[suite].keys()})

    x = np.arange(len(model_names))
    width = 0.35 if len(suites) == 2 else 0.8 / max(1, len(suites))

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, suite in enumerate(suites):
        offsets = x + (idx - (len(suites) - 1) / 2) * width
        values = [results[suite].get(model, 0.0) for model in model_names]
        bars = ax.bar(offsets, values, width, label=suite.replace("_", " ").title())
        ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in values], padding=3, fontsize=9)

    ax.set_ylabel("Success rate")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0, max(0.05, max(results[suite][model] for suite in suites for model in results[suite])) * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_title("LIBERO evaluation success rates")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved bar chart to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot grouped success-rate bars for LIBERO suites.")
    parser.add_argument("--results", type=str, help="Path to JSON file with suite->model->success_rate mapping.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for the figure.")
    args = parser.parse_args()

    results = load_results(args.results)
    plot(results, output=args.output)


if __name__ == "__main__":
    main()
