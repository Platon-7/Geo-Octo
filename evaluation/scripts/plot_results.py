#!/usr/bin/env python3
"""
Produce the evaluation bar chart comparing success rates across models.

Generates two subplots (LIBERO Object and LIBERO Spatial) with custom colors:
  - Baseline: warm orange
  - VGGT-Only: vibrant red
  - VGGT-Fusion: deep blue/teal
  - VGGT-Pointmap: rich yellow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation success rates.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/attention_snapshots/evaluation_bar_chart.png"),
        help="Path to save the bar chart.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    args = parser.parse_args()

    data_object = {
        "Baseline": 70.8,
        "VGGT-Only": 7.2,
        "VGGT-Fusion": 61.0,
        "VGGT-Pointmap": 9.0,
    }

    data_spatial = {
        "Baseline": 82.0,
        "VGGT-Only": 15.2,
        "VGGT-Fusion": 74.8,
        "VGGT-Pointmap": 41.4,
    }

    color_map = {
        "Baseline": "#FF8C42",
        "VGGT-Only": "#E63946",
        "VGGT-Fusion": "#127FAF",
        "VGGT-Pointmap": "#FFD166",
    }

    datasets = [
        ("LIBERO Object", data_object),
        ("LIBERO Spatial", data_spatial),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    axes = axes[0]

    for ax, (title, data) in zip(axes, datasets):
        labels = list(data.keys())
        values = np.array(list(data.values()))
        colors = [color_map[label] for label in labels]

        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.0)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        ax.set_ylim(0, 100)
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(title, fontsize=14, pad=12)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Evaluation Success Rates on LIBERO Suites", fontsize=18, y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600)
    print(f"[plot_evaluation_bar_chart] Saved bar chart to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
