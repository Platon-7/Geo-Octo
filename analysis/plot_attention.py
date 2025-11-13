#!/usr/bin/env python3
"""
Visualize attention snapshots by overlaying cosine-similarity heatmaps on the captured RGB frame.

Expected snapshot structure (.npz):
  {label}_rgb           -> (H, W, 3) uint8 image
  {label}_octo_tokens   -> (256, 512) float32 tokens
  {label}_vggt_tokens   -> (256, 512) float32 tokens (optional)
  {label}_meta          -> JSON-encoded metadata (optional)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2

    def _resize_heatmap(map_2d: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(map_2d, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)

except Exception:
    from scipy.ndimage import zoom  # type: ignore

    def _resize_heatmap(map_2d: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        zoom_y = target_hw[0] / map_2d.shape[0]
        zoom_x = target_hw[1] / map_2d.shape[1]
        return zoom(map_2d, (zoom_y, zoom_x), order=3)


def _load_policy_snapshot(npz_path: Path, label: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        rgb_key = f"{label}_rgb"
        octo_key = f"{label}_octo_tokens"
        vggt_key = f"{label}_vggt_tokens"
        meta_key = f"{label}_meta"

        missing = [k for k in (rgb_key, octo_key) if k not in data]
        if missing:
            raise KeyError(f"Snapshot {npz_path} missing required keys for policy '{label}': {missing}")

        payload = {
            "rgb": np.asarray(data[rgb_key]),
            "octo_tokens": np.asfarray(data[octo_key], dtype=np.float32),
            "vggt_tokens": (
                np.asfarray(data[vggt_key], dtype=np.float32) if vggt_key in data else None
            ),
        }
        if meta_key in data:
            try:
                payload["meta"] = json.loads(str(data[meta_key].item()))
            except Exception:
                payload["meta"] = {}
        else:
            payload["meta"] = {}
        return payload


def _cosine_similarity_map(octo: np.ndarray, vggt: np.ndarray) -> np.ndarray:
    if octo.shape != vggt.shape:
        raise ValueError(f"Token shape mismatch: octo {octo.shape} vs vggt {vggt.shape}")
    if octo.ndim != 2:
        raise ValueError(f"Expected (N, D) tokens but received shape {octo.shape}")

    octo_norm = octo / (np.linalg.norm(octo, axis=1, keepdims=True) + 1e-8)
    vggt_norm = vggt / (np.linalg.norm(vggt, axis=1, keepdims=True) + 1e-8)
    similarity = np.sum(octo_norm * vggt_norm, axis=1)

    side = int(round(np.sqrt(similarity.size)))
    if side * side != similarity.size:
        raise ValueError(
            f"Expected token count to be a perfect square (e.g., 256). Received {similarity.size}."
        )
    return similarity.reshape(side, side)


def _normalize_heatmap(h: np.ndarray) -> np.ndarray:
    min_val = float(np.min(h))
    max_val = float(np.max(h))
    if max_val - min_val < 1e-8:
        return np.zeros_like(h)
    return (h - min_val) / (max_val - min_val)


def _render_rgb(ax, rgb: np.ndarray, title: str) -> None:
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(title)


def _render_heatmap(
    ax,
    base_image: np.ndarray,
    heatmap: np.ndarray,
    title: str,
    alpha: float,
):
    ax.imshow(base_image)
    overlay = ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")
    ax.set_title(title)
    return overlay


def main() -> None:
    # Bar chart summary of evaluation results
    libero_object = {
        "Baseline": 70.8,
        "VGGT-Only": 7.2,
        "VGGT-Fusion": 61.0,
        "VGGT-Pointmap": 9.0,
    }
    libero_spatial = {
        "Baseline": 82.0,
        "VGGT-Only": 15.2,
        "VGGT-Fusion": 74.8,
        "VGGT-Pointmap": 41.4,
    }

    color_map = {
        "Baseline": "#ff8c42",       # warm orange
        "VGGT-Only": "#e63946",      # vibrant red
        "VGGT-Fusion": "#0081a7",    # deep teal/blue
        "VGGT-Pointmap": "#ffca3a",  # rich yellow
    }

    datasets = [("LIBERO Object", libero_object), ("LIBERO Spatial", libero_spatial)]

    parser = argparse.ArgumentParser(description="Plot evaluation success rates.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/attention_snapshots/evaluation_bar_chart.png"),
        help="Where to save the bar chart.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    axes = axes[0]

    for ax, (title, data) in zip(axes, datasets):
        names = list(data.keys())
        values = list(data.values())
        colors = [color_map[name] for name in names]

        bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=1.0)

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
        ax.set_xticklabels(names, rotation=15, ha="right")

    fig.suptitle("Evaluation Success Rates on LIBERO Suites", fontsize=18, y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600)
    print(f"[plot_attention] Saved bar chart to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
