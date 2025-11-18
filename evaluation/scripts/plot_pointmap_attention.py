#!/usr/bin/env python3
"""
Visualize how pointmap conditioning alters readout focus.

Loads a snapshot that contains:
  - {label}_octo_tokens                      -> (256, 512)
  - {label}_readout_pre_pointmap_tokens      -> (R, 512)
  - {label}_readout_post_pointmap_tokens     -> (R, 512)
  - {label}_rgb (optional)                   -> (H, W, 3)

Produces a 1x4 figure:
  [ RGB | Pre-Pointmap Similarity | Post-Pointmap Similarity | Post-Pre Difference ]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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


def _load_array(data: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    if key not in data:
        return None
    arr = np.asarray(data[key])
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return np.asfarray(arr, dtype=np.float32)


def _compute_patch_similarity(octo_tokens: np.ndarray, readout_tokens: np.ndarray) -> np.ndarray:
    if octo_tokens.ndim != 2:
        raise ValueError(f"Expected octo_tokens to be 2D, got shape {octo_tokens.shape}")
    if readout_tokens.ndim != 2:
        raise ValueError(f"Expected readout_tokens to be 2D, got shape {readout_tokens.shape}")
    if octo_tokens.shape[1] != readout_tokens.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: octo {octo_tokens.shape[1]} vs readout {readout_tokens.shape[1]}"
        )

    octo_norm = octo_tokens / (np.linalg.norm(octo_tokens, axis=1, keepdims=True) + 1e-8)
    readout_norm = readout_tokens / (np.linalg.norm(readout_tokens, axis=1, keepdims=True) + 1e-8)

    sims = octo_norm @ readout_norm.T  # (num_patches, num_readout)
    patch_scores = sims.mean(axis=1)

    side = int(round(np.sqrt(patch_scores.size)))
    if side * side != patch_scores.size:
        raise ValueError(f"Token count {patch_scores.size} is not a perfect square (expected 256, etc)")
    return patch_scores.reshape(side, side)


def _normalize_heatmap(h: np.ndarray) -> np.ndarray:
    min_val = float(np.min(h))
    max_val = float(np.max(h))
    if max_val - min_val < 1e-8:
        return np.zeros_like(h)
    return (h - min_val) / (max_val - min_val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize pointmap-conditioned readout attention.")
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to snapshot (.npz) file.")
    parser.add_argument("--label", type=str, default="pointmap", help="Policy label stored in the snapshot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/attention_snapshots/pointmap_readout_comparison.png"),
        help="Where to save the resulting figure.",
    )
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay transparency.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    args = parser.parse_args()

    with np.load(args.snapshot, allow_pickle=True) as data:
        label = args.label
        octo = _load_array(data, f"{label}_octo_tokens")
        pre = _load_array(data, f"{label}_readout_pre_pointmap_tokens")
        post = _load_array(data, f"{label}_readout_post_pointmap_tokens")
        rgb = _load_array(data, f"{label}_rgb")
        img_pre = _load_array(data, f"{label}_image_tokens_pre_pointmap")
        img_post = _load_array(data, f"{label}_image_tokens_post_pointmap")

    if pre is None or post is None:
        raise KeyError(
            "Snapshot is missing readout tensors required for visualization. "
            f"pre={pre is not None}, post={post is not None}."
        )

    patch_tokens_pre = img_pre if img_pre is not None else (img_post if img_post is not None else octo)
    patch_tokens_post = img_post if img_post is not None else (img_pre if img_pre is not None else octo)

    if patch_tokens_pre is None or patch_tokens_post is None:
        raise KeyError(
            "Snapshot is missing both transformer-level image tokens and fallback octo tokens."
        )

    pre_map = _compute_patch_similarity(patch_tokens_pre, pre)
    post_map = _compute_patch_similarity(patch_tokens_post, post)

    pre_overlay = _normalize_heatmap(pre_map)
    post_overlay = _normalize_heatmap(post_map)
    heatmap_cmap = "turbo"
    panels = 3
    fig, axes = plt.subplots(1, panels, figsize=(4.5 * panels, 5))

    if rgb is not None:
        axes[0].imshow(rgb.astype(np.uint8))
        axes[0].set_title("Policy RGB Input")
    else:
        axes[0].imshow(pre_overlay, cmap="gray")
        axes[0].set_title("Token Grid (No RGB)")
    axes[0].axis("off")

    if rgb is not None:
        pre_img = _resize_heatmap(pre_overlay, rgb.shape[:2])
        post_img = _resize_heatmap(post_overlay, rgb.shape[:2])
    else:
        pre_img = pre_overlay
        post_img = post_overlay

    base_img = rgb.astype(np.uint8) if rgb is not None else pre_overlay

    axes[1].imshow(base_img, alpha=1.0 if rgb is None else 1.0)
    if rgb is not None:
        axes[1].imshow(pre_img, cmap=heatmap_cmap, alpha=args.alpha)
    axes[1].axis("off")
    axes[1].set_title("Readout Attention (Before 3D Fusion)")

    axes[2].imshow(base_img, alpha=1.0 if rgb is None else 1.0)
    if rgb is not None:
        axes[2].imshow(post_img, cmap=heatmap_cmap, alpha=args.alpha)
    axes[2].axis("off")
    axes[2].set_title("Readout Attention (After 3D Fusion)")

    cmap_sm = plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=Normalize(0.0, 1.0))
    cmap_sm.set_array([])
    fig.colorbar(cmap_sm, ax=axes[1], fraction=0.046, pad=0.04, label="Similarity (norm)")
    fig.colorbar(cmap_sm, ax=axes[2], fraction=0.046, pad=0.04, label="Similarity (norm)")

    fig.suptitle("Readout Attention Shift from Pointmap Injection", fontsize=18, y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600)
    print(f"[plot_pointmap_attention] Saved visualization to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
