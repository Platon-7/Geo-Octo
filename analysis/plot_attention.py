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
            "octo_tokens": np.asarray(data[octo_key]),
            "vggt_tokens": np.asarray(data[vggt_key]) if vggt_key in data else None,
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


def _render_policy_axis(
    ax,
    rgb: np.ndarray,
    heatmap: np.ndarray,
    label: str,
    alpha: float,
    meta: Dict,
):
    ax.imshow(rgb)
    overlay = ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")

    title = label
    if meta:
        extras = []
        seconds = meta.get("seconds_elapsed")
        if seconds is not None:
            extras.append(f"t={seconds:.2f}s")
        freq = meta.get("control_freq")
        if freq is not None:
            extras.append(f"{freq:.1f}Hz")
        if extras:
            title += f" ({', '.join(extras)})"
    ax.set_title(title)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize attention snapshots from evaluation.")
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to the .npz snapshot file.")
    parser.add_argument(
        "--policies",
        type=str,
        default="baseline,vggt",
        help="Comma-separated list of policy labels stored in the snapshot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/attention_snapshots/attention_overlay.png"),
        help="Where to save the resulting visualization.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Transparency for heatmap overlay.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    args = parser.parse_args()

    labels = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not labels:
        raise ValueError("No policy labels provided.")

    snapshots = [_load_policy_snapshot(args.snapshot, label) for label in labels]

    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 6))
    axes = np.atleast_1d(axes)

    for ax, label, payload in zip(axes, labels, snapshots):
        rgb = payload["rgb"]
        octo_tokens = np.asarray(payload["octo_tokens"])
        vggt_tokens = payload["vggt_tokens"]
        if vggt_tokens is None:
            vggt_tokens = octo_tokens.copy()
        else:
            vggt_tokens = np.asarray(vggt_tokens)

        heat_low = _cosine_similarity_map(octo_tokens, vggt_tokens)
        heat_overlay = _resize_heatmap(_normalize_heatmap(heat_low), rgb.shape[:2])
        overlay = _render_policy_axis(ax, rgb, heat_overlay, label, alpha=args.alpha, meta=payload.get("meta", {}))
        fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"[plot_attention] Saved visualization to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
