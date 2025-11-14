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
from typing import Any, Dict, Optional, Tuple

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


def _load_policy_snapshot(npz_path: Path, label: str) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        rgb_key = f"{label}_rgb"
        octo_key = f"{label}_octo_tokens"
        vggt_key = f"{label}_vggt_tokens"
        meta_key = f"{label}_meta"

        payload = {
            "rgb": np.asarray(data[rgb_key]) if rgb_key in data else None,
            "octo_tokens": (
                np.asfarray(data[octo_key], dtype=np.float32) if octo_key in data else None
            ),
            "vggt_tokens": (
                np.asfarray(data[vggt_key], dtype=np.float32) if vggt_key in data else None
            ),
        }

        if payload["octo_tokens"] is None and payload["vggt_tokens"] is None:
            raise KeyError(
                f"Snapshot {npz_path} missing vision tokens for policy '{label}'. "
                "Expected either octo or VGGT tokens."
            )
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


def _render_placeholder(ax, title: str, message: str) -> None:
    ax.imshow(np.ones((2, 2, 3), dtype=np.float32), alpha=0.0)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, transform=ax.transAxes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize attention snapshots from evaluation.")
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to the combined .npz snapshot file.")
    parser.add_argument(
        "--policies",
        type=str,
        default="baseline,vggt,vggt_only",
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

    snapshots_map: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        payload = _load_policy_snapshot(args.snapshot, label)
        snapshots_map[label.lower()] = payload

    baseline_payload = snapshots_map.get("baseline")
    vggt_payload = snapshots_map.get("vggt")
    vggt_only_payload = snapshots_map.get("vggt_only") or snapshots_map.get("vggt-only")

    panels = []

    panels.append(
        {
            "title": "Reference RGB",
            "rgb": (baseline_payload or vggt_payload or vggt_only_payload or {}).get("rgb"),
        }
    )

    panels.append(
        {
            "title": "Baseline – Self-Similarity (Octo)",
            "rgb": baseline_payload.get("rgb") if baseline_payload else None,
            "octo": baseline_payload.get("octo_tokens") if baseline_payload else None,
            "mode": "self",
        }
    )

    panels.append(
        {
            "title": "VGGT Fusion – Cross-Modal Similarity",
            "rgb": vggt_payload.get("rgb") if vggt_payload else None,
            "octo": vggt_payload.get("octo_tokens") if vggt_payload else None,
            "vggt": vggt_payload.get("vggt_tokens") if vggt_payload else None,
            "mode": "cross",
        }
    )

    panels.append(
        {
            "title": "VGGT-Only – Self-Similarity (VGGT)",
            "rgb": (vggt_only_payload or baseline_payload or vggt_payload or {}).get("rgb"),
            "vggt": vggt_only_payload.get("vggt_tokens") if vggt_only_payload else None,
            "mode": "vggt_self",
        }
    )

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = np.array([axes])

    for ax, panel in zip(axes, panels):
        title = panel["title"]
        mode = panel.get("mode")
        rgb = panel.get("rgb")

        if mode is None:
            if rgb is not None:
                _render_rgb(ax, rgb, title)
            else:
                _render_placeholder(ax, title, "RGB frame unavailable.")
            continue

        if mode == "self":
            tokens = panel.get("octo")
            if tokens is not None and rgb is not None:
                heat_low = _cosine_similarity_map(tokens, tokens)
                heat_overlay = _resize_heatmap(_normalize_heatmap(heat_low), rgb.shape[:2])
                overlay = _render_heatmap(ax, rgb, heat_overlay, title, args.alpha)
                fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
            else:
                _render_placeholder(ax, title, "Octo tokens unavailable.")
        elif mode == "cross":
            octo_tokens = panel.get("octo")
            vggt_tokens = panel.get("vggt")
            if octo_tokens is not None and vggt_tokens is not None and rgb is not None:
                heat_low = _cosine_similarity_map(octo_tokens, vggt_tokens)
                heat_overlay = _resize_heatmap(_normalize_heatmap(heat_low), rgb.shape[:2])
                overlay = _render_heatmap(ax, rgb, heat_overlay, title, args.alpha)
                fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
            else:
                _render_placeholder(ax, title, "Required tokens unavailable.")
        elif mode == "vggt_self":
            vggt_tokens = panel.get("vggt")
            if vggt_tokens is not None and rgb is not None:
                heat_low = _cosine_similarity_map(vggt_tokens, vggt_tokens)
                heat_overlay = _resize_heatmap(_normalize_heatmap(heat_low), rgb.shape[:2])
                overlay = _render_heatmap(ax, rgb, heat_overlay, title, args.alpha)
                fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
            else:
                _render_placeholder(ax, title, "VGGT tokens unavailable.")
        else:
            _render_placeholder(ax, title, "Unsupported panel mode.")

    fig.suptitle(
        "Attention Snapshot Comparison: Baseline vs. VGGT Variants",
        fontsize=18,
        y=0.95,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600)
    print(f"[plot_attention] Saved visualization to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
