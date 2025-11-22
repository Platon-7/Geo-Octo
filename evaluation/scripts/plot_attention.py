#!/usr/bin/env python3
"""
Visualize attention snapshots by overlaying cosine-similarity heatmaps on the captured RGB frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        candidates = [label]
        if "_" in label:
            candidates.append(label.replace("_", "-"))
        if "-" in label:
            candidates.append(label.replace("-", "_"))
        candidates = list(dict.fromkeys(candidates))

        prefix = None
        for candidate in candidates:
            cand_prefix = f"{candidate}_"
            if any(k.startswith(cand_prefix) for k in data.files):
                prefix = candidate
                break

        if prefix is None:
            raise KeyError(
                f"Snapshot {npz_path} missing entries for policy '{label}'. "
                f"Available keys: {list(data.files)}"
            )

        rgb_key = f"{prefix}_rgb"
        octo_key = f"{prefix}_octo_tokens"
        vggt_key = f"{prefix}_vggt_tokens"
        meta_key = f"{prefix}_meta"

        payload = {
            "rgb": np.asarray(data[rgb_key]) if rgb_key in data else None,
            "octo_tokens": (
                np.asfarray(data[octo_key], dtype=np.float32) if octo_key in data else None
            ),
            "vggt_tokens": (
                np.asfarray(data[vggt_key], dtype=np.float32) if vggt_key in data else None
            ),
            "prefix_key": prefix,
        }

        if payload["octo_tokens"] is None and payload["vggt_tokens"] is None:
            raise KeyError(
                f"Snapshot {npz_path} missing vision tokens for policy '{label}'. "
                "Expected either octo or VGGT tokens."
            )
        attention_entries: Dict[str, np.ndarray] = {}
        attn_prefix = f"{prefix}_attn"
        for key in data.files:
            if key.startswith(attn_prefix):
                suffix = key[len(prefix) + 1 :]
                attention_entries[suffix] = np.asfarray(data[key], dtype=np.float32)
        if attention_entries:
            payload["attention_data"] = attention_entries
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


def _select_attention_overlay(entries: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    priorities = [
        "attn_readout_action_obs_primary",
        "attn_readout_action_obs_image_primary",
        "attn_readout_action_obs_obs_image_primary",
    ]
    for key in priorities:
        if key in entries:
            return entries[key]
    for key, value in entries.items():
        if not key.endswith("_layers"):
            return value
    return None


def _extract_attention_map(payload: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not payload:
        return None
    attn_entries = payload.get("attention_data")
    if not attn_entries:
        return None
    attn_map = _select_attention_overlay(attn_entries)
    if attn_map is None:
        return None
    return np.asarray(attn_map, dtype=np.float32)


def _render_attention_panel(
    ax, rgb: Optional[np.ndarray], attn_map: Optional[np.ndarray], title: str, alpha: float
):
    if attn_map is None:
        _render_placeholder(ax, title, "Attention map unavailable.")
        return None
    heat = _normalize_heatmap(attn_map)
    if rgb is not None:
        heat_overlay = _resize_heatmap(heat, rgb.shape[:2])
        return _render_heatmap(ax, rgb, heat_overlay, title, alpha)
    im = ax.imshow(heat, cmap="turbo")
    ax.axis("off")
    ax.set_title(title)
    return im


def _add_aligned_colorbar(fig, ax, im, label=None):
    """
    Adds a colorbar that perfectly matches the height of the image.
    If im is None, it adds an invisible dummy axis of the same size
    to ensure the main plot scales identically to plots with colorbars.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    if im is not None:
        fig.colorbar(im, cax=cax, label=label)
    else:
        # Create invisible axes to enforce same layout shrinkage
        cax.axis("off")


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

    fig = plt.figure(figsize=(24, 5))
    
    # LAYOUT:
    # 4 Main Columns.
    # Ratios: [1, 1, 1.6, 1]
    # Explanation: The middle column (1.6) holds TWO plots. 
    # This means each plot inside gets roughly 0.8 width, making them SMALLER than the outer plots (1.0).
    # Wspace 0.3 separates the methods clearly.
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 2, 1], wspace=0.3)

    # Column 0: RGB
    ax_rgb = fig.add_subplot(gs[0])
    
    # Column 1: Baseline
    ax_baseline_attn = fig.add_subplot(gs[1])
    
    # Column 2: VGGT Fusion Pair (Nested)
    # Wspace 0.15 makes them close, but gives just enough room for the left colorbar text.
    gs_fusion = gs[2].subgridspec(1, 2, wspace=0.25)
    ax_vggt_attn = fig.add_subplot(gs_fusion[0])
    ax_vggt_sim = fig.add_subplot(gs_fusion[1])
    
    # Column 3: VGGT Only
    ax_vggt_only_attn = fig.add_subplot(gs[3])

    # --- Plotting ---

    # 1. RGB Input (with invisible colorbar for scaling)
    rgb_source = (baseline_payload or vggt_payload or vggt_only_payload or {})
    rgb_img = rgb_source.get("rgb")
    if rgb_img is not None:
        _render_rgb(ax_rgb, rgb_img, "Policy RGB Input")
    else:
        _render_placeholder(ax_rgb, "Policy RGB Input", "RGB frame unavailable.")
    _add_aligned_colorbar(fig, ax_rgb, None, None)

    def _render_similarity_panel(ax, payload, mode, title):
        if not payload:
            _render_placeholder(ax, title, "Snapshot unavailable.")
            return None
        rgb = payload.get("rgb")
        if mode == "self":
            tokens = payload.get("octo_tokens")
            if tokens is not None and rgb is not None:
                heat_low = _cosine_similarity_map(tokens, tokens)
                heat_overlay = _resize_heatmap(_normalize_heatmap(heat_low), rgb.shape[:2])
                return _render_heatmap(ax, rgb, heat_overlay, title, args.alpha)
            else:
                _render_placeholder(ax, title, "Octo tokens unavailable.")
        elif mode == "cross":
            octo_tokens = payload.get("octo_tokens")
            vggt_tokens = payload.get("vggt_tokens")
            if octo_tokens is not None and vggt_tokens is not None and rgb is not None:
                heat_low = _cosine_similarity_map(octo_tokens, vggt_tokens)
                heat_overlay = _resize_heatmap(_normalize_heatmap(heat_low), rgb.shape[:2])
                return _render_heatmap(ax, rgb, heat_overlay, title, args.alpha)
            else:
                _render_placeholder(ax, title, "Required tokens unavailable.")
        elif mode == "vggt_self":
            vggt_tokens = payload.get("vggt_tokens")
            if vggt_tokens is not None and rgb is not None:
                heat_low = _cosine_similarity_map(vggt_tokens, vggt_tokens)
                heat_overlay = _resize_heatmap(_normalize_heatmap(heat_low), rgb.shape[:2])
                return _render_heatmap(ax, rgb, heat_overlay, title, args.alpha)
            else:
                _render_placeholder(ax, title, "VGGT tokens unavailable.")
        else:
            _render_placeholder(ax, title, "Unsupported mode.")
        return None

    # 2. Baseline
    baseline_attn_im = _render_attention_panel(
        ax_baseline_attn,
        baseline_payload.get("rgb") if baseline_payload else None,
        _extract_attention_map(baseline_payload),
        "Baseline – Readout Attention",
        args.alpha,
    )
    _add_aligned_colorbar(fig, ax_baseline_attn, baseline_attn_im, "Attention (norm)")

    # 3. VGGT Fusion (Attention)
    vggt_attn_im = _render_attention_panel(
        ax_vggt_attn,
        vggt_payload.get("rgb") if vggt_payload else None,
        _extract_attention_map(vggt_payload),
        "VGGT-Fusion – Readout Attention",
        args.alpha,
    )
    _add_aligned_colorbar(fig, ax_vggt_attn, vggt_attn_im, "Attention (norm)")

    # 4. VGGT Fusion (Similarity)
    vggt_sim_im = _render_similarity_panel(
        ax_vggt_sim,
        vggt_payload,
        "cross",
        "VGGT-Fusion – Cross Similarity",
    )
    _add_aligned_colorbar(fig, ax_vggt_sim, vggt_sim_im, "Similarity (norm)")

    # 5. VGGT Only
    vggt_only_attn_im = _render_attention_panel(
        ax_vggt_only_attn,
        vggt_only_payload.get("rgb") if vggt_only_payload else None,
        _extract_attention_map(vggt_only_payload),
        "VGGT-Only – Readout Attention",
        args.alpha,
    )
    _add_aligned_colorbar(fig, ax_vggt_only_attn, vggt_only_attn_im, "Attention (norm)")

    fig.suptitle(
        "Diagnostic Analysis of Spatial Attention and Feature Alignment",
        fontsize=18,
        y=0.98,
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches='tight')
    print(f"[plot_attention] Saved visualization to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()