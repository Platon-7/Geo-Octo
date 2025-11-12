"""
Utility for generating the two-panel "Input Quality" visualization (Plot 2).

Given a `.npz` snapshot saved by `run_libero_eval_vggt_pointmap.py` with
`--snapshot_enable True`, this script will render:
  • Left: the RGB observation used by the policy.
  • Right: a sparse 3D scatter of the VGGT pointmap (x, y, z) coloured by depth.

Example:
    python analysis/plot_pointmap_snapshot.py \
        --snapshot path/to/snapshot_task-Lift_Toolbox_ep00_step010.npz \
        --stride 6 \
        --output plot_pointmap.png
"""

import argparse
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (import registers 3D projection)


def _downsample(pointmap: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strided xyz coordinates for plotting."""
    xyz = pointmap[..., :3]
    xyz = xyz[::stride, ::stride, :]
    xs = xyz[..., 0].reshape(-1)
    ys = xyz[..., 1].reshape(-1)
    zs = xyz[..., 2].reshape(-1)
    return xs, ys, zs


def plot_snapshot(path: str, stride: int = 4, output: Optional[str] = None, show_confidence: bool = False) -> None:
    data = np.load(path)
    rgb = data["rgb"]
    if "rgb_preprocessed" not in data:
        print("[WARN] Snapshot missing 'rgb_preprocessed'; using raw RGB colours.")

    if "pointmap_raw" in data:
        pointmap = data["pointmap_raw"]
    else:
        pointmap = data["pointmap_normalized"]

    xs, ys, zs = _downsample(pointmap, stride)
    n_points = xs.shape[0]

    if show_confidence:
        payload = data.get("pointmap_raw", pointmap)[::stride, ::stride, 3].reshape(-1)
        payload = payload[:n_points]
        colors = payload
        colorbar_label = "confidence"
    else:
        # Colour points by RGB. Prefer the preprocessed VGGT input (expects float32 in [0,1]).
        rgb_for_coloring = data.get("rgb_preprocessed", rgb).astype(np.float32)
        if rgb_for_coloring.max() > 1.0:
            rgb_for_coloring = rgb_for_coloring / 255.0
        rgb_for_coloring = np.clip(rgb_for_coloring, 0.0, 1.0)
        colors = rgb_for_coloring[::stride, ::stride, :].reshape(-1, 3)
        colors = colors[:n_points]
        payload = None
        colorbar_label = None

    fig = plt.figure(figsize=(10, 4.5))

    ax_rgb = fig.add_subplot(1, 2, 1)
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("Policy RGB input")
    ax_rgb.axis("off")

    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    scatter_kwargs = dict(s=4, alpha=0.85, depthshade=False)
    if show_confidence:
        scatter = ax_3d.scatter(xs, ys, zs, c=colors, cmap=cm.viridis, **scatter_kwargs)
        fig.colorbar(scatter, ax=ax_3d, shrink=0.6, pad=0.1, label=colorbar_label)
    else:
        scatter = ax_3d.scatter(xs, ys, zs, c=colors, **scatter_kwargs)

    ax_3d.set_title("VGGT pointmap (RGB-coloured)" if not show_confidence else "VGGT pointmap (confidence)")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.view_init(elev=30.0, azim=-60.0)

    plt.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved figure to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot RGB & pointmap snapshot.")
    parser.add_argument("--snapshot", required=True, help="Path to snapshot .npz file.")
    parser.add_argument("--stride", type=int, default=4, help="Sampling stride for the point cloud.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the figure.")
    parser.add_argument(
        "--colour-by-confidence",
        action="store_true",
        help="Colour scatter points with confidence instead of depth.",
    )
    args = parser.parse_args()

    plot_snapshot(args.snapshot, stride=args.stride, output=args.output, show_confidence=args.colour_by_confidence)


if __name__ == "__main__":
    main()
