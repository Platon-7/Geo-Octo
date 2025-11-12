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


def plot_snapshot(
    path: str,
    stride: int = 4,
    output: Optional[str] = None,
    show_confidence: bool = False,
    confidence_threshold: Optional[float] = None,
    use_normalized: bool = False,
    invert_z: bool = False,
) -> None:
    data = np.load(path)
    rgb = data["rgb"]
    rgb_pre = data.get("rgb_preprocessed", rgb).astype(np.float32)
    if rgb_pre.max() > 1.0:
        rgb_pre /= 255.0
    rgb_pre = np.clip(rgb_pre, 0.0, 1.0)

    print(
        f"[DEBUG] snapshot={path}\n"
        f"  rgb.shape={rgb.shape} dtype={rgb.dtype} min={rgb.min()} max={rgb.max()}\n"
        f"  rgb_preprocessed.shape={rgb_pre.shape} dtype={rgb_pre.dtype} "
        f"min={rgb_pre.min():.4f} max={rgb_pre.max():.4f}"
    )

    pointmap_key = "pointmap_normalized" if use_normalized else "pointmap_raw"
    if pointmap_key not in data:
        raise ValueError(f"{pointmap_key} not found in snapshot.")
    pointmap = data[pointmap_key].astype(np.float32)

    print(
        f"  {pointmap_key}.shape={pointmap.shape} "
        f"min={float(np.nanmin(pointmap)):.4f} max={float(np.nanmax(pointmap)):.4f}"
    )

    xs, ys, zs = _downsample(pointmap, stride)
    if invert_z:
        zs = -zs
    n_points = xs.shape[0]

    conf = pointmap[..., 3]
    if confidence_threshold is not None:
        conf_ds = conf[::stride, ::stride].reshape(-1)[:n_points]
        mask = conf_ds >= confidence_threshold
        xs, ys, zs = xs[mask], ys[mask], zs[mask]
        n_points = xs.shape[0]
        print(f"  confidence threshold {confidence_threshold}: kept {n_points} points")
        if n_points == 0:
            raise ValueError("All points filtered out by confidence threshold.")
    else:
        mask = slice(None)

    if show_confidence:
        colors = conf[::stride, ::stride].reshape(-1)[:n_points]
        cmap = cm.viridis
        colorbar_label = "confidence"
    else:
        colors = rgb_pre[::stride, ::stride, :].reshape(-1, 3)[:n_points]
        cmap = None
        colorbar_label = None

    fig = plt.figure(figsize=(14, 5))

    ax_rgb = fig.add_subplot(1, 3, 1)
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("Policy RGB input")
    ax_rgb.axis("off")

    ax_depth = fig.add_subplot(1, 3, 2)
    depth_map = pointmap[..., 2]
    if invert_z:
        depth_map = -depth_map
    im_depth = ax_depth.imshow(depth_map, cmap="viridis")
    ax_depth.set_title("VGGT depth map")
    ax_depth.axis("off")
    fig.colorbar(im_depth, ax=ax_depth, fraction=0.046, pad=0.04, label="z")

    ax_3d = fig.add_subplot(1, 3, 3, projection="3d")
    scatter_kwargs = dict(s=4, alpha=0.85, depthshade=False)
    if show_confidence:
        scatter = ax_3d.scatter(xs, ys, zs, c=colors, cmap=cmap, **scatter_kwargs)
        fig.colorbar(scatter, ax=ax_3d, fraction=0.046, pad=0.04, label=colorbar_label)
    else:
        scatter = ax_3d.scatter(xs, ys, zs, c=colors, **scatter_kwargs)

    ax_3d.set_title("VGGT pointmap (RGB-coloured)" if not show_confidence else "VGGT pointmap (confidence)")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.view_init(elev=30.0, azim=-60.0)

    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    z_range = zs.max() - zs.min()
    ax_3d.set_box_aspect((x_range, y_range, z_range if z_range > 0 else 1.0))

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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Only plot points with confidence >= threshold.",
    )
    parser.add_argument(
        "--use-normalized",
        action="store_true",
        help="Use normalized pointmap instead of raw.",
    )
    parser.add_argument(
        "--invert-z",
        action="store_true",
        help="Flip the sign of the z-axis (for alternative camera conventions).",
    )
    args = parser.parse_args()

    plot_snapshot(
        args.snapshot,
        stride=args.stride,
        output=args.output,
        show_confidence=args.colour_by_confidence,
        confidence_threshold=args.confidence_threshold,
        use_normalized=args.use_normalized,
        invert_z=args.invert_z,
    )


if __name__ == "__main__":
    main()
