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
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (import registers 3D projection)


def plot_snapshot(
    path: str,
    stride: int = 4,
    output: Optional[str] = None,
    show_confidence: bool = False,
    confidence_threshold: Optional[float] = None,
    use_normalized: bool = False,
    invert_z: bool = False,
    flip_pc1: bool = False,
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

    coords = pointmap[..., :3]
    conf_map = pointmap[..., 3]
    h, w, _ = coords.shape

    coords_flat = coords.reshape(-1, 3)
    center = coords_flat.mean(axis=0, keepdims=True)
    coords_centered = coords_flat - center
    _, _, Vt = np.linalg.svd(coords_centered, full_matrices=False)
    coords_rot_flat = coords_centered @ Vt.T
    if coords_rot_flat[:, 2].mean() < 0:
        coords_rot_flat[:, 2] *= -1.0
    if invert_z:
        coords_rot_flat[:, 2] *= -1.0
    if flip_pc1:
        coords_rot_flat[:, 1] *= -1.0
        coords_rot_flat[:, 2] *= -1.0
    coords_rot = coords_rot_flat.reshape(h, w, 3)

    depth_map = coords_rot[:, :, 2]

    coords_ds = coords_rot[::stride, ::stride, :]
    xs = coords_ds[..., 0].reshape(-1)
    ys = coords_ds[..., 1].reshape(-1)
    zs = coords_ds[..., 2].reshape(-1)
    conf_ds = conf_map[::stride, ::stride].reshape(-1)

    if confidence_threshold is not None:
        mask = conf_ds >= confidence_threshold
        xs, ys, zs = xs[mask], ys[mask], zs[mask]
        conf_ds = conf_ds[mask]
        print(f"  confidence threshold {confidence_threshold}: kept {xs.shape[0]} points")
        if xs.size == 0:
            raise ValueError("All points filtered out by confidence threshold.")
    else:
        mask = slice(None)

    if show_confidence:
        colors = conf_ds
        cmap = cm.viridis
        colorbar_label = "confidence"
    else:
        colors_full = rgb_pre[::stride, ::stride, :].reshape(-1, 3)
        colors = colors_full[mask] if isinstance(mask, np.ndarray) else colors_full
        cmap = None
        colorbar_label = None

    fig = plt.figure(figsize=(15, 5))

    ax_rgb = fig.add_subplot(1, 3, 1)
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("Policy RGB input")
    ax_rgb.axis("off")

    ax_depth = fig.add_subplot(1, 3, 2)
    im_depth = ax_depth.imshow(depth_map, cmap="viridis")
    ax_depth.set_title("VGGT depth map (after PCA alignment)")
    ax_depth.axis("off")
    fig.colorbar(im_depth, ax=ax_depth, fraction=0.046, pad=0.04, label="PC3 height")

    ax_3d = fig.add_subplot(1, 3, 3, projection="3d")
    scatter_kwargs = dict(s=6, alpha=0.9, depthshade=False)
    if show_confidence:
        scatter = ax_3d.scatter(xs, ys, zs, c=colors, cmap=cmap, **scatter_kwargs)
        fig.colorbar(scatter, ax=ax_3d, fraction=0.046, pad=0.04, label=colorbar_label)
    else:
        scatter = ax_3d.scatter(xs, ys, zs, c=colors, **scatter_kwargs)

    ax_3d.set_title("VGGT pointmap (PCA-aligned)" if not show_confidence else "VGGT pointmap (confidence)")
    ax_3d.set_xlabel("PC1 (plane axis)")
    ax_3d.set_ylabel("PC2 (plane axis)")
    ax_3d.set_zlabel("PC3 (height)")
    ax_3d.view_init(elev=40.0, azim=-45.0)

    x_range = xs.max() - xs.min() if xs.size else 1.0
    y_range = ys.max() - ys.min() if ys.size else 1.0
    z_range = max(zs.max() - zs.min(), 1e-5) if zs.size else 1.0
    ax_3d.set_box_aspect((x_range, y_range, z_range * 2.0))

    plt.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved figure to {output}")
    else:
        plt.show()

    # Optional: scatter points on a "sphere" view (e.g., onto the unit sphere for intuition)
    projected = coords_rot_flat / (np.linalg.norm(coords_rot_flat, axis=1, keepdims=True) + 1e-8)
    projected = projected.reshape(h, w, 3)
    sphere_ds = projected[::stride, ::stride, :]
    xs_s, ys_s, zs_s = sphere_ds[..., 0].reshape(-1), sphere_ds[..., 1].reshape(-1), sphere_ds[..., 2].reshape(-1)
    if show_confidence:
        colors_s = conf_ds if isinstance(conf_ds, np.ndarray) else conf_ds
    else:
        colors_s = colors

    fig_sphere = plt.figure(figsize=(6, 6))
    ax_sphere = fig_sphere.add_subplot(1, 1, 1, projection="3d")
    scatter_s = ax_sphere.scatter(xs_s, ys_s, zs_s, c=colors_s, s=4, alpha=0.7, depthshade=False)
    ax_sphere.set_title("Spherical projection of VGGT geometry")
    ax_sphere.set_axis_off()
    ax_sphere.set_box_aspect((1, 1, 1))
    ax_sphere.view_init(elev=30.0, azim=-45.0)
    if output:
        base, ext = os.path.splitext(output)
        sphere_path = f"{base}_sphere{ext}"
        fig_sphere.savefig(sphere_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved spherical projection to {sphere_path}")
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
    parser.add_argument(
        "--flip-pc1",
        action="store_true",
        help="Rotate 180 degrees around the first PCA axis (flips PC2/PC3).",
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
        flip_pc1=args.flip_pc1,
    )


if __name__ == "__main__":
    main()
