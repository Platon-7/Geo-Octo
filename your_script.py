#!/usr/bin/env python3
"""
Fix tampered heart annotations in SegTHOR-like dataset.

Usage:
  python your_script.py --source_dir /absolute/path/to/data

Behavior:
  - Finds every Patient_XX/GT.nii.gz under --source_dir
  - Extracts only the heart label (default id=2)
  - Applies a fast sparse coordinate transform to the heart mask only,
    using the composed affine (no heavy volume-wide resampling). The known
    sequence is T1 -> R2 -> T3 -> T4 applied on coordinates.
  - Writes the corrected label map as GT_fixed.nii.gz in the same folder.
  - Non-heart labels are preserved exactly; heart voxels are only written
    into background positions to ensure other classes remain unaffected.

Notes:
  - Transform parameters are specified in voxel units; rotation is about Z.
  - For label images we use order=0 and prefilter=False for speed and to
    avoid interpolation artifacts.
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform


def build_forward_matrix() -> np.ndarray:
    """Return forward 4x4 matrix M that maps tampered -> corrected coords.

    Using homogeneous transforms:
      T1 = translate(+275, +200, 0)
      R2 = rotate_z(phi) with phi = -(27/180)*pi
      T3 = T1^{-1} = translate(-275, -200, 0)
      T4 = translate(+50, +40, +15)

    We empirically match the correct composition used in your working
    implementation: M = T1 @ Rz @ T3 @ T4
    """

    # Build transforms directly in array index order (x, y, z), consistent
    # with the original working version.
    def T(tx: float, ty: float, tz: float) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        m[0, 3] = tx
        m[1, 3] = ty
        m[2, 3] = tz
        return m

    phi = -27.0 * np.pi / 180.0
    cos_p = float(np.cos(phi))
    sin_p = float(np.sin(phi))
    Rz = np.array(
        [
            [cos_p, -sin_p, 0.0, 0.0],
            [sin_p, cos_p, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    T1 = T(275.0, 200.0, 0.0)
    T3 = T(-275.0, -200.0, 0.0)  # T1^{-1}
    T4 = T(50.0, 40.0, 15.0)

    M = T1 @ Rz @ T3 @ T4
    return M


def fix_single_gt(gt_path: str, heart_label: int) -> str:
    """Fix a single GT.nii.gz and save GT_fixed.nii.gz next to it.

    Returns the output path.
    """
    img = nib.load(gt_path)
    data = np.asanyarray(img.get_fdata(dtype=np.float32))
    # Convert to an integer label array without changing values
    # (typical GT are small ints 0..4). Preserve original dtype on save.
    if np.issubdtype(data.dtype, np.floating):
        # Cast via round to be safe if dataset was stored as float
        labels = data.astype(np.uint16)
    else:
        labels = data

    # Extract heart mask only
    heart_mask = labels == heart_label

    # Compose transform once
    M = build_forward_matrix()  # forward mapping tampered -> corrected
    Minv = np.linalg.inv(M)
    A_full = Minv[:3, :3].astype(np.float64)
    b_full = Minv[:3, 3].astype(np.float64)

    # Compute tight output ROI by transforming heart voxel coords forward
    idx = np.argwhere(heart_mask)
    corrected_heart = np.zeros_like(heart_mask, dtype=bool)
    if idx.size != 0:
        ones = np.ones((idx.shape[0], 1), dtype=np.float64)
        pts = np.concatenate([idx.astype(np.float64), ones], axis=1)  # (N,4) with (x,y,z,1)
        new = pts @ M.T
        new_xyz_f = new[:, :3]
        mins = np.floor(new_xyz_f.min(axis=0) - 1).astype(int)
        maxs = np.ceil(new_xyz_f.max(axis=0) + 2).astype(int)
        X, Y, Z = heart_mask.shape
        start = np.maximum(mins, 0)
        end = np.minimum(maxs, np.array([X, Y, Z], dtype=int))
        if np.any(end - start <= 0):
            # Fallback to whole-volume (very unlikely)
            roi_slices = (slice(0, X), slice(0, Y), slice(0, Z))
            start = np.array([0, 0, 0], dtype=int)
        else:
            roi_slices = (slice(start[0], end[0]), slice(start[1], end[1]), slice(start[2], end[2]))

        # Adjust offset for ROI-local coordinates: x = A @ (o + start) + b
        b_roi = (A_full @ start.astype(np.float64)) + b_full
        roi_shape = (end - start).tolist()
        roi_result = affine_transform(
            heart_mask.astype(np.uint8),
            matrix=A_full,
            offset=b_roi,
            output_shape=roi_shape,
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(bool)

        corrected_heart[roi_slices] = roi_result

    # Preserve non-heart labels exactly
    output_labels = labels.copy()
    output_labels[heart_mask] = 0  # remove tampered heart completely
    # Only write heart into background positions to avoid altering other classes
    write_positions = corrected_heart & (output_labels == 0)
    if np.issubdtype(output_labels.dtype, np.floating):
        output_labels[write_positions] = float(heart_label)
    else:
        output_labels[write_positions] = heart_label

    out_img = nib.Nifti1Image(output_labels.astype(labels.dtype, copy=False), img.affine, img.header)
    out_path = os.path.join(os.path.dirname(gt_path), "GT_fixed.nii.gz")
    nib.save(out_img, out_path)
    return out_path


def find_all_gt_paths(root: str) -> List[str]:
    # Look two levels deep for Patient_XX/GT.nii.gz
    # Use glob to be robust to exact patient dir naming
    pattern = os.path.join(os.path.abspath(root), "**", "GT.nii.gz")
    return sorted(glob(pattern, recursive=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix tampered heart annotations (GT.nii.gz -> GT_fixed.nii.gz)")
    parser.add_argument("--source_dir", required=True, type=str, help="Absolute path to data root containing Patient_* subfolders")
    parser.add_argument("--heart_label", default=2, type=int, help="Numeric label id for the heart (default: 2)")
    args = parser.parse_args()

    gt_paths = find_all_gt_paths(args.source_dir)
    if not gt_paths:
        raise SystemExit(f"No GT.nii.gz files found under {args.source_dir}")

    # Prebuild matrix once (also warms up numpy just a bit)
    _ = build_forward_matrix()

    # Sequential processing only
    results: List[str] = []
    for p in gt_paths:
        out = fix_single_gt(p, args.heart_label)
        results.append(out)

    # Print a brief report
    print(f"Fixed {len(results)} cases. Example output: {results[0]}")


if __name__ == "__main__":
    main()

