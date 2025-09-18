#!/usr/bin/env python3
"""
Fix tampered heart annotations in SegTHOR-like dataset.

Usage:
  python your_script.py --source_dir /absolute/path/to/data

Behavior:
  - Finds every Patient_XX/GT.nii.gz under --source_dir
  - Extracts only the heart label (default id=2)
  - Applies the inverse of the tampering by sampling the tampered mask
    with the forward tamper matrix (scipy.ndimage.affine_transform maps
    output -> input). The known sequence is T1 -> R2 -> T3 -> T4 applied
    on coordinates, which composes to M = T4 @ T3 @ R2 @ T1.
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


def build_tamper_affine_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """Return (A, b) for scipy.ndimage.affine_transform.

    Using homogeneous transforms:
      T1 = translate(+275, +200, 0)
      R2 = rotate_z(phi) with phi = -(27/180)*pi
      T3 = T1^{-1} = translate(-275, -200, 0)
      T4 = translate(+50, +40, +15)

    The composed tamper mapping on coordinates is:
      M = T4 @ T3 @ R2 @ T1

    For affine_transform, the mapping is input(x) sampled at x = A @ o + b.
    To recover the pre-tampered mask, we produce output o on the original grid
    and sample at x = M @ o in the tampered grid, so we pass A,b from M.
    """

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

    M = T4 @ T3 @ Rz @ T1  # 4x4
    A = M[:3, :3].astype(np.float64)
    b = M[:3, 3].astype(np.float64)
    return A, b


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
    A, b = build_tamper_affine_matrix()

    # Apply transform on the heart mask only
    corrected_heart = affine_transform(
        heart_mask.astype(np.uint8),
        matrix=A,
        offset=b,
        output_shape=heart_mask.shape,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    corrected_heart = corrected_heart.astype(bool)

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


def _fix_worker(args: Tuple[str, int]) -> str:
    """Top-level worker for multiprocessing (must be picklable)."""
    path, heart_label = args
    return fix_single_gt(path, heart_label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix tampered heart annotations (GT.nii.gz -> GT_fixed.nii.gz)")
    parser.add_argument("--source_dir", required=True, type=str, help="Absolute path to data root containing Patient_* subfolders")
    parser.add_argument("--heart_label", default=2, type=int, help="Numeric label id for the heart (default: 2)")
    parser.add_argument("--workers", default=0, type=int, help="Number of parallel workers (0=auto)")
    args = parser.parse_args()

    gt_paths = find_all_gt_paths(args.source_dir)
    if not gt_paths:
        raise SystemExit(f"No GT.nii.gz files found under {args.source_dir}")

    # Prebuild matrix once (also warms up numpy just a bit)
    _ = build_tamper_affine_matrix()

    # Process, optionally in parallel. Multiprocessing has overhead; the
    # operation is relatively fast per volume, so limit to a modest pool size.
    num_workers = args.workers
    if num_workers <= 0:
        try:
            import multiprocessing as mp

            cpu_cnt = max(1, mp.cpu_count() - 1)
            num_workers = min(8, cpu_cnt)
        except Exception:
            num_workers = 1

    results: List[str] = []
    if num_workers == 1:
        for p in gt_paths:
            out = fix_single_gt(p, args.heart_label)
            results.append(out)
    else:
        import multiprocessing as mp

        with mp.Pool(processes=num_workers) as pool:
            iterable = ((p, args.heart_label) for p in gt_paths)
            for out in pool.imap_unordered(_fix_worker, iterable):
                results.append(out)

    # Print a brief report
    print(f"Fixed {len(results)} cases. Example output: {results[0]}")


if __name__ == "__main__":
    main()

