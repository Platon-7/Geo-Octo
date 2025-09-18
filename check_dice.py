#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return 2.0 * inter / float(a_sum + b_sum)


def list_patients(root: str) -> List[str]:
    cand = sorted([p for p in glob(os.path.join(os.path.abspath(root), "Patient_*")) if os.path.isdir(p)])
    return cand


def load_labels(path: str) -> np.ndarray:
    img = nib.load(path)
    data = np.asanyarray(img.get_fdata(dtype=np.float32))
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.uint16)
    return data


def compute_dice_by_label(a: np.ndarray, b: np.ndarray, labels: List[int]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for lab in labels:
        scores[int(lab)] = dice_score(a == lab, b == lab)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-class Dice for GT_fixed vs baseline.")
    parser.add_argument("--source_dir", required=True, type=str, help="Path with Patient_*/ containing GT.nii.gz and GT_fixed.nii.gz")
    parser.add_argument("--baseline", default="gt", choices=["gt", "reference"], help="What to compare GT_fixed against")
    parser.add_argument("--reference_dir", type=str, default=None, help="If baseline=reference, path with clean GT under matching Patient_* folders")
    parser.add_argument("--heart_label", type=int, default=2, help="Numeric id for heart label")
    parser.add_argument("--background_label", type=int, default=0, help="Numeric id for background (excluded from 'non-heart unaffected' check)")
    args = parser.parse_args()

    patients = list_patients(args.source_dir)
    if not patients:
        raise SystemExit(f"No Patient_* directories found under {args.source_dir}")

    per_class_scores: Dict[int, List[float]] = {}
    failed_paths: List[str] = []

    for p in patients:
        fixed_path = os.path.join(p, "GT_fixed.nii.gz")
        base_path = os.path.join(p, "GT.nii.gz") if args.baseline == "gt" else os.path.join(
            os.path.join(args.reference_dir or ""), os.path.basename(p), "GT.nii.gz"
        )

        if not os.path.exists(fixed_path) or not os.path.exists(base_path):
            failed_paths.append(p)
            continue

        fixed = load_labels(fixed_path)
        base = load_labels(base_path)
        if fixed.shape != base.shape:
            failed_paths.append(p + " (shape mismatch)")
            continue

        labels = sorted(set(np.unique(fixed)).union(set(np.unique(base))))
        scores = compute_dice_by_label(fixed, base, labels)
        for lab, sc in scores.items():
            per_class_scores.setdefault(lab, []).append(sc)

    # Print report
    if failed_paths:
        print("Skipped due to missing files:")
        for s in failed_paths:
            print("  -", s)

    if not per_class_scores:
        raise SystemExit("No scores computed.")

    print("Per-class Dice (mean ± std, min)")
    for lab in sorted(per_class_scores):
        arr = np.array(per_class_scores[lab], dtype=np.float64)
        print(f"label {lab}: {arr.mean():.6f} ± {arr.std():.6f} (min {arr.min():.6f}) over {arr.size} cases")

    # Rubric checks helpful summary
    heart = args.heart_label
    if heart in per_class_scores:
        heart_arr = np.array(per_class_scores[heart], dtype=np.float64)
        print(f"Heart label {heart} -> mean {heart_arr.mean():.6f}, min {heart_arr.min():.6f}")
    # Non-heart exactness if compared against GT
    if args.baseline == "gt":
        # Exclude heart and background from the 'non-heart unaffected' check
        non_hearts = [
            l for l in per_class_scores.keys()
            if l != args.heart_label and l != args.background_label
        ]
        all_ok = True
        for l in non_hearts:
            arr = np.array(per_class_scores[l])
            if not np.allclose(arr, 1.0):
                all_ok = False
                break
        print(f"Non-heart unaffected (vs GT): {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()

