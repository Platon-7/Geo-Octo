import argparse
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _accumulate_stats_for_dataset(
    data_dir: str,
    dataset_name: str,
    split: str = "train",
    max_episodes: int = 50,
    max_tokens: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Accumulate per-dimension sums and squared sums over `vggt_tokens`.

    Returns (sum_vec, sumsq_vec, count) over the last dimension (feature dim).
    """
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds = builder.as_dataset(split=split)

    sum_vec = None  # shape [C]
    sumsq_vec = None  # shape [C]
    count = 0

    for epi_idx, episode in enumerate(tfds.as_numpy(ds)):
        if epi_idx >= max_episodes:
            break
        steps = list(episode.get("steps", []))
        if not steps:
            continue
        # Iterate steps and accumulate tokens
        for step in steps:
            obs = step.get("observation", {})
            tokens = obs.get("vggt_tokens")
            if tokens is None:
                continue

            arr = np.asarray(tokens)
            # Expected shapes: (64, 512) or (TBD)
            if arr.ndim == 2:
                # [N=64, C=512]
                pass
            elif arr.ndim == 3:
                # Some datasets might have (H, W, C) or (T, H, W)
                # Try to coerce to [N, C] when C is last
                if arr.shape[-1] <= 8:
                    # assume (H, W, C_small) -> flatten spatial and keep C_small as channels
                    arr = arr.reshape(-1, arr.shape[-1])
                else:
                    # assume (H, W, C=512)
                    arr = arr.reshape(-1, arr.shape[-1])
            else:
                # As a fallback, flatten everything but the last dim if present
                if arr.size == 0:
                    continue
                if arr.ndim > 2:
                    arr = arr.reshape(-1, arr.shape[-1])
                else:
                    # 1D vector, treat as a single token
                    arr = arr.reshape(1, -1)

            # Now arr is [N_tokens, C]
            arr = arr.astype(np.float32, copy=False)
            if sum_vec is None:
                sum_vec = np.zeros((arr.shape[-1],), dtype=np.float64)
                sumsq_vec = np.zeros((arr.shape[-1],), dtype=np.float64)

            # Cap total tokens for speed
            remaining = max_tokens - count
            if remaining <= 0:
                break
            if arr.shape[0] > remaining:
                arr = arr[:remaining]

            sum_vec += arr.sum(axis=0, dtype=np.float64)
            sumsq_vec += (arr.astype(np.float64) ** 2).sum(axis=0)
            count += arr.shape[0]

        if count >= max_tokens:
            break

    if sum_vec is None:
        raise RuntimeError(
            f"No vggt_tokens found in dataset {dataset_name} at {data_dir} (split={split})."
        )

    return sum_vec, sumsq_vec, count


def _finalize_stats(sum_vec: np.ndarray, sumsq_vec: np.ndarray, count: int):
    mean = sum_vec / max(1, count)
    var = np.maximum(sumsq_vec / max(1, count) - mean ** 2, 0.0)
    std = np.sqrt(var)
    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Compare VGGT token stats between two TFDS datasets"
    )
    parser.add_argument("--data_dir1", type=str, required=True)
    parser.add_argument("--dataset1", type=str, required=True)
    parser.add_argument("--data_dir2", type=str, required=True)
    parser.add_argument("--dataset2", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=500000)
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    print(f"Dataset 1: {args.dataset1} @ {args.data_dir1}")
    s1, s1q, n1 = _accumulate_stats_for_dataset(
        args.data_dir1, args.dataset1, args.split, args.max_episodes, args.max_tokens
    )
    m1, std1 = _finalize_stats(s1, s1q, n1)
    print(f"  tokens used: {n1}")
    print(f"  per-dim mean:   mean={float(m1.mean()):.6f} min={float(m1.min()):.6f} max={float(m1.max()):.6f}")
    print(f"  per-dim std:    mean={float(std1.mean()):.6f} min={float(std1.min()):.6f} max={float(std1.max()):.6f}")

    print()
    print(f"Dataset 2: {args.dataset2} @ {args.data_dir2}")
    s2, s2q, n2 = _accumulate_stats_for_dataset(
        args.data_dir2, args.dataset2, args.split, args.max_episodes, args.max_tokens
    )
    m2, std2 = _finalize_stats(s2, s2q, n2)
    print(f"  tokens used: {n2}")
    print(f"  per-dim mean:   mean={float(m2.mean()):.6f} min={float(m2.min()):.6f} max={float(m2.max()):.6f}")
    print(f"  per-dim std:    mean={float(std2.mean()):.6f} min={float(std2.min()):.6f} max={float(std2.max()):.6f}")

    print()
    # Simple deltas
    dm = m1 - m2
    ds = std1 - std2
    print("Differences (dataset1 - dataset2):")
    print(f"  mean diff: mean={float(dm.mean()):.6f} min={float(dm.min()):.6f} max={float(dm.max()):.6f}")
    print(f"  std  diff: mean={float(ds.mean()):.6f} min={float(ds.min()):.6f} max={float(ds.max()):.6f}")


if __name__ == "__main__":
    main()

