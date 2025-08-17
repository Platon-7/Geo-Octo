import os
import sys

# Add compatibility shim before importing anything else
try:
    import jax.numpy as jnp
    if not hasattr(jnp, 'DeviceArray'):
        jnp.DeviceArray = jnp.ndarray
        print("[FIX] Added DeviceArray compatibility shim")
except ImportError:
    print("[WARNING] Could not import JAX")
    
import json
import argparse
import numpy as np
import jax

from octo.model.octo_model import OctoModel


def tree_map_with_path(fn, pytree, path=()):
    if isinstance(pytree, dict):
        return {k: tree_map_with_path(fn, v, path + (k,)) for k, v in pytree.items()}
    return fn(path, pytree)


def tree_leaves_with_path(pytree, path=()):
    if isinstance(pytree, dict):
        for k, v in pytree.items():
            yield from tree_leaves_with_path(v, path + (k,))
    else:
        yield path, pytree


def main():
    parser = argparse.ArgumentParser(description="Compare finetuned params against base pretrained params")
    parser.add_argument("--finetuned", required=True, help="Path to finetuned checkpoint dir")
    parser.add_argument("--base", required=True, help="Path to base pretrained checkpoint dir")
    parser.add_argument("--topk", type=int, default=30, help="Show top-k largest relative changes")
    args = parser.parse_args()

    print("=== PARAMETER DELTA CHECK ===")
    print(f"[INFO] Loading finetuned from: {args.finetuned}")
    ft = OctoModel.load_pretrained(args.finetuned)
    print(f"[INFO] Loading base from: {args.base}")
    base = OctoModel.load_pretrained(args.base)

    deltas = []
    norms = []
    names = []

    for path, ft_arr in tree_leaves_with_path(ft.params):
        # Skip non-array leaves
        if not hasattr(ft_arr, "shape"):
            continue
        # Navigate base params for same path; if missing, skip
        try:
            b = base.params
            for key in path:
                b = b[key]
            base_arr = b
        except Exception:
            continue

        if not hasattr(base_arr, "shape") or base_arr.shape != ft_arr.shape:
            continue

        diff = np.asarray(ft_arr) - np.asarray(base_arr)
        diff_norm = float(np.linalg.norm(diff))
        base_norm = float(np.linalg.norm(base_arr) + 1e-12)
        rel = diff_norm / base_norm
        deltas.append(rel)
        norms.append((diff_norm, base_norm))
        names.append("/".join(path))

    order = np.argsort(deltas)[::-1]
    print(f"[STATS] Compared {len(names)} parameter leaves.")
    if len(order) == 0:
        print("No comparable leaves found.")
        return

    topk = order[: args.topk]
    print("\nTop-k largest relative changes:")
    for idx in topk:
        rel = deltas[idx]
        dn, bn = norms[idx]
        print(f"{names[idx]}  rel={rel:.3e}  |diff|={dn:.3e}  |base|={bn:.3e}")

    median_rel = float(np.median(deltas))
    mean_rel = float(np.mean(deltas))
    print(f"\n[SUMMARY] mean_rel={mean_rel:.3e}  median_rel={median_rel:.3e}")

    if mean_rel < 1e-4 and median_rel < 1e-5:
        print("[DIAGNOSIS] Parameters barely changed. Finetuning may not have applied or optimizer frozen.")
    else:
        print("[DIAGNOSIS] Parameters show expected drift from base.")


if __name__ == "__main__":
    main()