"""
Render the multi-panel "policy failure" visualization (Plot 3).

This script expects an `.npz` file storing:
  - rgb: (H, W, 3) uint8 array for the task frame.
  - obs_tokens: (N, D) array of observation tokens after the shared projection.
  - readout_token: (D,) array representing the action readout queried for the impending action.
Optional extras that will be overlaid into the caption if present:
  - action: (action_dim,) array with the model's action.
  - note: string describing the outcome ("Collision", etc.).

You can capture these tensors during evaluation by inserting the snippet shown in the
module docstring into the policy loop (using `model.module.apply(..., mutable=['intermediates'])`).

Example:
    python analysis/plot_attention_map.py \
        --snapshot path/to/concat_failure_attention.npz \
        --output figure_attention.png
"""

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _cosine_heatmap(obs_tokens: np.ndarray, readout_token: np.ndarray) -> np.ndarray:
    obs = obs_tokens.astype(np.float32)
    readout = readout_token.astype(np.float32)
    obs_norm = obs / (np.linalg.norm(obs, axis=1, keepdims=True) + 1e-8)
    readout_norm = readout / (np.linalg.norm(readout) + 1e-8)
    cos = obs_norm @ readout_norm
    side = int(np.sqrt(cos.shape[0]))
    return cos.reshape(side, side)


def plot(snapshot_path: str, output: Optional[str] = None, cmap: str = "inferno") -> None:
    data = np.load(snapshot_path, allow_pickle=True)

    rgb = data["rgb"]
    obs_tokens = data["obs_tokens"]
    readout = data["readout_token"]
    heatmap = _cosine_heatmap(obs_tokens, readout)

    action = data.get("action", None)
    note = data.get("note", "")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(rgb)
    axes[0].set_title("Camera frame")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    im = axes[1].imshow(heatmap, alpha=0.55, cmap=cmap, extent=(0, rgb.shape[1], rgb.shape[0], 0))
    axes[1].set_title("Cosine saliency")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="cosine similarity")

    axes[2].imshow(rgb)
    axes[2].axis("off")
    subtitle = "Outcome"
    caption = note if isinstance(note, str) else ""
    if action is not None:
        caption = caption + f"\nAction: {np.array2string(action, precision=3, separator=', ')}"
    axes[2].set_title(subtitle)
    axes[2].text(
        0.02,
        0.02,
        caption,
        transform=axes[2].transAxes,
        verticalalignment="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.65),
    )

    plt.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=250, bbox_inches="tight")
        print(f"[INFO] Saved attention figure to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot RGB + attention heatmap + outcome panels.")
    parser.add_argument("--snapshot", required=True, help="Path to .npz file with rgb, obs_tokens, readout_token.")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path.")
    parser.add_argument("--cmap", type=str, default="inferno", help="Matplotlib colormap for the heatmap overlay.")
    args = parser.parse_args()

    plot(args.snapshot, output=args.output, cmap=args.cmap)


if __name__ == "__main__":
    main()
