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
import imageio.v2 as imageio


def _cosine_heatmap(obs_tokens: np.ndarray, readout_token: np.ndarray) -> np.ndarray:
    obs = obs_tokens.astype(np.float32)
    readout = readout_token.astype(np.float32)
    obs_norm = obs / (np.linalg.norm(obs, axis=1, keepdims=True) + 1e-8)
    readout_norm = readout / (np.linalg.norm(readout) + 1e-8)
    cos = obs_norm @ readout_norm
    side = int(np.sqrt(cos.shape[0]))
    return cos.reshape(side, side)


def plot(
    snapshot_path: str,
    output: Optional[str] = None,
    cmap: str = "inferno",
    suptitle: Optional[str] = "VGGT-Augmented Policy Failure Analysis",
    failure_image: Optional[str] = None,
    failure_caption: Optional[str] = None,
    dpi: int = 400,
) -> None:
    data = np.load(snapshot_path, allow_pickle=True)

    rgb = data["rgb"]
    obs_tokens = data["obs_tokens"]
    readout = data["readout_token"]
    heatmap = _cosine_heatmap(obs_tokens, readout)

    action = data.get("action", None)
    note = data.get("note", "")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.97)

    axes[0].imshow(rgb)
    axes[0].set_title("(a) Camera view")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    im = axes[1].imshow(heatmap, alpha=0.55, cmap=cmap, extent=(0, rgb.shape[1], rgb.shape[0], 0))
    axes[1].set_title("(b) Saliency map")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="cosine similarity")

    if failure_image is not None:
        try:
            failure_rgb = imageio.imread(failure_image)
            axes[2].imshow(failure_rgb)
        except Exception as e:
            print(f"[WARN] Could not load failure image {failure_image}: {e}")
            axes[2].imshow(rgb)
    else:
        axes[2].imshow(rgb)
    axes[2].axis("off")
    subtitle = "(c) Outcome"
    caption_lines = []
    if failure_caption:
        caption_lines.append(failure_caption)
    if isinstance(note, str) and note:
        caption_lines.append(note)
    if action is not None:
        caption_lines.append(f"Action: {np.array2string(action, precision=3, separator=', ')}")
    caption = "\n".join(caption_lines)
    axes[2].set_title(subtitle)
    if caption:
        axes[2].text(
            0.02,
            0.02,
            caption,
            transform=axes[2].transAxes,
            verticalalignment="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.65),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"[INFO] Saved attention figure to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot RGB + attention heatmap + outcome panels.")
    parser.add_argument("--snapshot", required=True, help="Path to .npz file with rgb, obs_tokens, readout_token.")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path.")
    parser.add_argument("--cmap", type=str, default="inferno", help="Matplotlib colormap for the heatmap overlay.")
    parser.add_argument("--suptitle", type=str, default="VGGT-Augmented Policy Failure Analysis")
    parser.add_argument("--failure-image", type=str, default=None, help="Optional image showing the failure outcome.")
    parser.add_argument("--failure-caption", type=str, default=None, help="Caption text for the outcome panel.")
    parser.add_argument("--dpi", type=int, default=400, help="Output DPI.")
    args = parser.parse_args()

    plot(
        args.snapshot,
        output=args.output,
        cmap=args.cmap,
        suptitle=args.suptitle,
        failure_image=args.failure_image,
        failure_caption=args.failure_caption,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
