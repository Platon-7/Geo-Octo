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


def _plot_single_snapshot(axs, data, cmap: str, failure_image: Optional[str], failure_caption: Optional[str]) -> None:
    rgb = data["rgb"]
    obs_tokens = data["obs_tokens"]
    readout = data["readout_token"]
    heatmap = _cosine_heatmap(obs_tokens, readout)

    action = data.get("action", None)
    note = data.get("note", "")

    axs[0].imshow(rgb)
    axs[0].axis("off")

    axs[1].imshow(rgb)
    im = axs[1].imshow(heatmap, alpha=0.55, cmap=cmap, extent=(0, rgb.shape[1], rgb.shape[0], 0))
    axs[1].axis("off")
    return im, action, note


def plot(
    baseline_snapshot: str,
    method_snapshot: str,
    output: Optional[str] = None,
    cmap: str = "inferno",
    suptitle: Optional[str] = "Policy Failure Analysis",
    baseline_failure_image: Optional[str] = None,
    method_failure_image: Optional[str] = None,
    baseline_failure_caption: Optional[str] = None,
    method_failure_caption: Optional[str] = None,
    dpi: int = 400,
) -> None:
    baseline_data = np.load(baseline_snapshot, allow_pickle=True)
    method_data = np.load(method_snapshot, allow_pickle=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.99)

    baseline_im, baseline_action, baseline_note = _plot_single_snapshot(
        axes[0, :2],
        baseline_data,
        cmap,
        baseline_failure_image,
        baseline_failure_caption,
    )
    axes[0, 0].set_title("(a) Baseline camera view")
    axes[0, 1].set_title("(b) Baseline saliency")
    fig.colorbar(baseline_im, ax=axes[0, 1], fraction=0.046, pad=0.04, label="cosine similarity")

    if baseline_failure_image:
        try:
            baseline_fail_rgb = imageio.imread(baseline_failure_image)
            axes[0, 2].imshow(baseline_fail_rgb)
        except Exception as e:
            print(f"[WARN] Could not load baseline failure image {baseline_failure_image}: {e}")
            axes[0, 2].imshow(baseline_data["rgb"])
    else:
        axes[0, 2].imshow(baseline_data["rgb"])
    axes[0, 2].axis("off")
    caption_lines = []
    if baseline_failure_caption:
        caption_lines.append(baseline_failure_caption)
    if isinstance(baseline_note, str) and baseline_note:
        caption_lines.append(baseline_note)
    if baseline_action is not None:
        caption_lines.append(f"Action: {np.array2string(baseline_action, precision=3, separator=', ')}")
    caption = "\n".join(caption_lines)
    axes[0, 2].set_title("(c) Baseline outcome")
    if caption:
        axes[0, 2].text(
            0.02,
            0.02,
            caption,
            transform=axes[0, 2].transAxes,
            verticalalignment="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.65),
        )

    method_im, method_action, method_note = _plot_single_snapshot(
        axes[1, :2],
        method_data,
        cmap,
        method_failure_image,
        method_failure_caption,
    )
    axes[1, 0].set_title("(d) Token Fusion camera view")
    axes[1, 1].set_title("(e) Token Fusion saliency")
    fig.colorbar(method_im, ax=axes[1, 1], fraction=0.046, pad=0.04, label="cosine similarity")

    if method_failure_image:
        try:
            method_fail_rgb = imageio.imread(method_failure_image)
            axes[1, 2].imshow(method_fail_rgb)
        except Exception as e:
            print(f"[WARN] Could not load method failure image {method_failure_image}: {e}")
            axes[1, 2].imshow(method_data["rgb"])
    else:
        axes[1, 2].imshow(method_data["rgb"])
    axes[1, 2].axis("off")
    caption_lines = []
    if method_failure_caption:
        caption_lines.append(method_failure_caption)
    if isinstance(method_note, str) and method_note:
        caption_lines.append(method_note)
    if method_action is not None:
        caption_lines.append(f"Action: {np.array2string(method_action, precision=3, separator=', ')}")
    caption = "\n".join(caption_lines)
    axes[1, 2].set_title("(f) Token Fusion outcome")
    if caption:
        axes[1, 2].text(
            0.02,
            0.02,
            caption,
            transform=axes[1, 2].transAxes,
            verticalalignment="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.65),
        )

    for row in axes:
        for ax in row:
            if ax in [row[2] for row in axes]:
                continue
            ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"[INFO] Saved attention figure to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot RGB + attention heatmap + outcome panels.")
    parser.add_argument("--baseline-snapshot", required=True, help="Path to baseline .npz.")
    parser.add_argument("--method-snapshot", required=True, help="Path to method .npz.")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path.")
    parser.add_argument("--cmap", type=str, default="inferno", help="Matplotlib colormap for the heatmap overlay.")
    parser.add_argument("--suptitle", type=str, default="Policy Failure Analysis")
    parser.add_argument("--baseline-failure-image", type=str, default=None)
    parser.add_argument("--method-failure-image", type=str, default=None)
    parser.add_argument("--baseline-failure-caption", type=str, default=None)
    parser.add_argument("--method-failure-caption", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=400, help="Output DPI.")
    args = parser.parse_args()

    plot(
        baseline_snapshot=args.baseline_snapshot,
        method_snapshot=args.method_snapshot,
        output=args.output,
        cmap=args.cmap,
        suptitle=args.suptitle,
        baseline_failure_image=args.baseline_failure_image,
        method_failure_image=args.method_failure_image,
        baseline_failure_caption=args.baseline_failure_caption,
        method_failure_caption=args.method_failure_caption,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
