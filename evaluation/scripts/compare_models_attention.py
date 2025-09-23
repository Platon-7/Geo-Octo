import argparse
import os
import json
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# Octo imports
from octo.model.octo_model import OctoModel

# Eval helpers (image prep, env)
from evaluation.supporting_files.robot_utils import (
    set_seed_everywhere,
    get_model as _get_model_unused,  # not used; we load OctoModel directly
    get_image_resize_size,
)
from evaluation.supporting_files.libero_utils import (
    get_libero_env,
)

# ONNX VGGT compressor
from evaluation.scripts.run_libero_eval_vggt import (
    compute_compressed_vggt_tokens,
    GenerateConfig as VggtCfg,
)
from libero.libero import benchmark


def _build_vggt_ctx(onnx_path: str, compressor_path: str, input_res: int = 518, use_cuda: bool = True):
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("onnxruntime not available for VGGT token computation") from e

    providers = ['CUDAExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    from vggt_compression_analysis import VGGTCompressor
    compressor = VGGTCompressor.load_compressor(compressor_path)
    cfg = VggtCfg()
    cfg.vggt_input_res = input_res
    cfg.vggt_agg_layers = 24
    cfg.vggt_layer_indices = "3,10,16,22"
    vggt_ctx = {
        "session": session,
        "input_name": input_name,
        "compressor": compressor,
        "cfg": cfg,
    }
    return vggt_ctx


def _prepare_single_observation(model: OctoModel, image: np.ndarray, vggt_ctx: Optional[dict] = None) -> Tuple[dict, dict]:
    # Determine window and image size expected by model
    try:
        expected_window = int(model.example_batch["observation"]["timestep_pad_mask"].shape[1])
    except Exception:
        expected_window = 1

    # Resize image to model's expected policy image size
    resize_size = get_image_resize_size({"model_family": "octo"}, model)
    try:
        from evaluation.supporting_files.robot_utils import resize_image_for_policy
        img_resized = resize_image_for_policy(image, resize_size)
    except Exception:
        from PIL import Image
        target = (resize_size, resize_size) if isinstance(resize_size, int) else (int(resize_size[1]), int(resize_size[0]))
        img_resized = np.array(Image.fromarray(image).resize(target, Image.BILINEAR))

    # Build observation dict
    # Images: (1, T, H, W, 3)
    image_stack = np.stack([img_resized for _ in range(expected_window)], axis=0)
    observation = {
        "timestep": np.arange(expected_window, dtype=np.int32)[np.newaxis, ...],
        "task_completed": np.zeros((1, expected_window, 4), dtype=bool),
        "timestep_pad_mask": np.ones((1, expected_window), dtype=bool),
        "pad_mask_dict": {
            "timestep": np.ones((1, expected_window), dtype=bool),
        },
        "image_primary": image_stack[np.newaxis, ...],
    }
    observation["pad_mask_dict"]["image_primary"] = np.ones((1, expected_window), dtype=bool)

    # Optional VGGT tokens
    if vggt_ctx is not None:
        try:
            tokens = compute_compressed_vggt_tokens(img_resized, vggt_ctx)  # (64,512)
            vggt_stack = np.stack([tokens for _ in range(expected_window)], axis=0)  # (T,64,512)
            observation["vggt_tokens"] = vggt_stack[np.newaxis, ...]  # (1,T,64,512)
            observation["pad_mask_dict"]["vggt_tokens"] = np.ones((1, expected_window), dtype=bool)
        except Exception as e:
            print(f"[WARN] Failed to compute VGGT tokens: {e}")

    # Minimal empty task (no language)
    tasks = {k: v for k, v in model.example_batch["task"].items() if k != "language_instruction"}
    if "pad_mask_dict" not in tasks:
        tasks["pad_mask_dict"] = {}
    for k in list(tasks.keys()):
        if k != "pad_mask_dict":
            tasks["pad_mask_dict"][k] = np.ones((1,) + tasks[k].shape[1:-1], dtype=bool) if isinstance(tasks[k], np.ndarray) else np.ones((1,), dtype=bool)

    return observation, tasks


def _tokens_to_similarity(obs_tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    obs_tokens: (1, T, N, D) -> returns (N,N) cosine sim and (8,8) map for central token
    """
    x = obs_tokens[0, 0]  # (N,D)
    x = x.astype(np.float32)
    # Normalize
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = x_norm @ x_norm.T  # (N,N)
    # Central patch index (8x8 grid -> index 36)
    n = x.shape[0]
    side = int(round(np.sqrt(n)))
    c = (side // 2) * side + (side // 2)
    center_map = sim[c].reshape(side, side)
    return sim, center_map


def main():
    parser = argparse.ArgumentParser(description="Compare attention-like maps between two Octo checkpoints")
    parser.add_argument("--model_path1", required=True, type=str, help="Baseline model path (e.g., PCA+ONNX trained)")
    parser.add_argument("--model_path2", required=True, type=str, help="VGGT model path (e.g., AE/PyTorch trained)")
    parser.add_argument("--vggt_onnx_path", required=True, type=str, help="Path to VGGT ONNX model for token generation")
    parser.add_argument("--vggt_compressor_path", required=True, type=str, help="Path to VGGT compressor .pkl for ONNX tokens")
    parser.add_argument("--task_suite", default="libero_spatial", type=str)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--output", default="compare_attention.png", type=str)
    args = parser.parse_args()

    set_seed_everywhere(args.seed)

    # Load both models
    model1 = OctoModel.load_pretrained(args.model_path1)
    model2 = OctoModel.load_pretrained(args.model_path2)

    # Initialize a LIBERO env and get deterministic initial state
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(0)
    env, task_description = get_libero_env(task, "octo", 256)
    print(f"Task suite: {args.task_suite} | task_id=0 | description: {task_description}")
    obs = env.get_observation()
    # Use the first rendered image as input
    img = obs["image"] if "image" in obs else obs.get("base_image") or obs.get("front_image")
    img = np.asarray(img, dtype=np.uint8)

    # Build VGGT context for token generation
    vggt_ctx = _build_vggt_ctx(args.vggt_onnx_path, args.vggt_compressor_path, input_res=518, use_cuda=True)

    # Prepare observations for both models (same image and tokens)
    obs1, tasks1 = _prepare_single_observation(model1, img, vggt_ctx)
    obs2, tasks2 = _prepare_single_observation(model2, img, vggt_ctx)

    # Run transformer once (eval mode)
    outputs1 = model1.run_transformer(obs1, tasks1, obs1["timestep_pad_mask"], train=False)
    outputs2 = model2.run_transformer(obs2, tasks2, obs2["timestep_pad_mask"], train=False)

    # Debug: list observation groups and shapes to understand N
    obs_groups1 = {k: v.tokens.shape for k, v in outputs1.items() if k.startswith("obs_")}
    obs_groups2 = {k: v.tokens.shape for k, v in outputs2.items() if k.startswith("obs_")}
    print("obs groups (model1):", obs_groups1)
    print("obs groups (model2):", obs_groups2)

    # Prefer the mixed_vision group (64 tokens) if present; else fall back to concatenated obs
    tok_group1 = outputs1.get("obs_mixed_vision", outputs1["obs"]).tokens  # (1,T,N,D)
    tok_group2 = outputs2.get("obs_mixed_vision", outputs2["obs"]).tokens
    print("N per model (tokens at t0):", int(tok_group1.shape[2]), int(tok_group2.shape[2]))

    tok1 = tok_group1
    tok2 = tok_group2

    sim1, center1 = _tokens_to_similarity(tok1)
    sim2, center2 = _tokens_to_similarity(tok2)

    # Plot: side-by-side 64x64 similarities, and overlay center-map on image
    side = center1.shape[0]
    # Upsample heatmaps to image size for overlay
    import cv2
    heat1 = cv2.resize((center1 - center1.min()) / (center1.ptp() + 1e-6), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat2 = cv2.resize((center2 - center2.min()) / (center2.ptp() + 1e-6), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(sim1, cmap="viridis")
    ax1.set_title("Model 1: token-token sim (N x N)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(sim2, cmap="viridis")
    ax2.set_title("Model 2: token-token sim (N x N)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(img)
    ax3.imshow(heat1, cmap="jet", alpha=0.5)
    ax3.set_title("Model 1: center-patch similarity overlay")
    ax3.axis("off")

    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(img)
    ax4.imshow(heat2, cmap="jet", alpha=0.5)
    ax4.set_title("Model 2: center-patch similarity overlay")
    ax4.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()

