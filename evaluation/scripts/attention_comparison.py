import sys
import warnings
import json
# Add compatibility shim before importing anything else
try:
    import jax.numpy as jnp
    if not hasattr(jnp, 'DeviceArray'):
        jnp.DeviceArray = jnp.ndarray
        print("[FIX] Added DeviceArray compatibility shim")
except ImportError:
    print("[WARNING] Could not import JAX")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")


import argparse
import os
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Octo
from octo.model.octo_model import OctoModel

# LIBERO env helpers
from libero.libero import benchmark
from evaluation.supporting_files.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
)
from evaluation.supporting_files.robot_utils import (
    set_seed_everywhere,
    get_image_resize_size,
)

# Torch + VGGT (AE)
import torch
import torch.nn as nn
import torch.nn.functional as F
from vggt.models.vggt import VGGT


def _prepare_single_observation(model: OctoModel, image: np.ndarray) -> dict:
    # Determine window and image size expected by model
    try:
        expected_window = int(model.example_batch["observation"]["timestep_pad_mask"].shape[1])
    except Exception:
        expected_window = 1

    # Resize image to model’s expected policy image size
    resize_size = get_image_resize_size({"model_family": "octo"}, model)
    try:
        from evaluation.supporting_files.robot_utils import resize_image_for_policy
        img_resized = resize_image_for_policy(image, resize_size)
    except Exception:
        from PIL import Image
        target = (resize_size, resize_size) if isinstance(resize_size, int) else (int(resize_size[1]), int(resize_size[0]))
        img_resized = np.array(Image.fromarray(image).resize(target, Image.BILINEAR))

    image_stack = np.stack([img_resized for _ in range(expected_window)], axis=0)
    observation = {
        "timestep": np.arange(expected_window, dtype=np.int32)[np.newaxis, ...],
        "task_completed": np.zeros((1, expected_window, 4), dtype=bool),
        "timestep_pad_mask": np.ones((1, expected_window), dtype=bool),
        "pad_mask_dict": {"timestep": np.ones((1, expected_window), dtype=bool)},
        "image_primary": image_stack[np.newaxis, ...],
    }
    observation["pad_mask_dict"]["image_primary"] = np.ones((1, expected_window), dtype=bool)
    return observation


class WeightedLayerFuser(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        return (x * w.view(*([1] * (x.ndim - 2)), -1, 1)).sum(dim=-2)


class AECompressor(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, bottleneck_dim: int = 512, hidden_dim: int = 2048):
        super().__init__()
        self.fuser = WeightedLayerFuser(num_layers)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.output_norm = nn.LayerNorm(bottleneck_dim)

    @torch.no_grad()
    def compress_tokens(self, tokens_ld: torch.Tensor) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        tokens_ld = tokens_ld.to(device)
        fused = self.fuser(tokens_ld)          # [B,D]
        z = self.encoder(fused)                # [B,512]
        z = self.output_norm(z)                # normalize bottleneck to stabilize scale
        return z


class TorchVGGTExtractor:
    def __init__(self, device: torch.device, input_res: int, agg_layers: int, layer_indices: Optional[List[int]] = None):
        self.device = device
        self.input_res = input_res
        self.agg_layers = int(agg_layers)
        self.layer_indices = layer_indices
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()

    @torch.no_grad()
    def extract_layers(self, chw_images: np.ndarray) -> Tuple[np.ndarray, int]:
        x = torch.from_numpy(chw_images).to(self.device)  # [K,3,H,W]
        x = x.unsqueeze(1)                                # [K,1,3,H,W]
        output_list, patch_start_idx = self.model.aggregator(x)
        all_layers = []
        for t in output_list:                             # [K,1,P,2048]
            t = t[:, 0]                                   # [K,P,2048]
            t = t[:, patch_start_idx:, :]                 # keep only patch tokens
            all_layers.append(t)
        layers = torch.stack(all_layers, dim=0).permute(1, 0, 2, 3)  # [K,L,N,2048]
        if self.agg_layers < 24:
            idx = self.layer_indices if (self.layer_indices and len(self.layer_indices) > 0) else [3, 10, 16, 22]
            layers = layers[:, idx, :, :]
        K, L, N, D = layers.shape
        sqrt_n = int(round(np.sqrt(N)))
        return layers.detach().cpu().numpy(), sqrt_n


def _resize_and_stack_per_layer(features_klnd: np.ndarray, sqrt_n: int, target_side: int) -> np.ndarray:
    K, L, N, D = features_klnd.shape
    s = sqrt_n
    x = torch.from_numpy(features_klnd).float()      # [K,L,N,D]
    x = x.reshape(K * L, s, s, D).permute(0, 3, 1, 2)
    x_small = F.interpolate(x, size=(target_side, target_side), mode='bilinear', align_corners=False)
    x_small = x_small.permute(0, 2, 3, 1).contiguous().view(K, L, target_side * target_side, D)
    return x_small.numpy()


@torch.no_grad()
def _compute_vggt_tokens_ae(image: np.ndarray, extractor: TorchVGGTExtractor, compressor: AECompressor,
                            vggt_input_res: int, target_side: int = 16) -> np.ndarray:
    # Preprocess to CHW [0,1]
    from evaluation.supporting_files.load_fn import load_and_preprocess_images
    pre = load_and_preprocess_images([image], target_size=vggt_input_res)      # [1,3,H,W]
    klnd, sqrt_n = extractor.extract_layers(pre)                               # [K,L,N,D]
    k_l_s_d = _resize_and_stack_per_layer(klnd, sqrt_n, target_side)           # [K,L,256,D]
    K, L, S, D = k_l_s_d.shape
    tokens = torch.from_numpy(k_l_s_d).float().view(K * S, L, D).to(next(compressor.parameters()).device)
    z = compressor.compress_tokens(tokens)                                      # [K*S,512]
    z = z.view(K, S, -1).cpu().numpy().astype(np.float16)                       # [K,256,512]
    return z[0]


def _tokens_to_similarity(obs_tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(obs_tokens)[0, 0]         # (N,D)
    x = x.astype(np.float32)
    n = x.shape[0]
    side = int(np.floor(np.sqrt(n)))
    N_sq = side * side
    x_sq = x[:N_sq].reshape(N_sq, -1)

    x_norm = x_sq / (np.linalg.norm(x_sq, axis=1, keepdims=True) + 1e-8)
    sim = x_norm @ x_norm.T
    c = (side // 2) * side + (side // 2)
    center_map = sim[c].reshape(side, side)
    return sim.astype(np.float32), center_map.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs VGGT+AE (Torch) token-token similarity")
    parser.add_argument("--model_path1", required=True, type=str, help="Baseline Octo model path")
    parser.add_argument("--model_path2", required=True, type=str, help="VGGT+AE Octo model path")
    parser.add_argument("--vggt_ae_path", required=True, type=str, help="AE .pt path (e.g., vggt_autoencoder_24L_256x512.pt)")
    parser.add_argument("--vggt_input_res", type=int, default=518)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--task_suite", default="libero_spatial", type=str)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--output", default="compare_attention.png", type=str)
    args = parser.parse_args()

    set_seed_everywhere(args.seed)

    # Load models
    baseline = OctoModel.load_pretrained(args.model_path1)
    vggt_model = OctoModel.load_pretrained(args.model_path2)

    # Env and image
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = task_suite.get_task(0)
    env, task_description = get_libero_env(task, "octo", 256)
    print(f"Task: {task_description}")

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    for _ in range(10):
        obs, _, done, _ = env.step(get_libero_dummy_action("octo"))
        if done:
            obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
    img = get_libero_image(obs)

    # Observations
    obs1 = _prepare_single_observation(baseline, img)
    obs2 = _prepare_single_observation(vggt_model, img)

    # Tasks with language to match language_conditioned finetune
    tasks1 = baseline.create_tasks(texts=[task_description])
    tasks2 = vggt_model.create_tasks(texts=[task_description])

    # VGGT tokens via Torch AE for model 2 only
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    extractor = TorchVGGTExtractor(device, args.vggt_input_res, 24, None)
    # Probe L, D
    dummy = np.ones((1, 3, args.vggt_input_res, args.vggt_input_res), dtype=np.float32)
    klnd, _ = extractor.extract_layers(dummy)
    L = klnd.shape[1]
    D = klnd.shape[3]
    compressor = AECompressor(num_layers=L, input_dim=D, bottleneck_dim=512, hidden_dim=2048).to(device).eval()
    compressor.load_state_dict(torch.load(args.vggt_ae_path, map_location="cpu"))
    compressor = compressor.to(device).eval()

    tokens_256_512 = _compute_vggt_tokens_ae(img, extractor, compressor, vggt_input_res=args.vggt_input_res, target_side=16)
    obs2["vggt_tokens"] = np.stack([tokens_256_512 for _ in range(obs2["timestep_pad_mask"].shape[1])], axis=0)[np.newaxis, ...]
    obs2["pad_mask_dict"]["vggt_tokens"] = np.ones_like(obs2["timestep_pad_mask"], dtype=bool)

    # Forward once
    outs1 = baseline.run_transformer(obs1, tasks1, obs1["timestep_pad_mask"], train=False)
    outs2 = vggt_model.run_transformer(obs2, tasks2, obs2["timestep_pad_mask"], train=False)

    # Prefer mixed_vision if present (64 or 256 tokens depending on your setup)
    tok1 = outs1.get("obs_mixed_vision", outs1["obs"]).tokens
    tok2 = outs2.get("obs_mixed_vision", outs2["obs"]).tokens
    print("obs groups (baseline):", {k: v.tokens.shape for k, v in outs1.items() if k.startswith("obs_")})
    print("obs groups (vggt):    ", {k: v.tokens.shape for k, v in outs2.items() if k.startswith("obs_")})
    print("N per model:", int(tok1.shape[2]), int(tok2.shape[2]))

    # Token-token similarity and central patch heatmaps
    sim1, center1 = _tokens_to_similarity(tok1)
    sim2, center2 = _tokens_to_similarity(tok2)

    # Upsample heatmaps to image size for overlay
    try:
        import cv2
        def up(h):
            h = (h - h.min()) / (h.ptp() + 1e-6)
            h = np.asarray(h, dtype=np.float32).copy(order="C")
            return cv2.resize(h, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        heat1 = up(center1)
        heat2 = up(center2)
    except Exception:
        from PIL import Image
        def up(h):
            h = (h - h.min()) / (h.ptp() + 1e-6)
            return np.array(Image.fromarray((h * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.BICUBIC)) / 255.0
        heat1 = up(center1)
        heat2 = up(center2)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1); im1 = ax1.imshow(sim1, cmap="viridis")
    ax1.set_title("Baseline: token-token sim (N×N)"); plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax2 = plt.subplot(2, 2, 2); im2 = ax2.imshow(sim2, cmap="viridis")
    ax2.set_title("VGGT+AE: token-token sim (N×N)"); plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax3 = plt.subplot(2, 2, 3); ax3.imshow(img); ax3.imshow(heat1, cmap="jet", alpha=0.5)
    ax3.set_title("Baseline: center-patch similarity"); ax3.axis("off")
    ax4 = plt.subplot(2, 2, 4); ax4.imshow(img); ax4.imshow(heat2, cmap="jet", alpha=0.5)
    ax4.set_title("VGGT+AE: center-patch similarity"); ax4.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()