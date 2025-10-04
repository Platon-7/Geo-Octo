import os
import sys
import math
from typing import Tuple, Optional, List, Iterable

import numpy as np
from absl import app, flags, logging
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use Agg for headless environments
import matplotlib
matplotlib.use("Agg")

# Reuse the official VGGT implementation shipped in this repo
from vggt.models.vggt import VGGT


FLAGS = flags.FLAGS

# I/O and dataset discovery
flags.DEFINE_string("input_data_dir", None, "Path to the root directory containing ORIGINAL sub-datasets.", required=True)
flags.DEFINE_string("output_dir", "/home/pkarageorgis/AE_Compressor", "Directory to save trained AE .pt files.")
flags.DEFINE_string(
    "dataset_names",
    "libero_spatial_no_noops,libero_goal_no_noops,libero_object_no_noops,libero_10_no_noops",
    "Comma-separated TFDS dataset names to sample from. If a name fails to build, it will be skipped.",
)

# VGGT/feature extraction settings
flags.DEFINE_integer("vggt_input_res", 224, "Input resolution for VGGT model (square).")
flags.DEFINE_bool("use_cuda", True, "Use CUDA if available.")
flags.DEFINE_integer("batch_size_eval", 16, "Batch size for VGGT forward when sampling tokens.")
flags.DEFINE_integer("vggt_agg_layers", 24, "Number of layers to aggregate (24 for all, or e.g., 4 for subset).")
flags.DEFINE_string("vggt_layer_indices", "3,10,16,22", "Comma-separated 0-based indices for subset (only when vggt_agg_layers < 24).")

# AE settings
flags.DEFINE_string("target_size", "64,512", "Output compressed size as 'height,width' => (n_tokens, feature_dim).")
flags.DEFINE_integer("compression_samples", 2500, "Number of spatial tokens to sample for fitting the autoencoder.")
flags.DEFINE_integer("ae_epochs", 3, "Autoencoder training epochs.")
flags.DEFINE_float("ae_lr", 1e-3, "Autoencoder learning rate.")
flags.DEFINE_integer("ae_hidden", 2048, "Autoencoder hidden dimension for MLP bottleneck.")
flags.DEFINE_bool("use_weighted_layer_fusion", True, "If True, learn softmax layer weights; else uniform mean.")


# -------------------------
# Image preprocessing (aspect-preserving to target, 14-multiple rounding, CHW, white pad)
# -------------------------

def preprocess_images_in_memory(images_np: np.ndarray, target_size: int) -> np.ndarray:
    from PIL import Image

    processed_images = []
    for img_array in images_np:
        pil_image = Image.fromarray(img_array)

        # RGBA -> composite on white
        if pil_image.mode == 'RGBA':
            background = Image.new('RGBA', pil_image.size, (255, 255, 255, 255))
            pil_image = Image.alpha_composite(background, pil_image)
        pil_image = pil_image.convert('RGB')

        # Aspect-preserving resize with rounding to nearest multiple of 14
        width, height = pil_image.size
        if width >= height:
            new_width = target_size
            new_height = int(round(height * (new_width / width) / 14) * 14)
        else:
            new_height = target_size
            new_width = int(round(width * (new_height / height) / 14) * 14)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # Normalize to [0,1] and CHW
        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))

        # Pad to square with white background
        h_padding = target_size - arr.shape[1]
        w_padding = target_size - arr.shape[2]
        pad_top = h_padding // 2
        pad_bottom = h_padding - pad_top
        pad_left = w_padding // 2
        pad_right = w_padding - pad_left
        arr = np.pad(arr, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=1.0)
        processed_images.append(arr)

    return np.stack(processed_images, axis=0)


# -------------------------
# Autoencoder compressor
# -------------------------

class WeightedLayerFuser(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., L, D]
        w = torch.softmax(self.weights, dim=0)  # [L]
        return (x * w.view(*([1] * (x.ndim - 2)), -1, 1)).sum(dim=-2)


class AECompressor(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        bottleneck_dim: int = 512,
        hidden_dim: int = 2048,
        use_weighted_layer_fusion: bool = True,
    ):
        super().__init__()
        self.use_weighted_layer_fusion = bool(use_weighted_layer_fusion)
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

    def forward(self, tokens_ld: torch.Tensor):
        if self.use_weighted_layer_fusion:
            encoder_input = self.fuser(tokens_ld)
        else:
            encoder_input = tokens_ld.mean(dim=-2)
        z = self.encoder(encoder_input)
        z = self.output_norm(z)
        recon = self.decoder(z)
        return z, recon, encoder_input

    @torch.no_grad()
    def compress_tokens(self, tokens_ld: torch.Tensor) -> torch.Tensor:
        self.eval()
        device = self.fuser.weights.device
        tokens_ld = tokens_ld.to(device)
        if self.use_weighted_layer_fusion:
            fused = self.fuser(tokens_ld)
        else:
            fused = tokens_ld.mean(dim=-2)
        z = self.encoder(fused)
        z = self.output_norm(z)
        return z

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))


def _parse_target_size(s: str) -> Tuple[int, int]:
    h, w = s.split(',')
    return int(h), int(w)


# -------------------------
# VGGT feature extraction using torch
# -------------------------

class TorchVGGTExtractor:
    def __init__(self, device: torch.device, input_res: int, agg_layers: int, layer_indices: Optional[List[int]] = None):
        self.device = device
        self.input_res = input_res
        self.agg_layers = int(agg_layers)
        self.layer_indices = layer_indices
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()

    @torch.no_grad()
    def extract_layers(self, chw_images: np.ndarray):
        x = torch.from_numpy(chw_images).to(self.device)  # [K,3,H,W]
        x = x.unsqueeze(1)  # [K,1,3,H,W]
        output_list, patch_start_idx = self.model.aggregator(x)
        all_layers = []
        for t in output_list:  # [K,1,P,2048]
            t = t[:, 0]  # [K,P,2048]
            t = t[:, patch_start_idx:, :]  # keep only patch tokens => [K,N,2048]
            all_layers.append(t)
        layers = torch.stack(all_layers, dim=0).permute(1, 0, 2, 3)  # [K,L,N,2048]

        if self.agg_layers < 24:
            idx = self.layer_indices if self.layer_indices else [3, 10, 16, 22]
            layers = layers[:, idx, :, :]

        K, L, N, D = layers.shape
        sqrt_n = int(round(np.sqrt(N)))
        return layers.detach().cpu().numpy(), sqrt_n


_logged_resized_shape = False

def resize_and_stack_per_layer(features_klnd: np.ndarray, sqrt_n: int) -> np.ndarray:
    global _logged_resized_shape
    K, L, N, D = features_klnd.shape
    s = sqrt_n
    x = torch.from_numpy(features_klnd).float()          # [K,L,N,D]
    x = x.reshape(K * L, s, s, D).permute(0, 3, 1, 2)

    target_h, _ = _parse_target_size(FLAGS.target_size)
    target_side = int(np.sqrt(target_h))

    x_small = F.interpolate(x, size=(target_side, target_side), mode='bilinear', align_corners=False)
    x_small = x_small.permute(0, 2, 3, 1).contiguous().view(K, L, target_side * target_side, D)

    if not _logged_resized_shape:
        logging.info(f"VERIFY 2: Feature shape after spatial resize is {x_small.shape} -> (Batch, Layers, N, Feat_Dim)")
        _logged_resized_shape = True
    return x_small.numpy()


# -------------------------
# Sampling utilities
# -------------------------

def _first_image_from_builder(builder) -> np.ndarray:
    ds = builder.as_dataset(split='train').take(1)
    episode = next(iter(tfds.as_numpy(ds)))
    first_step = next(iter(episode['steps']))
    return np.asarray(first_step['observation']['image'])


def _iter_images_from_builders(builders: List[tfds.core.DatasetBuilder]):
    for builder in builders:
        try:
            ds = builder.as_dataset(split='train')
        except Exception:
            continue
        for episode in tfds.as_numpy(ds):
            steps = list(episode.get('steps', []))
            if not steps:
                continue
            for step in steps:
                obs = step.get('observation', {})
                if 'image' in obs:
                    img = obs['image']
                elif 'image_primary' in obs:
                    img = obs['image_primary']
                else:
                    continue
                if img is None:
                    continue
                yield np.asarray(img)


def _sample_tokens_for_ae(
    builders: List[tfds.core.DatasetBuilder],
    extractor: TorchVGGTExtractor,
    num_tokens: int,
    batch_size: int,
    builder_splits: Optional[List[str]] = None,
    shuffle_buffer_size: int = 8192,
    yield_batch_size: int = 256,
    per_builder_token_cap: Optional[int] = None,
) -> Iterable[torch.Tensor]:
    from collections import defaultdict

    builder_counts = defaultdict(int)
    total_yielded = 0

    samples_pbar = tqdm(total=num_tokens, desc="Collecting AE samples", dynamic_ncols=True, leave=False, file=sys.stdout)
    logging.info("Sampling across %d datasets; per-dataset cap=%s", len(builders), str(per_builder_token_cap))

    for builder in builders:
        builder_yielded = 0
        buffer_tensor: Optional[torch.Tensor] = None

        available_splits = list(builder.info.splits.keys()) or ["train"]
        target_splits = builder_splits if builder_splits is not None else (["train"] if "train" in available_splits else available_splits)

        for split in target_splits:
            if split not in available_splits:
                continue
            if (per_builder_token_cap is not None and builder_yielded >= per_builder_token_cap) or total_yielded >= num_tokens:
                break

            ds = builder.as_dataset(split=split)
            episodes = tfds.as_numpy(ds)
            episodes_pbar = tqdm(episodes, desc=f"Processing {builder.name}:{split}", dynamic_ncols=True, leave=False, file=sys.stdout)

            for episode in episodes_pbar:
                if (per_builder_token_cap is not None and builder_yielded >= per_builder_token_cap) or total_yielded >= num_tokens:
                    break

                steps = list(episode.get("steps", []))
                if not steps:
                    continue
                obs = [s.get("observation", {}) for s in steps]
                if not obs or "image" not in obs[0]:
                    continue

                images_np = np.stack([o["image"] for o in obs], axis=0)
                if images_np.size == 0:
                    continue

                chw_images = preprocess_images_in_memory(images_np, FLAGS.vggt_input_res)

                for j in range(0, chw_images.shape[0], batch_size):
                    if (per_builder_token_cap is not None and builder_yielded >= per_builder_token_cap) or total_yielded >= num_tokens:
                        break

                    batch = chw_images[j:j + batch_size]
                    klnd, sqrt_n = extractor.extract_layers(batch)
                    k_l_64_d = resize_and_stack_per_layer(klnd, sqrt_n)  # [K,L,64,D]
                    K, L, S64, D = k_l_64_d.shape
                    tokens = torch.from_numpy(k_l_64_d).float().view(K * S64, L, D).cpu()

                    buffer_tensor = tokens if buffer_tensor is None else torch.cat([buffer_tensor, tokens], dim=0)
                    if buffer_tensor.shape[0] > shuffle_buffer_size:
                        perm = torch.randperm(buffer_tensor.shape[0])
                        buffer_tensor = buffer_tensor[perm[:shuffle_buffer_size]]

                    while buffer_tensor is not None and buffer_tensor.shape[0] > 0:
                        remaining_global = num_tokens - total_yielded
                        remaining_builder = (per_builder_token_cap - builder_yielded) if per_builder_token_cap is not None else buffer_tensor.shape[0]
                        n_to_yield = min(yield_batch_size, buffer_tensor.shape[0], remaining_global, remaining_builder)
                        if n_to_yield <= 0:
                            break

                        perm = torch.randperm(buffer_tensor.shape[0])
                        batch_idx = perm[:n_to_yield]
                        keep_idx = perm[n_to_yield:]
                        yield_batch = buffer_tensor[batch_idx]
                        buffer_tensor = buffer_tensor[keep_idx] if keep_idx.numel() > 0 else None

                        yield yield_batch
                        n = yield_batch.shape[0]
                        total_yielded += n
                        builder_yielded += n
                        builder_counts[builder.name] += n
                        samples_pbar.update(n)

                        if total_yielded >= num_tokens:
                            logging.info("Per-dataset token counts this epoch: %s", dict(builder_counts))
                            episodes_pbar.close()
                            samples_pbar.close()
                            return
                        if per_builder_token_cap is not None and builder_yielded >= per_builder_token_cap:
                            break

            episodes_pbar.close()
            logging.info("Dataset %s yielded %d tokens so far", builder.name, builder_yielded)
            if per_builder_token_cap is not None and builder_yielded >= per_builder_token_cap:
                break

    logging.info("Per-dataset token counts this epoch: %s", dict(builder_counts))
    samples_pbar.close()


@torch.no_grad()
def eval_ae(builders, extractor, model, num_tokens=512, batch_size=None):
    if batch_size is None:
        batch_size = FLAGS.batch_size_eval
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")
    batches = _sample_tokens_for_ae(
        builders,
        extractor,
        num_tokens=num_tokens,
        batch_size=batch_size,
        builder_splits=["train"],
        per_builder_token_cap=math.ceil(num_tokens / max(1, len(builders))),
        yield_batch_size=256,
    )
    total, count = 0.0, 0
    device = next(model.parameters()).device
    for batch in batches:
        batch = batch.to(device)
        _, recon, fused_mean = model(batch)
        total += float(loss_fn(recon, fused_mean).cpu().item()) * batch.shape[0]
        count += batch.shape[0]
    logging.info("AE eval loss (approx, %d tokens): %.6f", count, total / max(1, count))


# -------------------------
# Fit AE
# -------------------------

def _fit_autoencoder(builders: List[tfds.core.DatasetBuilder], extractor: TorchVGGTExtractor, target_size: Tuple[int, int], device: torch.device) -> AECompressor:
    target_h, target_w = target_size  # expect (64, 512)
    first_image = _first_image_from_builder(builders[0])
    chw = preprocess_images_in_memory(np.asarray([first_image]), FLAGS.vggt_input_res)
    klnd, sqrt_n = extractor.extract_layers(chw)
    L = klnd.shape[1]
    D = klnd.shape[3]

    model = AECompressor(
        num_layers=L,
        input_dim=D,
        bottleneck_dim=target_w,
        hidden_dim=FLAGS.ae_hidden,
        use_weighted_layer_fusion=FLAGS.use_weighted_layer_fusion,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=FLAGS.ae_lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(FLAGS.ae_epochs):
        num_builders = len(builders)
        tokens_per_builder = math.ceil(FLAGS.compression_samples / max(1, num_builders))

        batches = _sample_tokens_for_ae(
            builders,
            extractor,
            num_tokens=FLAGS.compression_samples,
            batch_size=FLAGS.batch_size_eval,
            builder_splits=["train"],
            per_builder_token_cap=tokens_per_builder,
            yield_batch_size=256,
        )
        seen = 0
        running = 0.0
        pbar = tqdm(total=FLAGS.compression_samples, desc=f"AE train epoch {epoch+1}/{FLAGS.ae_epochs}", leave=False, dynamic_ncols=True, file=sys.stdout)
        for batch in batches:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            z, recon, reconstruction_target = model(batch)
            loss = loss_fn(recon, reconstruction_target.detach())
            loss.backward()
            opt.step()
            seen += batch.shape[0]
            running += float(loss.detach().cpu().item()) * batch.shape[0]
            pbar.update(batch.shape[0])
            if seen >= FLAGS.compression_samples:
                break
        pbar.close()
        logging.info("AE epoch %d: loss=%.6f (N=%d)", epoch + 1, running / max(1, seen), seen)

    return model


# -------------------------
# Main
# -------------------------

def main(_):
    logging.set_verbosity(logging.INFO)

    device = torch.device('cuda' if (FLAGS.use_cuda and torch.cuda.is_available()) else 'cpu')
    logging.info("Using device: %s", device)

    # Discover builders from input root
    input_root = FLAGS.input_data_dir

    # Prefer the provided dataset_names; skip those that fail
    requested_names = [n.strip() for n in FLAGS.dataset_names.split(',') if n.strip()]
    original_builders: List[tfds.core.DatasetBuilder] = []
    for name in requested_names:
        try:
            b = tfds.builder(name, data_dir=input_root)
            # Touch info to ensure it's real
            _ = list(b.info.splits.keys())
            original_builders.append(b)
        except Exception as e:
            logging.warning("Skipping dataset %s: %s", name, e)

    if not original_builders:
        raise ValueError("No valid dataset builders found from dataset_names under input_data_dir")

    for b in original_builders:
        logging.info("Detected dataset: %s | splits=%s | data_dir=%s", b.name, list(b.info.splits.keys()), getattr(b, "data_dir", None))

    # Prepare VGGT extractor (24 layers or subset)
    if FLAGS.vggt_agg_layers < 24:
        try:
            layer_indices = [int(x) for x in FLAGS.vggt_layer_indices.split(',') if x.strip() != '']
        except Exception:
            layer_indices = [3, 10, 16, 22]
    else:
        layer_indices = None
    extractor = TorchVGGTExtractor(device, FLAGS.vggt_input_res, FLAGS.vggt_agg_layers, layer_indices)

    # Prepare / train AE compressor
    target_size = _parse_target_size(FLAGS.target_size)

    logging.info("Fitting autoencoder compressor (samples=%d, epochs=%d)...", FLAGS.compression_samples, FLAGS.ae_epochs)
    compressor = _fit_autoencoder(original_builders, extractor, target_size, device)
    eval_ae(original_builders, extractor, compressor, num_tokens=512)

    # Save AE
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    ae_path = os.path.join(FLAGS.output_dir, f"vggt_autoencoder_{FLAGS.vggt_agg_layers}L_{target_size[0]}x{target_size[1]}.pt")
    compressor.cpu().save(ae_path)
    logging.info("Saved AE compressor to %s", ae_path)


if __name__ == "__main__":
    app.run(main)
