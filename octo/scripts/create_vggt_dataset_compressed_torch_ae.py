import os
from typing import Tuple, Optional, List, Iterable

import numpy as np
from absl import app, flags, logging
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the official VGGT implementation shipped in this repo
from vggt.models.vggt import VGGT


# -------------------------
# Flags (largely mirrors the ONNX script, but runs PyTorch VGGT + hooks)
# -------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string("input_data_dir", None, "Path to the root directory containing ORIGINAL sub-datasets.", required=True)
flags.DEFINE_string("output_data_dir", None, "Path where the NEW compressed TFDS datasets will be written.", required=True)
flags.DEFINE_integer("vggt_input_res", 224, "Input resolution for VGGT model (square).")
flags.DEFINE_bool("use_cuda", True, "Use CUDA if available.")
flags.DEFINE_integer("batch_size_eval", 16, "Evaluation batch size (images per forward).")

# Layer aggregation identical to ONNX path
flags.DEFINE_integer("vggt_agg_layers", 24, "Number of layers to aggregate (24 for all, or e.g., 4 for subset).")
flags.DEFINE_string("vggt_layer_indices", "3,10,16,22", "Comma-separated 0-based indices for subset (only when vggt_agg_layers < 24).")

# Autoencoder compressor options
flags.DEFINE_string("ae_path", None, "Path to save/load the autoencoder compressor (.pt). If not found and train_if_missing is True, a new one is trained.")
flags.DEFINE_bool("train_if_missing", True, "If True and ae_path doesn't exist, fit a new autoencoder.")
flags.DEFINE_integer("compression_samples", 2500, "Number of spatial tokens to sample for fitting the autoencoder.")
flags.DEFINE_string("target_size", "64,512", "Output compressed size as 'height,width' => (n_tokens, feature_dim).")
flags.DEFINE_integer("ae_epochs", 3, "Autoencoder training epochs (lightweight).")
flags.DEFINE_float("ae_lr", 1e-3, "Autoencoder learning rate.")
flags.DEFINE_integer("ae_hidden", 2048, "Autoencoder hidden dimension for MLP bottleneck.")
flags.DEFINE_bool("overwrite", False, "If True, force-retrains the autoencoder and overwrites any existing output dataset.")
flags.DEFINE_bool("use_weighted_layer_fusion", True, "If True, learn softmax layer weights; if False, use uniform mean across layers.")
flags.DEFINE_bool("pointmap_viz_enable", True, "If True, generate a few VGGT pointmap visualizations before writing the dataset.")
flags.DEFINE_integer("pointmap_viz_count", 8, "Number of images to visualize with VGGT pointmap head before writing dataset.")
flags.DEFINE_string("pointmap_viz_dir", None, "Directory to save pointmap visualizations (defaults under output_data_dir/pointmap_viz).")
flags.DEFINE_bool("ae_verification_enable", True, "If True, run a visual AE verification before creating the dataset.")
flags.DEFINE_integer("ae_verification_count", 4, "Number of images to use for the AE verification.")
flags.DEFINE_string("ae_verification_dir", None, "Directory to save AE verification plots (defaults under output_data_dir/ae_verification).")


# -------------------------
# Image preprocessing (identical to ONNX script)
# -------------------------
def preprocess_images_in_memory(images_np: np.ndarray, target_size: int) -> np.ndarray:
    """
    Mirrors evaluation preprocessing (RGBA-on-white, aspect-preserving resize, 14-multiple rounding,
    bilinear resample, CHW, white padding to a square of target_size).
    """
    from PIL import Image

    processed_images = []
    for img_array in images_np:
        pil_image = Image.fromarray(img_array)

        # 1) RGBA -> composite on white
        if pil_image.mode == 'RGBA':
            background = Image.new('RGBA', pil_image.size, (255, 255, 255, 255))
            pil_image = Image.alpha_composite(background, pil_image)
        pil_image = pil_image.convert('RGB')

        # 2) Aspect-preserving resize with rounding to nearest multiple of 14
        width, height = pil_image.size
        if width >= height:
            new_width = target_size
            new_height = int(round(height * (new_width / width) / 14) * 14)
        else:
            new_height = target_size
            new_width = int(round(width * (new_height / height) / 14) * 14)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # 3) Normalize to [0,1] and transpose to CHW
        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))

        # 4) Pad to square with white background
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
    """Learnable weighted fusion across the layer dimension.

    Given tokens of shape [..., L, D], applies softmax-normalized weights over L to produce [..., D].
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))  # start ~uniform after softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., L, D]
        w = torch.softmax(self.weights, dim=0)  # [L]
        return (x * w.view(*([1] * (x.ndim - 2)), -1, 1)).sum(dim=-2)


class AECompressor(nn.Module):
    """
    Weighted fusion (over L) -> AE compression D(=2048) -> 512.

    - fuser: learnable softmax weights over layers
    - encoder/decoder: per-token autoencoder operating on D-d fused vectors
    """

    def __init__(self, num_layers: int, input_dim: int, bottleneck_dim: int = 512, hidden_dim: int = 2048, use_weighted_layer_fusion: bool = True):
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

    def forward(self, tokens_ld: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens_ld: [B, L, D] token set for B spatial positions
        Returns:
            z: [B, 512]
            recon: [B, D]
            fused_mean: [B, D] (uniform mean over layers, used as stable target)
        """
        if self.use_weighted_layer_fusion:
            fused_weighted = self.fuser(tokens_ld)
        else:
            fused_weighted = tokens_ld.mean(dim=-2)
        fused_mean = tokens_ld.mean(dim=-2)
        
        # Add logging here, only during training and only once
        if self.training and not hasattr(self, '_logged_encoder_input_shape'):
            logging.info(f"VERIFY 3: Input shape to AE encoder is {fused_weighted.shape} -> (Batch_of_Tokens, Fused_Feat_Dim)")
            self._logged_encoder_input_shape = True
            
        z = self.encoder(fused_weighted)
        z = self.output_norm(z)
        recon = self.decoder(z)
        return z, recon, fused_mean

    @torch.no_grad()
    def compress_tokens(self, tokens_ld: torch.Tensor) -> torch.Tensor:
        self.eval()
        device = self.fuser.weights.device
        tokens_ld = tokens_ld.to(device)
        if self.use_weighted_layer_fusion:
            fused_weighted = self.fuser(tokens_ld)
        else:
            fused_weighted = tokens_ld.mean(dim=-2)
        z = self.encoder(fused_weighted)
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
    def extract_layers(self, chw_images: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Args:
            chw_images: [K, 3, H, W] float32 in [0,1]

        Returns:
            features: np.ndarray with shape [K, L, N, D] (D=2048), N=sqrt_n^2
            sqrt_n: int, so that N = sqrt_n*sqrt_n
        """
        x = torch.from_numpy(chw_images).to(self.device)  # [K,3,H,W]
        x = x.unsqueeze(1)  # [K,1,3,H,W]
        output_list, patch_start_idx = self.model.aggregator(x)
        # output_list length == 24, each [K, 1, P, 2C]
        all_layers = []
        for t in output_list:  # [K,1,P,2048]
            t = t[:, 0]  # [K,P,2048]
            t = t[:, patch_start_idx:, :]  # keep only patch tokens => [K,N,2048]
            all_layers.append(t)
        # Stack along layer dimension => [L,K,N,2048] -> transpose to [K,L,N,2048]
        layers = torch.stack(all_layers, dim=0).permute(1, 0, 2, 3)

        if self.agg_layers < 24:
            idx = self.layer_indices
            if not idx:
                idx = [3, 10, 16, 22]
            layers = layers[:, idx, :, :]
            
        # Add logging here
        if not hasattr(self, '_logged_initial_shape'):
            logging.info(f"VERIFY 1: Initial VGGT feature shape is {layers.shape} -> (Batch, Layers, Num_Patches, Feat_Dim)")
            self._logged_initial_shape = True

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

    # Use target_size to determine spatial grid (e.g., 64 -> 8x8, 256 -> 16x16)
    target_h, _ = _parse_target_size(FLAGS.target_size)
    target_side = int(np.sqrt(target_h))
    # print(f"INTERPOLATION_DEBUG: Resizing spatial dimensions from {s}x{s} (N={N}) "
    #       f"to {target_side}x{target_side} (N={target_h}).")

    x_small = F.interpolate(x, size=(target_side, target_side), mode='bilinear', align_corners=False)
    x_small = x_small.permute(0, 2, 3, 1).contiguous().view(K, L, target_side * target_side, D)

    if not _logged_resized_shape:
        logging.info(f"VERIFY 2: Feature shape after spatial resize is {x_small.shape} -> (Batch, Layers, N, Feat_Dim)")
        _logged_resized_shape = True
    return x_small.numpy()


def _iter_images_from_builders(builders: List[tfds.core.DatasetBuilder]):
    """Yield raw RGB images [H,W,3] from the provided builders (train split only)."""
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


@torch.no_grad()
def visualize_pointmaps(builders: List[tfds.core.DatasetBuilder], extractor: TorchVGGTExtractor, max_images: int, out_dir: str, input_res: int):
    os.makedirs(out_dir, exist_ok=True)
    device = extractor.device
    saved = 0
    for img in _iter_images_from_builders(builders):
        try:
            chw = preprocess_images_in_memory(np.asarray([img]), input_res)  # [1,3,H,W]
            x = torch.from_numpy(chw).to(device)
            x = x.unsqueeze(1)  # [1,1,3,H,W]
            preds = extractor.model(x)

            # Prefer point head confidence; fallback to depth or its conf
            conf = None
            if isinstance(preds, dict) and 'world_points_conf' in preds:
                conf = preds['world_points_conf'][0, 0].detach().cpu().numpy()
            elif 'depth' in preds:
                depth = preds['depth'][0, 0, ..., 0].detach().cpu().numpy()
                # normalize depth to [0,1]
                mn, mx = float(np.nanmin(depth)), float(np.nanmax(depth))
                conf = (depth - mn) / max(1e-6, (mx - mn))
            elif 'depth_conf' in preds:
                conf = preds['depth_conf'][0, 0].detach().cpu().numpy()
            else:
                # As last resort, visualize norm of world_points if available
                if 'world_points' in preds:
                    pts = preds['world_points'][0, 0].detach().cpu().numpy()  # [H,W,3]
                    conf = np.linalg.norm(pts, axis=-1)
                    mn, mx = float(np.nanmin(conf)), float(np.nanmax(conf))
                    conf = (conf - mn) / max(1e-6, (mx - mn))
                else:
                    continue

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(img)
            axs[0].set_title('Original')
            axs[0].axis('off')
            im = axs[1].imshow(conf, cmap='viridis')
            axs[1].set_title('VGGT pointmap/conf')
            axs[1].axis('off')
            fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
            out_path = os.path.join(out_dir, f'pointmap_{saved:03d}.png')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            saved += 1
            if saved >= max_images:
                break
        except Exception as _e:
            continue
    logging.info("Saved %d pointmap visualizations to %s", saved, out_dir)


@torch.no_grad()
def verify_autoencoder_reconstruction(
    builders: List[tfds.core.DatasetBuilder],
    extractor: TorchVGGTExtractor,
    compressor: AECompressor,
    num_samples: int,
    output_dir: str,
    device: torch.device,
):
    """
    Visual and numerical check of the AE reconstruction vs. the intended target (uniform mean across layers).
    """
    from itertools import islice
    from sklearn.metrics.pairwise import cosine_similarity
    import torch.nn.functional as F

    logging.info("--- Starting Autoencoder Verification Step ---")
    os.makedirs(output_dir, exist_ok=True)

    image_iterator = _iter_images_from_builders(builders)
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("idx\tMSE\tmean_cos\n")

    for i, original_image_np in enumerate(islice(image_iterator, num_samples)):
        try:
            # 1) Extract tokens and build ground-truth fused (uniform mean over layers)
            chw_image = preprocess_images_in_memory(np.asarray([original_image_np]), FLAGS.vggt_input_res)
            klnd, sqrt_n = extractor.extract_layers(chw_image)  # [1,L,N,D]
            k_l_t_d = resize_and_stack_per_layer(klnd, sqrt_n)  # [1,L,T,D]
            K, L, T, D = k_l_t_d.shape
            gt_fused = torch.from_numpy(k_l_t_d).float().to(device).mean(dim=1)  # [1,T,D]

            # 2) Compression and reconstruction
            tokens_to_compress = torch.from_numpy(k_l_t_d).float().view(K * T, L, D).to(device)
            z = compressor.compress_tokens(tokens_to_compress)  # [T,512]
            recon = compressor.decoder(z).view(K, T, D)  # [1,T,D]

            # 3) Metrics: MSE and cosine
            mse_loss = float(F.mse_loss(recon, gt_fused).detach().cpu().item())
            gt_np = gt_fused.squeeze(0).detach().cpu().numpy()
            recon_np = recon.squeeze(0).detach().cpu().numpy()
            # mean cosine across tokens
            eps = 1e-8
            a = gt_np / (np.linalg.norm(gt_np, axis=1, keepdims=True) + eps)
            b = recon_np / (np.linalg.norm(recon_np, axis=1, keepdims=True) + eps)
            mean_cos = float((a * b).sum(axis=1).mean())

            with open(metrics_path, "a") as f:
                f.write(f"{i}\t{mse_loss:.6f}\t{mean_cos:.6f}\n")

            # 4) Visualization: original + similarity maps
            gt_sim = cosine_similarity(gt_np)
            recon_sim = cosine_similarity(recon_np)

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            axs[0].imshow(original_image_np)
            axs[0].set_title("Original Image")
            axs[0].axis('off')
            im1 = axs[1].imshow(gt_sim)
            axs[1].set_title("GT Similarity (mean over layers)")
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            im2 = axs[2].imshow(recon_sim)
            axs[2].set_title(f"Recon Similarity (MSE: {mse_loss:.4f}, cos: {mean_cos:.5f})")
            fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
            out_path = os.path.join(output_dir, f"ae_verification_{i:02d}.png")
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            logging.info("Saved AE verification plot to %s", out_path)
        except Exception as e:
            logging.error("Failed to verify AE on sample %d: %s", i, e, exc_info=True)

    logging.info("--- Autoencoder Verification Finished ---")


# -------------------------
# TFDS Builder using the AE compressor
# -------------------------
def transpose_list_of_dicts(list_of_dicts):
    from collections import defaultdict
    transposed = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            transposed[k].append(v)
    final = {}
    for k, val_list in transposed.items():
        if isinstance(val_list[0], dict):
            final[k] = transpose_list_of_dicts(val_list)
        else:
            try:
                final[k] = np.stack(val_list)
            except Exception:
                final[k] = val_list
    return final


def _first_image_from_builder(builder) -> np.ndarray:
    ds = builder.as_dataset(split='train').take(1)
    episode = next(iter(tfds.as_numpy(ds)))          # episode is a dict of NumPy arrays / iterables
    first_step = next(iter(episode['steps']))         # <- iterate directly, DO NOT wrap again
    return np.asarray(first_step['observation']['image'])


class CompressedVggtDatasetTorch(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    def __init__(self, original_builder, extractor: TorchVGGTExtractor, compressor: AECompressor, target_size: Tuple[int, int], **kwargs):
        self._original_builder = original_builder
        self._extractor = extractor
        self._compressor = compressor
        self._target_size = target_size  # (64, 512)
        self.name = f"{self._original_builder.name}_vggt_compressed_torch"
        super().__init__(**kwargs)

    def _info(self):
        original_info = self._original_builder.info
        step_features = dict(original_info.features['steps'].feature)
        observation_features = dict(step_features['observation'])
        target_h, target_w = self._target_size
        observation_features['vggt_tokens'] = tfds.features.Tensor(
            shape=(target_h, target_w), dtype=np.float16,
            doc=f'VGGT tokens resized to {target_h} and compressed to {target_w} via AE.')
        step_features['observation'] = tfds.features.FeaturesDict(observation_features)
        final_features = tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset(tfds.features.FeaturesDict(step_features)),
            'episode_metadata': original_info.features['episode_metadata']})
        return tfds.core.DatasetInfo(builder=self,
            description=f"Libero dataset with VGGT tokens (torch VGGT, AE compression to {target_h}x{target_w}).",
            features=final_features)

    def _split_generators(self, dl_manager):
        return {'train': self._generate_examples(split='train')}

    def _generate_examples(self, split: str):
        ds = self._original_builder.as_dataset(split=split)
        num_episodes = self._original_builder.info.splits[split].num_examples
        batch_size = FLAGS.batch_size_eval
        i = 0
        for episode in tfds.as_numpy(ds):
            steps_list_of_dicts = list(episode['steps'])
            if not steps_list_of_dicts:
                i += 1; continue
            steps = transpose_list_of_dicts(steps_list_of_dicts)
            if 'image' not in steps['observation'] or len(steps['observation']['image']) == 0:
                i += 1; continue

            images_np = steps['observation']['image']  # [T,H,W,3]
            # Preprocess to CHW
            chw_images = preprocess_images_in_memory(images_np, FLAGS.vggt_input_res)  # [T,3,H,W]

            per_image_tokens = []
            # Process in batches
            for j in range(0, chw_images.shape[0], batch_size):
                batch = chw_images[j:j+batch_size]
                klnd, sqrt_n = self._extractor.extract_layers(batch)  # [K,L,N,D]
                k_l_64_d = resize_and_stack_per_layer(klnd, sqrt_n)   # [K,L,64,D]

                # AE compression per spatial location: flatten L*D -> 512, then reshape to [K,64,512]
                K, L, S64, D = k_l_64_d.shape
                # Prepare [K*64, L, D]
                tokens = torch.from_numpy(k_l_64_d).float().view(K * S64, L, D)
                with torch.no_grad():
                    z = self._compressor.compress_tokens(tokens)
                z = z.view(K, S64, -1).cpu().numpy().astype(np.float16)
                per_image_tokens.append(z)

            if per_image_tokens:
                tokens_array = np.concatenate(per_image_tokens, axis=0)  # [T,64,512]
                if len(tokens_array) != len(steps_list_of_dicts):
                    logging.warning("Token/step mismatch in episode %d. Skipping.", i)
                    i += 1; continue
                for t in range(len(tokens_array)):
                    steps_list_of_dicts[t]['observation']['vggt_tokens'] = tokens_array[t]
                yield i, {'steps': steps_list_of_dicts, 'episode_metadata': episode['episode_metadata']}
            i += 1


def _sample_tokens_for_ae(
    builders: List[tfds.core.DatasetBuilder],
    extractor: TorchVGGTExtractor,
    num_tokens: int,
    batch_size: int,
    builder_splits: Optional[List[str]] = None,   # e.g., ['train']; None => prefer 'train' if present
    shuffle_buffer_size: int = 8192,
    yield_batch_size: int = 1024,
    per_builder_token_cap: Optional[int] = None,  # balance per DATASET (builder)
) -> Iterable[torch.Tensor]:
    """
    Stream tokens across builders (datasets), balancing roughly per builder.
    Uses a bounded shuffle buffer and yields batches as soon as available.
    """
    from collections import defaultdict

    builder_counts = defaultdict(int)
    total_yielded = 0

    samples_pbar = tqdm(
        total=num_tokens,
        desc="Collecting AE samples",
        dynamic_ncols=True,
        leave=False,
        file=sys.stdout,
        disable=False,
    )
    logging.info("Sampling across %d datasets; per-dataset cap=%s",
                 len(builders), str(per_builder_token_cap))

    for builder in builders:
        builder_yielded = 0
        buffer_tensor: Optional[torch.Tensor] = None  # reset buffer per dataset

        available_splits = list(builder.info.splits.keys()) or ["train"]
        # Prefer 'train' if present; else fall back to all available
        target_splits = (
            builder_splits
            if builder_splits is not None
            else (["train"] if "train" in available_splits else available_splits)
        )

        for split in target_splits:
            if split not in available_splits:
                continue
            if (per_builder_token_cap is not None and builder_yielded >= per_builder_token_cap) or total_yielded >= num_tokens:
                break

            ds = builder.as_dataset(split=split)
            episodes = tfds.as_numpy(ds)
            episodes_pbar = tqdm(
                episodes,
                desc=f"Processing {builder.name}:{split}",
                dynamic_ncols=True,
                leave=False,
                file=sys.stdout,
                disable=False,
            )

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

                    # Append to buffer; cap size to keep memory bounded
                    buffer_tensor = tokens if buffer_tensor is None else torch.cat([buffer_tensor, tokens], dim=0)
                    if buffer_tensor.shape[0] > shuffle_buffer_size:
                        perm = torch.randperm(buffer_tensor.shape[0])
                        buffer_tensor = buffer_tensor[perm[:shuffle_buffer_size]]

                    # Yield respecting global and per-dataset caps
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
        batch_size = FLAGS.batch_size_eval  # safe now (resolved at call time)
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


def _fit_autoencoder(builders: List[tfds.core.DatasetBuilder], extractor: TorchVGGTExtractor, target_size: Tuple[int, int], device: torch.device) -> AECompressor:
    target_h, target_w = target_size  # expect (64, 512)
    # Determine L and D by running a tiny probe
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
            builder_splits=["train"],          # only train from each dataset
            per_builder_token_cap=tokens_per_builder,
            yield_batch_size=256,              # tighter balance than 1024
        )
        seen = 0
        running = 0.0
        pbar = tqdm(total=FLAGS.compression_samples,
                    desc=f"AE train epoch {epoch+1}/{FLAGS.ae_epochs}",
                    leave=False,
                    dynamic_ncols=True,
                    file=sys.stdout,
                    disable=False)
        for batch in batches:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            z, recon, fused_mean = model(batch)
            loss = loss_fn(recon, fused_mean.detach())
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
    output_root = FLAGS.output_data_dir
    # Replace your automatic discovery block with this:
    dataset_names = ["libero_spatial_no_noops", "libero_goal_no_noops", "libero_object_no_noops", "libero_10_no_noops"]
    original_builders = []
    for name in dataset_names:
        try:
            b = tfds.builder(name, data_dir=input_root)
            original_builders.append(b)
        except Exception as e:
            logging.warning("Skipping dataset %s: %s", name, e)

    for b in original_builders:
        logging.info("Detected dataset: %s | splits=%s | data_dir=%s",
                    b.name, list(b.info.splits.keys()), getattr(b, "data_dir", None))

    # Prepare VGGT extractor (24 layers or subset)
    if FLAGS.vggt_agg_layers < 24:
        try:
            layer_indices = [int(x) for x in FLAGS.vggt_layer_indices.split(',') if x.strip() != '']
        except Exception:
            layer_indices = [3, 10, 16, 22]
    else:
        layer_indices = None
    extractor = TorchVGGTExtractor(device, FLAGS.vggt_input_res, FLAGS.vggt_agg_layers, layer_indices)

    # Optional: pre-build visual sanity via pointmap head
    if FLAGS.pointmap_viz_enable:
        viz_dir = FLAGS.pointmap_viz_dir or os.path.join(output_root, "pointmap_viz")
        try:
            visualize_pointmaps(original_builders, extractor, max_images=int(FLAGS.pointmap_viz_count), out_dir=viz_dir, input_res=FLAGS.vggt_input_res)
        except Exception as e:
            logging.warning("Pointmap visualization failed: %s", e)

    # Prepare / train AE compressor
    target_size = _parse_target_size(FLAGS.target_size)
    ae_path = FLAGS.ae_path or os.path.join(output_root, f"vggt_autoencoder_{FLAGS.vggt_agg_layers}L_{target_size[0]}x{target_size[1]}.pt")

    # Determine if we need to train a new autoencoder
    should_train_ae = FLAGS.overwrite or (FLAGS.train_if_missing and not os.path.exists(ae_path))
    
    if should_train_ae:
        if FLAGS.overwrite and os.path.exists(ae_path):
            logging.warning(f"--overwrite is True. Deleting existing autoencoder at {ae_path} and retraining.")
        
        logging.info("Fitting autoencoder compressor (samples=%d, epochs=%d)...", FLAGS.compression_samples, FLAGS.ae_epochs)
        compressor = _fit_autoencoder(original_builders, extractor, target_size, device)
        eval_ae(original_builders, extractor, compressor, num_tokens=512)  # or pass batch_size=FLAGS.batch_size_eval
        os.makedirs(os.path.dirname(ae_path), exist_ok=True)
        compressor.cpu().save(ae_path)
        logging.info("Saved AE compressor to %s", ae_path)
        compressor = compressor.to(device).eval()
    else:
        # If we are not training, we must load the existing one.
        logging.info(f"Loading existing autoencoder from {ae_path}. Use --overwrite to retrain.")
        # The logic to load the compressor is already below, so we just need to ensure 'compressor' is initialized
        compressor = None

    if compressor is None:
        # Load
        # Compute L and D similar to fit (quick probe)
        first_image = _first_image_from_builder(original_builders[0])
        chw = preprocess_images_in_memory(np.asarray([first_image]), FLAGS.vggt_input_res)
        klnd, _ = extractor.extract_layers(chw)
        L = klnd.shape[1]; D = klnd.shape[3]
        compressor = AECompressor(
            num_layers=L,
            input_dim=D,
            bottleneck_dim=target_size[1],
            hidden_dim=FLAGS.ae_hidden,
            use_weighted_layer_fusion=FLAGS.use_weighted_layer_fusion,
        )
        compressor.load(ae_path, map_location='cpu')
        compressor = compressor.to(device).eval()
        logging.info("Loaded AE compressor from %s", ae_path)

    # AE verification (optional) before writing any datasets
    if FLAGS.ae_verification_enable:
        verification_dir = FLAGS.ae_verification_dir or os.path.join(output_root, "ae_verification")
        try:
            verify_autoencoder_reconstruction(
                builders=original_builders,
                extractor=extractor,
                compressor=compressor,
                num_samples=int(FLAGS.ae_verification_count),
                output_dir=verification_dir,
                device=device,
            )
        except Exception as e:
            logging.warning("AE verification failed: %s", e)

    # Build each compressed dataset
    for builder in original_builders:
        logging.info("###### PROCESSING DATASET: %s ######", builder.name)
        try:
            new_builder = CompressedVggtDatasetTorch(
                original_builder=builder,
                extractor=extractor,
                compressor=compressor,
                target_size=target_size,
                data_dir=output_root,
            )

            dataset_output_dir = os.path.join(output_root, new_builder.name)

            # Check if the dataset already exists and decide what to do
            if tf.io.gfile.exists(dataset_output_dir):
                if FLAGS.overwrite:
                    logging.warning(f"--overwrite is True. Deleting existing dataset at {dataset_output_dir}")
                    tf.io.gfile.rmtree(dataset_output_dir)
                else:
                    logging.info(f"Dataset already exists at {dataset_output_dir}. Skipping generation. Use --overwrite to replace it.")
                    continue  # Skip to the next builder in the loop

            new_builder.download_and_prepare()
            logging.info("Successfully created TFDS dataset '%s' at '%s'.", new_builder.name, output_root)
        except Exception as e:
            logging.error("Failed to process dataset %s. Error: %s", builder.name, e, exc_info=True)


if __name__ == "__main__":
    app.run(main)