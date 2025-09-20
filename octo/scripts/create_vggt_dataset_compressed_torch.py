import os
from typing import Tuple, Optional, List, Iterable

import numpy as np
from absl import app, flags, logging
import tensorflow as tf
import tensorflow_datasets as tfds

import torch
import torch.nn as nn
import torch.nn.functional as F

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
class AECompressor(nn.Module):
    """
    Lightweight MLP autoencoder trained to map per-spatial token vectors of shape (L*D) -> 512, and reconstruct back.

    We assume the upstream pipeline first resizes per-layer spatial tokens to 8x8 (64 tokens),
    so compression is applied independently per spatial location. This mirrors PCA's per-token behavior,
    but with a learned non-linear projection.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 512, hidden_dim: int = 2048):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.encoder(x)

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

        K, L, N, D = layers.shape
        sqrt_n = int(round(np.sqrt(N)))
        return layers.detach().cpu().numpy(), sqrt_n


def resize_and_stack_per_layer(features_klnd: np.ndarray, sqrt_n: int) -> np.ndarray:
    """
    Converts [K, L, N, D] into [K, L, 64, D] by bilinear resizing each layer's spatial grid to 8x8.
    Uses PyTorch for vectorized CPU/GPU interpolation for numerical parity with training.
    """
    K, L, N, D = features_klnd.shape
    s = sqrt_n
    x = torch.from_numpy(features_klnd).float()  # [K,L,N,D]
    x = x.view(K * L, s, s, D).permute(0, 3, 1, 2)  # [K*L, D, s, s]
    x_small = F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)  # [K*L, D, 8, 8]
    x_small = x_small.permute(0, 2, 3, 1).contiguous().view(K, L, 64, D)  # [K,L,64,D]
    return x_small.numpy()


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
                flat = torch.from_numpy(k_l_64_d).float().view(K * S64, L * D)
                with torch.no_grad():
                    z = self._compressor.compress(flat)
                z = z.view(K, S64, -1).cpu().numpy().astype(np.float16)  # [K,64,512]
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


# -------------------------
# Autoencoder fitting utilities
# -------------------------
def _sample_tokens_for_ae(builders: List[tfds.core.DatasetBuilder], extractor: TorchVGGTExtractor, num_tokens: int, batch_size: int) -> Iterable[torch.Tensor]:
    """
    Yields mini-batches of flattened per-token vectors (shape [B, L*D]) for AE fitting,
    collected across datasets until reaching approximately num_tokens samples.
    """
    remaining = int(num_tokens)
    for builder in builders:
        ds = builder.as_dataset(split='train')
        for episode in tfds.as_numpy(ds):
            steps = list(episode['steps'])
            if not steps:
                continue
            obs = [s['observation'] for s in steps]
            if 'image' not in obs[0]:
                continue
            images_np = np.stack([o['image'] for o in obs], axis=0)
            chw_images = preprocess_images_in_memory(images_np, FLAGS.vggt_input_res)

            # Batch through VGGT
            for j in range(0, chw_images.shape[0], batch_size):
                batch = chw_images[j:j+batch_size]
                klnd, sqrt_n = extractor.extract_layers(batch)    # [K,L,N,D]
                k_l_64_d = resize_and_stack_per_layer(klnd, sqrt_n)  # [K,L,64,D]
                K, L, S64, D = k_l_64_d.shape
                flat = torch.from_numpy(k_l_64_d).float().view(K * S64, L * D)  # per-spatial token

                # Yield in chunks of up to remaining
                idx = 0
                while idx < flat.shape[0] and remaining > 0:
                    take = min(remaining, 2048)  # avoid very small batches
                    yield flat[idx: idx + take]
                    idx += take
                    remaining -= take
                    if remaining <= 0:
                        return


def _fit_autoencoder(builders: List[tfds.core.DatasetBuilder], extractor: TorchVGGTExtractor, target_size: Tuple[int, int], device: torch.device) -> AECompressor:
    target_h, target_w = target_size  # expect (64, 512)
    # Determine input dim = L * D (after layer selection), by running a tiny probe
    probe_ds = builders[0].as_dataset(split='train').take(1)
    probe = next(iter(tfds.as_numpy(probe_ds)))
    first_image = np.asarray(probe['steps'][0]['observation']['image'])
    chw = preprocess_images_in_memory(np.asarray([first_image]), FLAGS.vggt_input_res)
    klnd, sqrt_n = extractor.extract_layers(chw)
    L = klnd.shape[1]
    D = klnd.shape[3]
    input_dim = L * D

    model = AECompressor(input_dim=input_dim, bottleneck_dim=target_w, hidden_dim=FLAGS.ae_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=FLAGS.ae_lr)
    loss_fn = nn.MSELoss()

    # Build a simple stream of batches
    batches = _sample_tokens_for_ae(builders, extractor, FLAGS.compression_samples, FLAGS.batch_size_eval)

    model.train()
    for epoch in range(FLAGS.ae_epochs):
        seen = 0
        running = 0.0
        for batch in batches:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            z, recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            seen += batch.shape[0]
            running += float(loss.detach().cpu().item()) * batch.shape[0]
            if seen >= FLAGS.compression_samples:
                break
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
    dataset_names = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    if not dataset_names:
        raise ValueError("No valid datasets found under input_data_dir")
    original_builders = []
    for name in dataset_names:
        b = tfds.builder(name, data_dir=input_root)
        original_builders.append(b)

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
    ae_path = FLAGS.ae_path or os.path.join(output_root, f"vggt_autoencoder_{FLAGS.vggt_agg_layers}L_{target_size[0]}x{target_size[1]}.pt")
    compressor = None
    if FLAGS.train_if_missing and (not os.path.exists(ae_path)):
        logging.info("Fitting autoencoder compressor (samples=%d, epochs=%d)...", FLAGS.compression_samples, FLAGS.ae_epochs)
        compressor = _fit_autoencoder(original_builders, extractor, target_size, device)
        os.makedirs(os.path.dirname(ae_path), exist_ok=True)
        compressor.cpu().save(ae_path)
        logging.info("Saved AE compressor to %s", ae_path)
        compressor = compressor.to(device).eval()
    else:
        compressor = None

    if compressor is None:
        # Load
        # Compute input_dim similar to fit (quick probe)
        probe_ds = original_builders[0].as_dataset(split='train').take(1)
        probe = next(iter(tfds.as_numpy(probe_ds)))
        first_image = np.asarray(probe['steps'][0]['observation']['image'])
        chw = preprocess_images_in_memory(np.asarray([first_image]), FLAGS.vggt_input_res)
        klnd, _ = extractor.extract_layers(chw)
        L = klnd.shape[1]; D = klnd.shape[3]
        input_dim = L * D
        compressor = AECompressor(input_dim=input_dim, bottleneck_dim=target_size[1], hidden_dim=FLAGS.ae_hidden)
        compressor.load(ae_path, map_location='cpu')
        compressor = compressor.to(device).eval()
        logging.info("Loaded AE compressor from %s", ae_path)

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

            # Overwrite if requested by removing destination dir
            dataset_output_dir = os.path.join(output_root, new_builder.name)
            if tf.io.gfile.exists(dataset_output_dir):
                logging.warning("Overwriting existing dataset at %s", dataset_output_dir)
                tf.io.gfile.rmtree(dataset_output_dir)

            new_builder.download_and_prepare()
            logging.info("Successfully created TFDS dataset '%s' at '%s'.", new_builder.name, output_root)
        except Exception as e:
            logging.error("Failed to process dataset %s. Error: %s", builder.name, e, exc_info=True)


if __name__ == "__main__":
    app.run(main)

