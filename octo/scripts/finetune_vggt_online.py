import datetime
from functools import partial
import os
from typing import Optional, List, Tuple

from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import optax
import tensorflow as tf
import tqdm
import wandb
import sys
import jaxlib
from jaxlib import version as jv
from flax.core import unfreeze, freeze
import jax.numpy as jnp

# --- Added for online VGGT feature computation ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vggt.models.vggt import VGGT
from contextlib import nullcontext

from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)
from octo.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    process_text,
    Timer,
    TrainState,
)

try:
    from jax_smi import initialise_tracking  # type: ignore
    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

print("==== TRAIN START ENV ====")
print("PYTHON exe:", sys.executable)
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("jax:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("cuda baked:", getattr(jv, "__cuda_version__", None))
print("cudnn baked:", getattr(jv, "__cudnn_version__", None))
print("jaxlib path:", jaxlib.__file__)
print("devices:", jax.devices())
print("=========================")

# Keep existing flags
flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
flags.DEFINE_bool("dump_train_images", False, "If True, save a few input images for inspection.")
flags.DEFINE_integer("dump_train_images_max", 20, "Max number of training images to dump.")
flags.DEFINE_string("dump_train_images_dir", "./train_image_dumps", "Directory to save dumped training images.")
flags.DEFINE_bool("use_vision_encoder", True, "If True then use Octo's vision encoder, else discard it.")
flags.DEFINE_enum(
    "vggt_concat_mode",
    "tokens",
    ["tokens", "features"],
    "How to combine VGGT tokens with patch tokens: 'tokens' (concat along token axis) or 'features' (concat along feature axis).",
)

# Config file remains the same
default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/finetune_config.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

# --- New CLI flags for online VGGT + AE ---
flags.DEFINE_string("ae_path", None, "Path to trained AE .pt file (required to enable online VGGT)")
flags.DEFINE_bool("vggt_use_cuda", True, "Use CUDA for VGGT/AE if available.")
flags.DEFINE_integer("vggt_input_res", 224, "VGGT input resolution (square).")
flags.DEFINE_integer("vggt_eval_batch_size", 16, "Batch size for VGGT forward inside process_batch.")
flags.DEFINE_integer("vggt_agg_layers", 24, "Number of layers to aggregate (24 for all; or subset).")
flags.DEFINE_string("vggt_layer_indices", "3,10,16,22", "Comma-separated 0-based indices (used when vggt_agg_layers < 24).")
flags.DEFINE_string("vggt_target_size", "64,512", "Target compressed size as 'height,width' => (n_tokens, feature_dim).")


# =========================
# Online VGGT + AE helpers (self-contained, no external imports)
# =========================

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


class TorchVGGTExtractor:
    def __init__(self, device: torch.device, input_res: int, agg_layers: int, layer_indices: Optional[List[int]] = None):
        self.device = device
        self.input_res = input_res
        self.agg_layers = int(agg_layers)
        self.layer_indices = layer_indices
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()
        # Pick AMP dtype (bf16 on Hopper/Ampere+, else fp16). CPU -> no autocast
        if self.device.type == 'cuda':
            try:
                major, _ = torch.cuda.get_device_capability()
                self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
            except Exception:
                self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

    @torch.no_grad()
    def extract_layers(self, chw_images: np.ndarray):
        x = torch.from_numpy(chw_images).to(self.device)  # [K,3,H,W]
        x = x.unsqueeze(1)  # [K,1,3,H,W]
        amp_ctx = (
            torch.cuda.amp.autocast(dtype=self.amp_dtype)
            if (self.device.type == 'cuda' and self.amp_dtype is not None)
            else nullcontext()
        )
        with amp_ctx:
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


def preprocess_images_in_memory(images_np: np.ndarray, target_size: int) -> np.ndarray:
    """Aspect-preserving resize to nearest multiple of 14 and white-pad to square; returns CHW floats in [0,1]."""
    from PIL import Image

    processed_images = []
    for img_array in images_np:
        pil_image = Image.fromarray(img_array)

        if pil_image.mode == 'RGBA':
            background = Image.new('RGBA', pil_image.size, (255, 255, 255, 255))
            pil_image = Image.alpha_composite(background, pil_image)
        pil_image = pil_image.convert('RGB')

        width, height = pil_image.size
        if width >= height:
            new_width = target_size
            new_height = int(round(height * (new_width / width) / 14) * 14)
        else:
            new_height = target_size
            new_width = int(round(width * (new_height / height) / 14) * 14)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))

        h_padding = target_size - arr.shape[1]
        w_padding = target_size - arr.shape[2]
        pad_top = h_padding // 2
        pad_bottom = h_padding - pad_top
        pad_left = w_padding // 2
        pad_right = w_padding - pad_left
        arr = np.pad(arr, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=1.0)
        processed_images.append(arr)

    return np.stack(processed_images, axis=0)


def resize_and_stack_per_layer(
    features_klnd: np.ndarray,
    sqrt_n: int,
    target_tokens_hw: int,
    device: torch.device,
) -> torch.Tensor:
    """GPU bilinear downsample per-layer spatial grid to target_side x target_side and return torch.Tensor [K,L,T,D]."""
    K, L, N, D = features_klnd.shape
    s = sqrt_n
    x = torch.from_numpy(features_klnd).to(device=device, dtype=torch.float32)  # [K,L,N,D]
    x = x.reshape(K * L, s, s, D).permute(0, 3, 1, 2).contiguous()

    target_side = int(np.sqrt(target_tokens_hw))
    x_small = F.interpolate(x, size=(target_side, target_side), mode='bilinear', align_corners=False)
    x_small = x_small.permute(0, 2, 3, 1).contiguous().view(K, L, target_side * target_side, D)
    return x_small  # stays on device


# Global state for online VGGT
VGGT_ONLINE_STATE = {
    "device": None,
    "extractor": None,
    "compressor": None,
    "target_tokens": 64,
}


def _init_vggt_online_if_needed():
    if VGGT_ONLINE_STATE["extractor"] is not None and VGGT_ONLINE_STATE["compressor"] is not None:
        return

    device = torch.device('cuda' if (FLAGS.vggt_use_cuda and torch.cuda.is_available()) else 'cpu')

    # Parse layer indices
    if FLAGS.vggt_agg_layers < 24:
        try:
            layer_indices = [int(x) for x in FLAGS.vggt_layer_indices.split(',') if x.strip() != '']
        except Exception:
            layer_indices = [3, 10, 16, 22]
    else:
        layer_indices = None

    extractor = TorchVGGTExtractor(device, FLAGS.vggt_input_res, FLAGS.vggt_agg_layers, layer_indices)

    # Probe L and D to initialize AE
    # We'll create a small synthetic zero image to avoid touching TFDS here; shapes are fixed (VGGT will still output expected dims)
    fake = np.zeros((1, FLAGS.vggt_input_res, FLAGS.vggt_input_res, 3), dtype=np.uint8)
    chw = preprocess_images_in_memory(fake, FLAGS.vggt_input_res)
    klnd, _ = extractor.extract_layers(chw)
    L = klnd.shape[1]
    D = klnd.shape[3]

    # Load AE
    if FLAGS.ae_path is None or len(str(FLAGS.ae_path)) == 0:
        raise ValueError("--ae_path is required for finetune_vggt_online.py")
    target_hw, target_dim = _parse_target_size(FLAGS.vggt_target_size)

    compressor = AECompressor(
        num_layers=L,
        input_dim=D,
        bottleneck_dim=target_dim,
        hidden_dim=2048,
        use_weighted_layer_fusion=True,
    )
    compressor.load(FLAGS.ae_path, map_location='cpu')
    compressor = compressor.to(device).eval()

    VGGT_ONLINE_STATE.update({
        "device": device,
        "extractor": extractor,
        "compressor": compressor,
        "target_tokens": target_hw,
    })
    logging.info("Initialized VGGT online extractor + AE: device=%s, target=(%d,%d)", device, target_hw, target_dim)


def compute_vggt_tokens_for_batch(image_5d: np.ndarray) -> np.ndarray:
    """
    Compute (B,T,64,512) tokens for the provided (B,T,H,W,C) images using online VGGT + AE.
    Returns float32 numpy array.
    """
    _init_vggt_online_if_needed()
    extractor: TorchVGGTExtractor = VGGT_ONLINE_STATE["extractor"]
    compressor: AECompressor = VGGT_ONLINE_STATE["compressor"]
    target_tokens_hw: int = VGGT_ONLINE_STATE["target_tokens"]

    images_np = np.asarray(image_5d)
    if images_np.ndim != 5 or images_np.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected (B,T,H,W,C) images; got shape {images_np.shape}")
    B, T, H, W, C = images_np.shape

    # Flatten across time
    images_bt = images_np.reshape(B * T, H, W, C)
    chw = preprocess_images_in_memory(images_bt, FLAGS.vggt_input_res)  # [K,3,H',W']

    # Process in chunks to bound memory
    batch_size = max(1, int(FLAGS.vggt_eval_batch_size))
    tokens_per_image_list: List[np.ndarray] = []
    for j in range(0, chw.shape[0], batch_size):
        sub = chw[j:j + batch_size]
        klnd, sqrt_n = extractor.extract_layers(sub)         # [K,L,N,D] on CPU numpy
        k_l_t_d = resize_and_stack_per_layer(klnd, sqrt_n, target_tokens_hw, extractor.device)   # torch [K,L,T,D] on device

        K, L, TT, D = k_l_t_d.shape
        toks_ld = k_l_t_d.view(K * TT, L, D)  # [K*T, L, D] on device
        with torch.no_grad():
            z = compressor.compress_tokens(toks_ld)
        z = z.view(K, TT, -1).detach().cpu().numpy().astype(np.float32)  # [K, T, bottleneck]
        tokens_per_image_list.append(z)

    tokens_bt_t_d = np.concatenate(tokens_per_image_list, axis=0)   # [B*T, 64, 512]
    tokens_b_t_t_d = tokens_bt_t_d.reshape(B, T, tokens_bt_t_d.shape[1], tokens_bt_t_d.shape[2])
    return tokens_b_t_t_d


# =========================
# Main finetuning script (kept same; only injection points are marked)
# =========================

def main(_):
    # initialize_compilation_cache()
    # Ensure VisionMixer picks up concat mode without polluting the batch
    os.environ["VGGT_CONCAT_MODE"] = FLAGS.vggt_concat_mode
    devices = jax.devices()
    logging.info(
        f"""
        Octo Finetuning Script
        ======================
        Pretrained model: {FLAGS.config.pretrained_path}
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.name}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}

        # Devices: {jax.device_count()}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
    """
    )

    #########
    # Setup Jax Data Parallelism
    #########

    assert (
        FLAGS.config.batch_size % len(devices) == 0
    ), f"Batch size ({FLAGS.config.batch_size}) must be divisible by the number of devices ({len(devices)})"
    assert (
        FLAGS.config.viz_kwargs.eval_batch_size % len(devices) == 0
    ), f"Eval batch size ({FLAGS.config.viz_kwargs.eval_batch_size}) must be divisible by the number of devices ({len(devices)})"

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    #########
    # Setup WandB
    #########

    name = format_name_with_config(
        FLAGS.name,
        FLAGS.config.to_dict(),
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        config=FLAGS.config.to_dict(),
        id=wandb_id,
        name=name,
        mode="disabled" if FLAGS.debug else None,
        **FLAGS.config.wandb,
    )

    #########
    # Load Pretrained model + optionally modify config
    #########

    pretrained_model = OctoModel.load_pretrained(
        FLAGS.config.pretrained_path,
        step=FLAGS.config.pretrained_step,
    )
    flat_config = flax.traverse_util.flatten_dict(
        pretrained_model.config, keep_empty_nodes=True
    )
    for d_key in flax.traverse_util.flatten_dict(
        FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    ):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]

    config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    config.update(FLAGS.config.get("update_config", ConfigDict()))
    config = config.to_dict()

    # Wrap the pretrained 'primary' tokenizer with VisionMixer, inheriting its exact spec
    try:
        obs_toks = config["model"].get("observation_tokenizers", {})
        old_primary = obs_toks.get("primary")
        if isinstance(old_primary, dict):
            obs_toks["primary"] = ModuleSpec.create(
                "octo.model.components.tokenizers:VisionMixer",
                patch_tokenizer_spec=old_primary,
                vggt_tokenizer_spec={
                    "module": "octo.model.components.tokenizers:VGGTTokenizer",
                },
                concat_mode=FLAGS.vggt_concat_mode,
            )
            config["model"]["observation_tokenizers"] = obs_toks
            config["model"]["repeat_task_tokens"] = True
            logging.info("Wrapped 'primary' with VisionMixer, inheriting pretrained encoder spec.")
    except Exception as _e:
        logging.warning("Could not wrap primary with VisionMixer: %s", _e)

    #########
    # Setup Data Loader
    #########

    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    # Initialize online VGGT + AE once (before data iterator) to catch early errors
    _init_vggt_online_if_needed()

    def process_batch(batch):
        # Keep existing text processing
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        if "task" not in batch:
            batch["task"] = {}

        # Compute online VGGT tokens from already-augmented images if present
        obs = batch.get("observation", {})
        image_primary = obs.get("image_primary")
        try:
            if image_primary is not None:
                vggt_tokens = compute_vggt_tokens_for_batch(image_primary)
                # Ensure dtype float32 for JAX stability
                obs["vggt_tokens"] = vggt_tokens.astype(np.float32)
                batch["observation"] = obs
        except Exception as e:
            logging.warning("Online VGGT token computation failed for this batch: %s", e)

        # Respect environment toggle to optionally drop tokens
        import os as _os
        if _os.environ.get("DISABLE_VGGT_TOKENS", "0") == "1" and "vggt_tokens" in obs:
            obs.pop("vggt_tokens", None)
            pad = obs.get("pad_mask_dict")
            if pad is not None and "vggt_tokens" in pad:
                pad.pop("vggt_tokens", None)

        # If using VGGT-only mode, drop images and ensure pad mask alignment
        if not FLAGS.use_vision_encoder:
            for k in list(obs.keys()):
                if "image" in k:
                    obs.pop(k, None)
            pad = obs.get("pad_mask_dict")
            if pad is None:
                obs["pad_mask_dict"] = {}
                pad = obs["pad_mask_dict"]
            for k in list(pad.keys()):
                if "image" in k:
                    pad.pop(k, None)
            if "vggt_tokens" in obs and "timestep_pad_mask" in obs:
                pad["vggt_tokens"] = obs["timestep_pad_mask"].astype(bool)
            batch["task"].pop("image_primary", None)
        else:
            # Vision-encoder path: set image goal if available
            if "image_primary" in batch["observation"]:
                batch["task"]["image_primary"] = batch["observation"]["image_primary"][:, 0]

        return batch

    dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )
    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    obs = example_batch.get("observation", {})
    img = obs.get("image_primary")
    if img is not None:
        print("[finetune-online] image_primary shape:", getattr(img, "shape", None), "dtype:", getattr(img, "dtype", None))
    print("example_batch observation keys:", list(example_batch["observation"].keys()))

    # ---- DIAGNOSTIC: Inspect example_batch leaves ----
    def _print_leaf_info(key_path, x):
        try:
            shape = getattr(x, "shape", None)
            dtype = getattr(x, "dtype", type(x))
            sliceable = True
            err = None
            try:
                _ = x[:1]
            except Exception as _e:
                sliceable = False
                err = str(_e)
            print(f"[BATCH] {'/'.join(key_path):<60} shape={shape} dtype={dtype} sliceable={sliceable}")
            if not sliceable:
                print(f"        -> slice error: {err}")
        except Exception as e:
            print(f"[BATCH] {'/'.join(key_path)}: <error printing leaf> {e}")

    def _walk(prefix, obj):
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                _walk(prefix + [str(k)], obj[k])
        elif isinstance(obj, (list, tuple)):
            for idx, v in enumerate(obj):
                _walk(prefix + [f"[{idx}]"] , v)
        else:
            _print_leaf_info(prefix, obj)

    print("\n=== DIAGNOSTIC: example_batch leaf summary (pre-initialization) ===")
    _walk([], example_batch)
    print("=== END DIAGNOSTIC ===\n")

    #########
    # Load Pretrained Model
    #########

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)

    logging.info("Final observation_tokenizers keys: %s",
             list(config["model"]["observation_tokenizers"].keys()))

    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
    )

    # Manual weight injection retained from original file
    params_new = unfreeze(model.params)
    params_pre = unfreeze(pretrained_model.params)

    def get_(d, path):
        for k in path:
            if k not in d: return None
            d = d[k]
        return d

    def set_(d, path, value):
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = value

    flat_pre = flatten_dict(params_pre)
    flat_new_before = flatten_dict(params_new)

    # 1) Shape-safe copy: pretrained SmallStem16_0 -> new patch_tokenizer.SmallStem16_0
    src_prefix = ("octo_transformer", "observation_tokenizers_primary", "SmallStem16_0")
    dst_prefix = ("octo_transformer", "observation_tokenizers_primary", "patch_tokenizer", "SmallStem16_0")

    num_copied = 0
    for k, v in flat_pre.items():
        if len(k) >= len(src_prefix) and k[:len(src_prefix)] == src_prefix:
            tail = k[len(src_prefix):]
            dst_key = dst_prefix + tail
            if dst_key in flat_new_before and flat_new_before[dst_key].shape == v.shape:
                set_(params_new, dst_key, v)
                num_copied += 1
    print(f"---> Copied {num_copied} SmallStem16 leaves into patch_tokenizer")

    # 2) Remap top-level obs projection and positional embedding (old -> new)
    remaps = [
        (("octo_transformer", "obs_image_primary_pos_embedding"),
        ("octo_transformer", "obs_primary_pos_embedding")),
        (("octo_transformer", "obs_image_primary_projection", "kernel"),
        ("octo_transformer", "obs_primary_projection", "kernel")),
        (("octo_transformer", "obs_image_primary_projection", "bias"),
        ("octo_transformer", "obs_primary_projection", "bias")),
    ]
    for src, dst in remaps:
        src_val = get_(params_pre, src)
        dst_val = get_(params_new, dst)
        if src_val is not None and dst_val is not None and src_val.shape == dst_val.shape:
            set_(params_new, dst, src_val)
            print(f"---> Remapped {'.'.join(src)} -> {'.'.join(dst)}")

    final_merged_params = merge_params(freeze(params_new), pretrained_model.params)
    model = model.replace(params=final_merged_params)
    print("Manual weight injection complete.")

    # Verification block retained
    flat_final = flatten_dict(unfreeze(model.params))

    def obs_tok_roots(flat):
        return sorted({k[1] for k in flat if len(k)>=2 and k[0]=='octo_transformer' and k[1].startswith('observation_tokenizers_')})

    print("NEW obs tokenizer roots:", obs_tok_roots(flat_final))
    print("PRE obs tokenizer roots:", obs_tok_roots(flat_pre))

    ROOT = "observation_tokenizers_primary"
    new_obs = {k:v for k,v in flat_final.items() if len(k)>=2 and k[0]=='octo_transformer' and k[1]==ROOT}

    loaded = [k for k in new_obs if k in flat_pre and flat_pre[k].shape == new_obs[k].shape]
    missing = [k for k in new_obs if k not in flat_pre]
    mismatch = [k for k in new_obs if k in flat_pre and flat_pre[k].shape != new_obs[k].shape]
    print(f"obs subtree: {ROOT}")
    print(f"loaded: {len(loaded)}  missing: {len(missing)}  mismatch: {len(mismatch)}")

    # =========================
    # Optimizer and Train State (unchanged)
    # =========================

    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    tx, lr_callable, param_norm_callable = create_optimizer(
        params,
        **FLAGS.config.optimizer.to_dict(),
    )
    train_state = TrainState.create(
        model=model,
        tx=tx,
        rng=jax.random.PRNGKey(FLAGS.config.seed),
    )

    #########
    # Save all metadata (unchanged)
    #########

    if FLAGS.config.save_dir is not None:
        # Allow full resume: if a resume_dir is provided, use it directly
        resume_dir = FLAGS.config.get("resume_dir", None)
        if resume_dir is not None and isinstance(resume_dir, str) and len(resume_dir) > 0:
            save_dir = resume_dir
        else:
            save_dir = tf.io.gfile.join(
                FLAGS.config.save_dir,
                FLAGS.config.wandb.project,
                FLAGS.config.wandb.group or "",
                wandb_id,
            )
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        logging.info("Saving to %s", save_dir)
        save_callback = SaveCallback(save_dir)

        # Add window_size to top of config, to make eval easier
        new_config = ConfigDict(model.config)
        new_config["window_size"] = example_batch["observation"][
            "timestep_pad_mask"
        ].shape[1]
        model = model.replace(config=new_config)

        # Save finetuning config
        with tf.io.gfile.GFile(
            tf.io.gfile.join(save_dir, "finetune_config.json"), "w"
        ) as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        logging.warning("save_dir not passed in, not saving checkpoints")

    example_batch_spec = jax.tree_map(
        lambda arr: (arr.shape, str(arr.dtype)), example_batch
    )
    wandb.config.update(
        dict(example_batch_spec=example_batch_spec), allow_val_change=True
    )

    #########
    # Define loss, train_step, and eval_step (unchanged)
    #########

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # action head knows to pull out the "action" readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
        donate_argnums=(0,),
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        grad_norm = optax.global_norm(grads)
        new_params = optax.apply_updates(state.model.params, updates)
        new_model = state.model.replace(params=new_params)
        new_state = state.replace(model=new_model, opt_state=new_opt_state, rng=rng, step=state.step + 1)
        info.update({
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm_callable(state.model.params),
            "learning_rate": lr_callable(state.step),
        })
        return new_state, info

    #########
    # Resume from checkpoint (unchanged)
    #########

    start_step = 0
    if save_dir is not None:
        try:
            latest_step = save_callback.state_checkpointer.latest_step()
        except Exception:
            latest_step = None
        if latest_step is not None:
            train_state = save_callback.state_checkpointer.restore(latest_step, items=train_state)
            start_step = int(train_state.step)
            logging.info("Restored checkpoint from %s at step %d", save_dir, start_step)

    #########
    # Callbacks (unchanged)
    #########

    if FLAGS.config.modality == "image_conditioned":
        modes_to_evaluate = ["image_conditioned"]
    elif FLAGS.config.modality == "text_conditioned":
        modes_to_evaluate = ["text_conditioned"]
    elif FLAGS.config.modality == "multimodal":
        modes_to_evaluate = ["image_conditioned", "text_conditioned"]
    else:
        modes_to_evaluate = ["base"]

    dataset_kwargs_list = [FLAGS.config.dataset_kwargs]

    val_callback = ValidationCallback(
        loss_fn=loss_fn,
        process_batch_fn=process_batch,
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        modes_to_evaluate=modes_to_evaluate,
        **FLAGS.config.val_kwargs,
    )

    viz_callback = VisualizationCallback(
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        modes_to_evaluate=modes_to_evaluate,
        **FLAGS.config.viz_kwargs,
    )

    if "rollout_kwargs" in FLAGS.config:
        try:
            import libero.envs  # type: ignore
            rollout_callback = RolloutVisualizationCallback(
                text_processor=text_processor,
                unnormalization_statistics=dataset.dataset_statistics["action"],
                **FLAGS.config.rollout_kwargs.to_dict(),
            )
        except Exception as e:
            rollout_callback = None
            logging.warning(f"Could not create RolloutVisualizationCallback: {e}")
    else:
        rollout_callback = None

    #########
    # Train loop (unchanged)
    #########

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    dumped = 0
    dump_enabled = FLAGS.dump_train_images
    if dump_enabled:
        try:
            import imageio
            os.makedirs(FLAGS.dump_train_images_dir, exist_ok=True)
        except Exception:
            dump_enabled = False
            logging.warning("Disabling dump_train_images: imageio not available or cannot create output dir.")

    _batch_check_printed = False

    for i in tqdm.tqdm(
        range(start_step, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps) - start_step,
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        if not _batch_check_printed:
            print("\n" + "="*50)
            print("DEBUG: Final batch check (what the model receives)!")
            print("Batch observation keys:", list(batch['observation'].keys()))
            if 'vggt_tokens' in batch['observation']:
                vggt_shape = batch['observation']['vggt_tokens'].shape
                vggt_dtype = batch['observation']['vggt_tokens'].dtype
                print(f"  -> SUCCESS: 'vggt_tokens' are in the final batch!")
                print(f"     Shape: {vggt_shape} (Batch, Window, H, W)")
                print(f"     DType: {vggt_dtype}")
            else:
                print("  -> CRITICAL WARNING: 'vggt_tokens' were dropped somewhere in the data pipeline!")
            print("="*50 + "\n")
            _batch_check_printed = True

            image_obs_keys = [k for k in batch['observation'].keys() if 'image' in k]
            if image_obs_keys:
                print(f"  -> WARNING: image observation keys still present: {image_obs_keys}")
            else:
                print("  -> WARNING: No image observations present; using only VGGT tokens.")
            print("="*50 + "\n")
            _batch_check_printed = True

        if dump_enabled and dumped < FLAGS.dump_train_images_max:
            obs_dump = batch.get("observation", {})
            img_dump = obs_dump.get("image_primary")
            if img_dump is not None:
                try:
                    import numpy as _np
                    arr = _np.asarray(img_dump)
                    if arr.ndim >= 5:
                        frame = arr[0, 0]
                    elif arr.ndim == 4:
                        frame = arr[0]
                    else:
                        frame = arr
                    out_path = os.path.join(FLAGS.dump_train_images_dir, f"step_{i:06d}_idx_{dumped:03d}.png")
                    import imageio
                    imageio.imwrite(out_path, frame)
                    dumped += 1
                except Exception as _e:
                    pass

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")

            with timer("val"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i)

            with timer("visualize"):
                viz_metrics = viz_callback(train_state, i + 1)
                wandb_log(viz_metrics, step=i)

            if rollout_callback is not None:
                with timer("rollout"):
                    rollout_metrics = rollout_callback(train_state, i + 1)
                    wandb_log(rollout_metrics, step=i)

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)

        # Early stop aligned with original script constraint
        if (i + 1) >= 100000:
            logging.info(
                "Early stopping at step %d (target 150000). Cosine scheduler remains configured for %d steps.",
                i + 1,
                int(FLAGS.config.num_steps),
            )
            if (i + 1) % FLAGS.config.save_interval != 0 and save_dir is not None:
                logging.info("Saving final checkpoint before early stop...")
                save_callback(train_state, i + 1)
            break


if __name__ == "__main__":
    app.run(main)
