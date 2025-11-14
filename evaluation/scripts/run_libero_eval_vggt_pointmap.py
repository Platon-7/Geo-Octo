import sys
import warnings
import json
import numpy as np
from datetime import datetime
from contextlib import nullcontext

# Add compatibility shim before importing anything else
try:
    import jax.numpy as jnp
    if not hasattr(jnp, 'DeviceArray'):
        jnp.DeviceArray = jnp.ndarray
        print("[FIX] Added DeviceArray compatibility shim")
except ImportError:
    print("[WARNING] Could not import JAX")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")

# 1. Load the statistics
stats_path = "/home/pkarageorgis/geo_octo/libero_datasets/unified_stats/unified_dataset_statistics_libero_object_no_vggt.json"
with open(stats_path, 'r') as f:
    dataset_statistics = json.load(f)

action_mean = np.array(dataset_statistics['action']['mean'])
action_std = np.array(dataset_statistics['action']['std'])

import logging

import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

import draccus
import tqdm
from libero.libero import benchmark

import wandb

from evaluation.supporting_files.load_fn import load_and_preprocess_images

# Torch + VGGT
import torch
import torch.nn as nn
import torch.nn.functional as F
from vggt.models.vggt import VGGT


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from evaluation.supporting_files.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from evaluation.supporting_files.robot_utils import (
    DATE_TIME,
    get_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from evaluation.supporting_files.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# add at top, near action_mean/std:
action_mask = np.array(dataset_statistics['action'].get('mask')) if 'mask' in dataset_statistics['action'] else None


# Silence noisy libraries
for name in ("octo", "octo.octo", "octo.utils", "octo.model", "flax", "transformers"):
    lg = logging.getLogger(name)
    lg.setLevel(logging.ERROR)
    lg.propagate = False


class DropNoisyOcto(logging.Filter):
    DROP = (
        "contains extra items compared to example_batch",
        "is missing items compared to example_batch",
        "No pad_mask_dict found",
        "No task inputs matching image_primary were found",
        "Skipping observation tokenizer: obs_wrist",
        "repeating task tokens at each timestep",
        "embodiment_action_dim is highly recommended",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(s in record.getMessage() for s in self.DROP)


dropper = DropNoisyOcto()
root = logging.getLogger()
root.addFilter(dropper)
for h in root.handlers:
    h.addFilter(dropper)

try:
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

logging.captureWarnings(True)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    checkpoint_step: Optional[int] = None            # Optional checkpoint step to load (latest if None)

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # VGGT (PyTorch) parameters
    #################################################################################################################
    use_vggt_tokens: bool = True                     # Whether to compute and feed VGGT tokens online
    vggt_input_res: int = 224                        # Input resolution for VGGT model
    vggt_use_cuda: bool = True                       # Whether to use CUDA when available
    vggt_agg_layers: int = 24                        # 24 for all layers, <24 to slice subset
    vggt_layer_indices: str = "3,10,16,22"           # Used when vggt_agg_layers < 24 (0-based indices)
    vggt_ae_path: Optional[str] = None               # Path to saved AE compressor (.pt)
    vggt_device_id: Optional[int] = None             # CUDA device index for VGGT helper (None -> default)
    vggt_eval_batch_size: int = 2                    # Micro-batch size for VGGT helper

    vggt_only_eval: bool = False                     # Used when finetuning removed vision encoder completely

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    #################################################################################################################
    # Attention snapshot capture (disabled by default)
    #################################################################################################################
    capture_attention_snapshot: bool = False         # Enable saving intermediate vision tokens for analysis
    attention_snapshot_task: Optional[str] = "pick up the ketchup and place it in the basket"
    attention_snapshot_episode_idx: int = 3          # Zero-based episode index to capture from
    attention_snapshot_seconds: float = 1.0          # Seconds after episode start to capture observation
    attention_snapshot_control_freq: Optional[float] = None  # Override control frequency if env attribute missing
    attention_snapshot_output_dir: str = "./analysis/attention_snapshots"
    attention_snapshot_filename: Optional[str] = None
    attention_snapshot_label: Optional[str] = None   # Required when capture enabled to distinguish policies
    attention_snapshot_request_pointmap_debug: bool = False  # Save pointmap/readout debug tensors when capturing
    pointmap_key: str = "pointmap"                   # Snapshot helper expects this key for normalized pointmap
    normalize_pointmap: bool = True                  # Whether to snapshot normalized XYZ pointmap

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)

    proprio_projector = None
    if cfg.model_family == "openvla" and cfg.use_proprio:
        from evaluation.supporting_files.openvla_utils import get_proprio_projector

        proprio_projector = get_proprio_projector(
            cfg,
            getattr(model, "llm_dim", None),
            proprio_dim=8,
        )

    action_head = None
    if cfg.model_family == "openvla" and (cfg.use_l1_regression or cfg.use_diffusion):
        from evaluation.supporting_files.openvla_utils import get_action_head

        action_head = get_action_head(cfg, getattr(model, "llm_dim", None))

    noisy_action_projector = None
    if cfg.model_family == "openvla" and cfg.use_diffusion:
        from evaluation.supporting_files.openvla_utils import get_noisy_action_projector

        noisy_action_projector = get_noisy_action_projector(cfg, getattr(model, "llm_dim", None))

    processor = None
    if cfg.model_family == "openvla":
        from evaluation.supporting_files.openvla_utils import get_processor

        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    observation = {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]) 
        ),
    }

    return observation, img


def process_action(action, model_family, action_mean=None, action_std=None):
    if model_family == "openvla":
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
    # change the octo branch in process_action:
    elif model_family == "octo":
        if action_mean is None or action_std is None:
            raise ValueError("Action statistics (mean, std) must be provided for Octo model evaluation!")
        action_mean = action_mean[:action.shape[-1]]
        action_std = action_std[:action.shape[-1]]
        if action_mask is not None:
            mask = action_mask[:action.shape[-1]]
            return np.where(mask, (action * action_std) + action_mean, action)
        return (action * action_std) + action_mean
    else:
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


class AttentionSnapshotManager:
    """Manage when and how to record intermediate vision tokens for later visualization."""

    def __init__(self, cfg: GenerateConfig, task_description: str, log_fn=log_message):
        self.cfg = cfg
        self.task_description = task_description or ""
        self._log = log_fn
        self.enabled = bool(getattr(cfg, "capture_attention_snapshot", False))
        self.request_pointmap_debug = bool(getattr(cfg, "attention_snapshot_request_pointmap_debug", False))
        self._task_match = False
        self.task_match = False
        self.captured = False
        self._pending = False
        self._pending_policy_step = None
        self._freq_cache: Optional[float] = None
        self.output_dir: Optional[Path] = None
        self.output_path: Optional[Path] = None

        if not self.enabled:
            return

        target_task = (cfg.attention_snapshot_task or "").strip().lower()
        current_task = self.task_description.strip().lower()
        self._task_match = (not target_task) or (current_task == target_task)
        self.task_match = self._task_match

        if self._task_match:
            if not cfg.attention_snapshot_label:
                raise ValueError(
                    "attention_snapshot_label must be provided when capture_attention_snapshot is enabled "
                    "and the target task matches."
                )
            base_filename = cfg.attention_snapshot_filename or self._default_filename(current_task or "task")
            self.output_dir = Path(cfg.attention_snapshot_output_dir).expanduser()
            self.output_path = self.output_dir / base_filename

    def _default_filename(self, task_slug: str) -> str:
        slug = "_".join(task_slug.split()) or "task"
        seconds = self.cfg.attention_snapshot_seconds
        sec_str = f"{seconds:.2f}".replace(".", "p")
        episode = self.cfg.attention_snapshot_episode_idx
        return f"{slug}_episode{episode}_t{sec_str}s.npz"

    def _resolve_control_freq(self, control_freq: Optional[float]) -> float:
        if self.cfg.attention_snapshot_control_freq is not None:
            return float(self.cfg.attention_snapshot_control_freq)
        if control_freq is not None:
            return float(control_freq)
        return 20.0  # LIBERO default

    def _target_step(self, control_freq: float) -> int:
        seconds = max(0.0, float(self.cfg.attention_snapshot_seconds))
        return max(0, int(round(seconds * control_freq)))

    def should_capture(self, episode_idx: int, policy_step_idx: int, control_freq: Optional[float]) -> bool:
        if not self.enabled or self.captured or not self._task_match or self.output_path is None:
            return False
        if episode_idx != self.cfg.attention_snapshot_episode_idx:
            return False

        freq = self._resolve_control_freq(control_freq)
        target_step = self._target_step(freq)
        if policy_step_idx < target_step:
            return False

        self._pending = True
        self._pending_policy_step = policy_step_idx
        self._freq_cache = freq
        return True

    def build_capture_spec(self) -> dict:
        spec = {"request_tokens": True}
        if self.request_pointmap_debug:
            spec["request_pointmap_debug"] = True
        return spec

    def commit(
        self,
        payload: Optional[dict],
        image: Optional[np.ndarray],
        episode_idx: int,
        control_freq: Optional[float],
    ) -> None:
        if not self._pending:
            return

        self._pending = False
        freq = self._freq_cache or self._resolve_control_freq(control_freq)
        policy_step_idx = self._pending_policy_step or 0
        seconds_actual = policy_step_idx / freq if freq else None

        if payload is None:
            self._log("Attention snapshot capture requested but returned payload was None.")
            return

        octo_tokens = payload.get("octo_tokens")
        if octo_tokens is None:
            self._log("Attention snapshot payload missing 'octo_tokens'; skipping save.")
            return

        vggt_tokens = payload.get("vggt_tokens")
        image_np = np.asarray(image) if image is not None else None

        if self.output_dir is None or self.output_path is None:
            self._log("Attention snapshot output path is undefined; skipping save.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if self.output_path.exists():
            with np.load(self.output_path, allow_pickle=True) as data:
                existing = {k: data[k] for k in data.files}

        label = self.cfg.attention_snapshot_label or "policy"

        new_entries = {
            f"{label}_octo_tokens": np.asarray(octo_tokens, dtype=np.float32),
        }
        if vggt_tokens is not None:
            new_entries[f"{label}_vggt_tokens"] = np.asarray(vggt_tokens, dtype=np.float32)
        if image_np is not None:
            new_entries[f"{label}_rgb"] = image_np.astype(np.uint8)
        readout_pre = payload.get("readout_pre_pointmap_tokens")
        if readout_pre is not None:
            new_entries[f"{label}_readout_pre_pointmap_tokens"] = np.asarray(readout_pre, dtype=np.float32)
        readout_post = payload.get("readout_post_pointmap_tokens")
        if readout_post is not None:
            new_entries[f"{label}_readout_post_pointmap_tokens"] = np.asarray(readout_post, dtype=np.float32)
        pm_raw = payload.get("pointmap_raw")
        if pm_raw is not None:
            new_entries[f"{label}_pointmap_raw"] = np.asarray(pm_raw, dtype=np.float32)
        pm_norm = payload.get("pointmap_normalized")
        if pm_norm is not None:
            new_entries[f"{label}_pointmap_normalized"] = np.asarray(pm_norm, dtype=np.float32)
        rgb_pre = payload.get("rgb_preprocessed")
        if rgb_pre is not None:
            new_entries[f"{label}_rgb_preprocessed"] = np.asarray(rgb_pre, dtype=np.float32)

        metadata = {
            "policy_label": label,
            "task_description": self.task_description,
            "episode_idx": episode_idx,
            "target_seconds": self.cfg.attention_snapshot_seconds,
            "captured_policy_step": policy_step_idx,
            "control_freq": freq,
            "seconds_elapsed": seconds_actual,
            "timestamp": datetime.utcnow().isoformat(),
            "pretrained_checkpoint": str(self.cfg.pretrained_checkpoint),
            "run_id_note": self.cfg.run_id_note,
            "pointmap_debug_requested": self.request_pointmap_debug,
        }
        new_entries[f"{label}_meta"] = np.array(json.dumps(metadata), dtype=object)

        existing.update(new_entries)
        np.savez_compressed(self.output_path, **existing)

        self.captured = True
        seconds_str = f"{seconds_actual:.2f}s" if seconds_actual is not None else "N/A"
        self._log(
            f"Saved attention snapshot for policy '{label}' at {seconds_str} "
            f"to {self.output_path}"
        )

# ===== VGGT (PyTorch) helpers =====

class WeightedLayerFuser(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        return (x * w.view(*([1] * (x.ndim - 2)), -1, 1)).sum(dim=-2)


class AECompressor(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, bottleneck_dim: int = 512, hidden_dim: int = 8192):
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
        fused = self.fuser(tokens_ld)
        z = self.encoder(fused)
        z = self.output_norm(z)
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
        x = x.unsqueeze(1)  # [K,1,3,H,W]
        output_list, patch_start_idx = self.model.aggregator(x)
        all_layers = []
        for t in output_list:  # [K,1,P,8192]
            t = t[:, 0]  # [K,P,8192]
            t = t[:, patch_start_idx:, :]
            all_layers.append(t)
        layers = torch.stack(all_layers, dim=0).permute(1, 0, 2, 3)  # [K,L,N,8192]
        if self.agg_layers < 24:
            idx = self.layer_indices if (self.layer_indices and len(self.layer_indices) > 0) else [3, 10, 16, 22]
            layers = layers[:, idx, :, :]
        K, L, N, D = layers.shape
        sqrt_n = int(round(np.sqrt(N)))
        return layers.detach().cpu().numpy(), sqrt_n


def resize_and_stack_per_layer(features_klnd: np.ndarray, sqrt_n: int) -> np.ndarray:
    K, L, N, D = features_klnd.shape
    s = sqrt_n
    x = torch.from_numpy(features_klnd).float()
    x = x.reshape(K * L, s, s, D).permute(0, 3, 1, 2)
    target_side = 16  # for 256 tokens
    x_small = F.interpolate(x, size=(target_side, target_side), mode='bilinear', align_corners=False)
    x_small = x_small.permute(0, 2, 3, 1).contiguous().view(K, L, target_side * target_side, D)
    return x_small.numpy()


def _parse_layer_indices(s: str) -> Optional[List[int]]:
    try:
        idx = [int(x) for x in s.split(',') if x.strip() != '']
        return idx
    except Exception:
        return [3, 10, 16, 22]


def compute_compressed_vggt_tokens_torch(image: np.ndarray, vggt_ctx: dict) -> Optional[np.ndarray]:
    if vggt_ctx is None:
        return None

    extractor: TorchVGGTExtractor = vggt_ctx.get("extractor")
    compressor: AECompressor = vggt_ctx.get("compressor")
    cfg: GenerateConfig = vggt_ctx.get("cfg")
    device: torch.device = vggt_ctx.get("device")

    if extractor is None or compressor is None:
        raise RuntimeError("VGGT torch context is missing required components (extractor or compressor).")

    # Preprocess to CHW float32 in [0,1]
    pre = load_and_preprocess_images([image], target_size=cfg.vggt_input_res)
    klnd, sqrt_n = extractor.extract_layers(pre)               # [K,L,N,D]
    k_l_256_d = resize_and_stack_per_layer(klnd, sqrt_n)        # [K,L,256,D]
    K, L, S256, D = k_l_256_d.shape
    tokens = torch.from_numpy(k_l_256_d).float().view(K * S256, L, D).to(device)
    with torch.no_grad():
        z = compressor.compress_tokens(tokens)                  # [K*256,512]
    z = z.view(K, S256, -1).detach().cpu().numpy().astype(np.float16)  # [K,256,512]
    return z[0]


class VGGTPointmapRunner:
    """Compute VGGT pointmaps (xyz + confidence) on demand for debugging."""

    def __init__(self, device: torch.device, input_res: int, batch_size: int = 2):
        self.device = device
        self.input_res = int(input_res)
        self.batch_size = max(1, int(batch_size))
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def compute_pointmap(self, images_bt3hw: np.ndarray) -> np.ndarray:
        b, t, h, w, c = images_bt3hw.shape
        flat = images_bt3hw.reshape(b * t, h, w, c).astype(np.uint8)
        chw = _preprocess_images_for_vggt(flat, self.input_res)
        outputs = []
        amp_ctx = torch.cuda.amp.autocast(enabled=self.device.type == "cuda")

        for start in range(0, chw.shape[0], self.batch_size):
            batch = torch.from_numpy(chw[start : start + self.batch_size]).to(self.device)
            batch = batch.unsqueeze(1)  # (k,1,3,H,W)
            with amp_ctx:
                preds = self.model(batch)
            if isinstance(preds, dict) and "world_points" in preds:
                pts = preds["world_points"][:, 0]
                conf = preds.get("world_points_conf")
                if conf is not None:
                    conf = conf[:, 0][..., None]
                else:
                    conf = torch.ones((*pts.shape[:3], 1), device=pts.device)
                out = torch.cat([pts, conf], dim=-1)
            elif isinstance(preds, dict) and "depth" in preds:
                depth = preds["depth"][:, 0, ..., 0][..., None]
                conf = preds.get("depth_conf")
                if conf is not None:
                    conf = conf[:, 0][..., None]
                else:
                    conf = torch.ones_like(depth)
                zeros = torch.zeros_like(depth)
                out = torch.cat([zeros, zeros, depth, conf], dim=-1)
            else:
                raise RuntimeError("VGGT did not return world_points/depth predictions.")
            outputs.append(out.detach().cpu().numpy())

        stacked = np.concatenate(outputs, axis=0)
        return stacked.reshape(b, t, stacked.shape[1], stacked.shape[2], stacked.shape[3]).astype(np.float32)


def _preprocess_images_for_vggt(images_np: np.ndarray, target_size: int) -> np.ndarray:
    from PIL import Image

    processed = []
    for img_array in images_np:
        pil_image = Image.fromarray(img_array)
        if pil_image.mode == "RGBA":
            background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
            pil_image = Image.alpha_composite(background, pil_image)
        pil_image = pil_image.convert("RGB")

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
        arr = np.pad(
            arr,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=1.0,
        )
        processed.append(arr)

    return np.stack(processed, axis=0)


def _normalize_pointmap_batch(pointmaps: np.ndarray, keep_conf: bool = True) -> np.ndarray:
    if pointmaps.ndim != 5:
        return pointmaps
    x = pointmaps.astype(np.float32)
    if keep_conf and x.shape[-1] >= 4:
        xyz = x[..., :3]
        conf = x[..., 3:4]
        mean = np.nanmean(xyz, axis=(-3, -2), keepdims=True)
        std = np.nanstd(xyz, axis=(-3, -2), keepdims=True) + 1e-6
        xyz = (xyz - mean) / std
        return np.concatenate([xyz, conf], axis=-1)
    mean = np.nanmean(x, axis=(-3, -2), keepdims=True)
    std = np.nanstd(x, axis=(-3, -2), keepdims=True) + 1e-6
    return (x - mean) / std


def _compute_pointmap_buffers(
    image: np.ndarray,
    runner: "VGGTPointmapRunner",
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    img_bt = image[np.newaxis, np.newaxis, ...].astype(np.uint8)
    pm_bt = runner.compute_pointmap(img_bt)
    pm_raw = pm_bt[0, 0]
    pm_norm = pm_raw if not normalize else _normalize_pointmap_batch(pm_bt)[0, 0]
    return {
        "raw": pm_raw.astype(np.float32),
        "normalized": pm_norm.astype(np.float32),
    }


class VGGTPointmapRunner:
    """Compute VGGT world-point maps (xyz + confidence) online."""

    def __init__(
        self,
        device: torch.device,
        input_res: int,
        batch_size: int = 2,
    ):
        self.device = device
        self.input_res = int(input_res)
        self.batch_size = max(1, int(batch_size))
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def compute_pointmap(self, images_bt3hw: np.ndarray) -> np.ndarray:
        """
        Args:
            images_bt3hw: (B, T, H, W, 3) uint8/float32 images in [0,255].
        Returns:
            np.ndarray with shape (B, T, H', W', 4) containing xyz + confidence.
        """
        b, t, h, w, c = images_bt3hw.shape
        flat = images_bt3hw.reshape(b * t, h, w, c).astype(np.uint8)
        chw = _preprocess_images_for_vggt(flat, self.input_res)
        outputs = []
        amp_ctx = torch.cuda.amp.autocast(enabled=self.device.type == "cuda")

        for start in range(0, chw.shape[0], self.batch_size):
            batch = torch.from_numpy(chw[start : start + self.batch_size]).to(self.device)
            batch = batch.unsqueeze(1)  # (k,1,3,H,W)
            with amp_ctx:
                preds = self.model(batch)
            if isinstance(preds, dict) and "world_points" in preds:
                pts = preds["world_points"][:, 0]  # (k,H,W,3)
                conf = preds.get("world_points_conf")
                if conf is not None:
                    conf = conf[:, 0][..., None]
                else:
                    conf = torch.ones((*pts.shape[:3], 1), device=pts.device)
                out = torch.cat([pts, conf], dim=-1)
            elif isinstance(preds, dict) and "depth" in preds:
                depth = preds["depth"][:, 0, ..., 0][..., None]
                conf = preds.get("depth_conf")
                if conf is not None:
                    conf = conf[:, 0][..., None]
                else:
                    conf = torch.ones_like(depth)
                zeros = torch.zeros_like(depth)
                out = torch.cat([zeros, zeros, depth, conf], dim=-1)
            else:
                raise RuntimeError("VGGT did not return point/depth predictions.")
            outputs.append(out.detach().cpu().numpy())

        stacked = np.concatenate(outputs, axis=0)
        return stacked.reshape(b, t, stacked.shape[1], stacked.shape[2], stacked.shape[3]).astype(np.float32)


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    vggt_ctx: Optional[dict] = None,
    episode_idx: int = 0,
    snapshot_manager: Optional[AttentionSnapshotManager] = None,
):
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Try to infer control frequency for timing calculations
    control_freq = getattr(env, "control_freq", None)
    if control_freq is None and hasattr(env, "env"):
        control_freq = getattr(env.env, "control_freq", None)
    if control_freq is None and hasattr(env, "wrapped_env"):
        control_freq = getattr(env.wrapped_env, "control_freq", None)

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs)

            compressed_tokens = None
            if cfg.model_family == "octo" and cfg.use_vggt_tokens and vggt_ctx is not None:
                try:
                    compressed_tokens = compute_compressed_vggt_tokens_torch(img, vggt_ctx)
                    observation["vggt_tokens"] = compressed_tokens
                except Exception as _e:
                    log_message(f"[VGGT] Failed to compute tokens at t={t}: {_e}", log_file)

            replay_images.append(img)

            if len(action_queue) == 0:
                policy_step_idx = max(0, t - cfg.num_steps_wait)
                capture_payload = None
                capture_now = (
                    snapshot_manager is not None
                    and cfg.model_family == "octo"
                    and snapshot_manager.should_capture(episode_idx, policy_step_idx, control_freq)
                )
                if capture_now:
                    spec = snapshot_manager.build_capture_spec()
                    if cfg.attention_snapshot_request_pointmap_debug:
                        spec["pointmap_options"] = {
                            "vggt_input_res": cfg.vggt_input_res,
                            "vggt_eval_batch_size": cfg.vggt_eval_batch_size,
                            "vggt_use_cuda": cfg.vggt_use_cuda,
                            "vggt_device_id": cfg.vggt_device_id,
                            "normalize_pointmap": cfg.normalize_pointmap,
                        }
                    actions_output = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                        capture_spec=spec,
                    )
                    if isinstance(actions_output, tuple) and len(actions_output) == 2:
                        actions, capture_payload = actions_output
                    else:
                        actions = actions_output
                        capture_payload = None
                    if compressed_tokens is not None:
                        if capture_payload is None:
                            capture_payload = {}
                        capture_payload["vggt_tokens"] = compressed_tokens
                    snapshot_manager.commit(
                        payload=capture_payload,
                        image=img,
                        episode_idx=episode_idx,
                        control_freq=control_freq,
                    )
                else:
                    actions = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family, action_mean, action_std)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    vggt_ctx: Optional[dict] = None,
) -> Tuple[int, int, bool]:
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, cfg.env_img_res)

    def _snapshot_log(message: str) -> None:
        log_message(message, log_file)

    snapshot_manager = AttentionSnapshotManager(cfg, task_description, _snapshot_log)

    if cfg.capture_attention_snapshot and not snapshot_manager.task_match:
        log_message(
            f"Skipping task '{task_description}' because it does not match attention_snapshot_task.",
            log_file,
        )
        return total_episodes, total_successes, False

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            vggt_ctx=vggt_ctx,
            episode_idx=episode_idx,
            snapshot_manager=snapshot_manager,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

        if snapshot_manager.captured:
            log_message("Attention snapshot captured; ending task early.", log_file)
            break

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes, snapshot_manager.captured


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # =========================================================================
    # --- PART 1: SETUP PyTorch VGGT + AE ---
    # =========================================================================
    vggt_ctx: Optional[dict] = None
    if cfg.model_family == "octo" and cfg.use_vggt_tokens:
        if cfg.vggt_ae_path is None or not os.path.exists(cfg.vggt_ae_path):
            print("[VGGT] AE path missing or not found; continuing without VGGT tokens")
        else:
            try:
                device = torch.device('cuda' if (cfg.vggt_use_cuda and torch.cuda.is_available()) else 'cpu')
                if device.type == 'cuda':
                    torch.backends.cudnn.benchmark = True
                # Determine layer indices if subset
                layer_indices = _parse_layer_indices(cfg.vggt_layer_indices) if cfg.vggt_agg_layers < 24 else None
                extractor = TorchVGGTExtractor(device, cfg.vggt_input_res, cfg.vggt_agg_layers, layer_indices)

                # Probe L, D with a dummy white image
                dummy = np.ones((1, 3, cfg.vggt_input_res, cfg.vggt_input_res), dtype=np.float32)
                klnd, _ = extractor.extract_layers(dummy)
                L = klnd.shape[1]
                D = klnd.shape[3]

                compressor = AECompressor(num_layers=L, input_dim=D, bottleneck_dim=512, hidden_dim=8192)
                compressor.load_state_dict(torch.load(cfg.vggt_ae_path, map_location='cpu'))
                compressor = compressor.to(device).eval()

                vggt_ctx = {
                    "extractor": extractor,
                    "compressor": compressor,
                    "cfg": cfg,
                    "device": device,
                }
                print("Loaded PyTorch VGGT + AE compressor for online tokens.")
            except Exception as e:
                print(f"[VGGT] Failed to initialize PyTorch VGGT+AE: {e}; continuing without VGGT tokens")
                vggt_ctx = None

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    try:
        mcfg = model.config
        obs_tok = mcfg.get("model", {}).get("observation_tokenizers") or mcfg.get("observation_tokenizers")
        heads = mcfg.get("heads") or mcfg.get("model", {}).get("heads")
        act_dim = None
        if isinstance(heads, dict):
            act = heads.get("action", {})
            if isinstance(act, dict):
                act_dim = act.get("dim") or act.get("readout_dim")
    except Exception as e:
        print("[DEBUG] config introspection error:", e)

    log_file, local_log_filepath, run_id = setup_logging(cfg)

    log_message("\n" + "="*50, log_file)
    log_message("EVALUATION CONFIGURATION", log_file)
    log_message(f"  Model Path:      {cfg.pretrained_checkpoint}", log_file)
    log_message(f"  Checkpoint Step: {cfg.checkpoint_step}", log_file)
    log_message("="*50 + "\n", log_file)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes, snapshot_captured = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            vggt_ctx=vggt_ctx,
        )
        if snapshot_captured:
            log_message("Attention snapshot captured; stopping further task evaluation.", log_file)
            break

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()