import os
import random
import time
from typing import Any, Dict, List, Optional, Union
from collections import deque

import numpy as np
import torch

# Avoid importing OpenVLA utilities at module import time to keep Octo-only usage light.

# Initialize important constants
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Model image size configuration
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    "octo": 256,  # fallback only; actual size inferred from model.example_batch when available
}

# Maintain short histories to provide a true 2-frame window without changing callers
IMAGE_HISTORY: deque = deque(maxlen=2)
PROPRIO_HISTORY: deque = deque(maxlen=2)
# Add VGGT token history for windowed inputs
VGGT_HISTORY: deque = deque(maxlen=2)
# Add PointMap history for windowed inputs
POINTMAP_HISTORY: deque = deque(maxlen=2)


def _extract_vision_tokens_for_snapshot(
    model: torch.nn.Module,
    observation: Dict[str, Any],
    task: Dict[str, Any],
    timestep_index: int,
    capture_spec: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Optional[np.ndarray]]]:
    """
    Extract intermediate vision tokens from the Octo model for analysis snapshots.
    """
    try:
        from flax.core import freeze
        import jax
        from octo.model.components.tokenizers import VisionMixer
    except Exception as exc:
        print(f"[SNAPSHOT] Missing dependencies for token extraction: {exc}", flush=True)
        return None

    if not hasattr(model, "module") or not hasattr(model.module, "octo_transformer"):
        return None

    try:
        bound_module = model.module.bind({"params": model.params})
    except Exception as exc:
        print(f"[SNAPSHOT] Failed to bind Octo module: {exc}", flush=True)
        return None

    transformer = getattr(bound_module, "octo_transformer", None)
    if transformer is None or not hasattr(transformer, "observation_tokenizers"):
        return None

    obs_tokenizers = transformer.observation_tokenizers
    obs_frozen = freeze(observation)
    task_frozen = freeze(task)

    octo_tokens = None
    vggt_tokens = None

    for tokenizer in obs_tokenizers.values():
        if isinstance(tokenizer, VisionMixer):
            try:
                patch_group = tokenizer.patch_tokenizer(obs_frozen, task_frozen, train=False)
            except Exception:
                patch_group = None
            if patch_group is not None:
                octo_tokens = np.asarray(jax.device_get(patch_group.tokens))
            try:
                vg_group = tokenizer.vggt_tokenizer(obs_frozen, task_frozen, train=False)
            except Exception:
                vg_group = None
            if vg_group is not None:
                vggt_tokens = np.asarray(jax.device_get(vg_group.tokens))
            break

    if octo_tokens is None:
        for tokenizer in obs_tokenizers.values():
            try:
                group = tokenizer(obs_frozen, task_frozen, train=False)
            except Exception:
                continue
            if group is None:
                continue
            try:
                octo_tokens = np.asarray(jax.device_get(group.tokens))
                break
            except Exception:
                continue

    def _select(tokens: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if tokens is None:
            return None
        if tokens.ndim < 3:
            return tokens.astype(np.float32)
        # Expect shape (B, T, N, D)
        batch_idx = min(0, tokens.shape[0] - 1)
        tdim = tokens.shape[1] if tokens.ndim >= 4 else 1
        if tdim == 0:
            return None
        idx = timestep_index
        if idx < 0:
            idx = tdim + idx
        idx = max(0, min(idx, tdim - 1))
        if tokens.ndim == 4:
            return tokens[batch_idx, idx].astype(np.float32)
        if tokens.ndim == 3:
            return tokens[batch_idx].astype(np.float32)
        return tokens.astype(np.float32)

    payload: Dict[str, Optional[np.ndarray]] = {
        "octo_tokens": _select(octo_tokens),
        "vggt_tokens": _select(vggt_tokens),
    }

    if capture_spec and capture_spec.get("request_pointmap_debug"):
        debug_tokens = _extract_pointmap_readout_tokens(
            model,
            observation,
            task,
            timestep_index=timestep_index,
        )
        if debug_tokens is not None:
            payload.update(debug_tokens)

    return payload


def _extract_pointmap_readout_tokens(
    model: torch.nn.Module,
    observation: Dict[str, Any],
    task: Dict[str, Any],
    timestep_index: int,
) -> Optional[Dict[str, Optional[np.ndarray]]]:
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        print(f"[SNAPSHOT] Unable to import JAX for pointmap debug extraction: {exc}", flush=True)
        return None

    def _to_jax(x: Any):
        if isinstance(x, np.ndarray):
            return jnp.asarray(x)
        if isinstance(x, (list, tuple)):
            return jnp.asarray(x)
        if isinstance(x, dict):
            return {k: _to_jax(v) for k, v in x.items()}
        return x

    obs_jax = _to_jax(observation)
    task_jax = _to_jax(task)

    try:
        _, debug_vars = model.module.apply(
            {"params": model.params},
            obs_jax,
            task_jax,
            obs_jax["timestep_pad_mask"],
            train=False,
            method=model.module.octo_transformer,
            mutable=["debug"],
        )
    except Exception as exc:
        print(f"[SNAPSHOT] Failed to capture pointmap debug tokens: {exc}", flush=True)
        return None

    debug_collection = debug_vars.get("debug", {})
    if not debug_collection:
        return None

    def _extract_named(name_suffix: str) -> Optional[np.ndarray]:
        for key, value in debug_collection.items():
            if key.endswith(name_suffix):
                arr = np.asarray(jax.device_get(value))
                if arr.ndim >= 2:
                    b = min(arr.shape[0] - 1, 0)
                    t = min(max(timestep_index, 0), arr.shape[1] - 1) if arr.ndim >= 3 else 0
                    if arr.ndim == 4:
                        return arr[b, t].astype(np.float32)
                    if arr.ndim == 3:
                        return arr[b, t].astype(np.float32)
                    if arr.ndim == 2:
                        return arr[b].astype(np.float32)
                return arr.astype(np.float32)
        return None

    pre_tokens = _extract_named("_readout_pre_pointmap")
    post_tokens = _extract_named("_readout_post_pointmap")
    pointmap_embed = _extract_named("_pointmap_embed")

    if pre_tokens is None and post_tokens is None and pointmap_embed is None:
        return None

    return {
        "pointmap_readout_pre_pointmap_tokens": pre_tokens,
        "pointmap_readout_post_pointmap_tokens": post_tokens,
        "pointmap_embed_tokens": pointmap_embed,
    }


def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg: Any, wrap_diffusion_policy_for_droid: bool = False) -> Any:
    """
    Load and initialize model for evaluation based on configuration.

    Args:
        cfg: Configuration object with model parameters
        wrap_diffusion_policy_for_droid: Whether to wrap diffusion policy for DROID

    Returns:
        torch.nn.Module: The loaded model

    Raises:
        ValueError: If model family is not supported
    """
    # Support both dataclass-like and dict cfg
    model_family = cfg.get("model_family") if isinstance(cfg, dict) else getattr(cfg, "model_family", None)
    if model_family == "openvla":
        from evaluation.supporting_files.openvla_utils import get_vla

        model = get_vla(cfg)
    elif model_family == "octo":
        # Lazy import to avoid hard dependency if unused
        from octo.model.octo_model import OctoModel

        pretrained_checkpoint = cfg.get("pretrained_checkpoint") if isinstance(cfg, dict) else getattr(cfg, "pretrained_checkpoint", None)
        checkpoint_step = cfg.get("checkpoint_step") if isinstance(cfg, dict) else getattr(cfg, "checkpoint_step", None)
        if checkpoint_step is not None:
            model = OctoModel.load_pretrained(str(pretrained_checkpoint), step=int(checkpoint_step))
        else:
            model = OctoModel.load_pretrained(str(pretrained_checkpoint))
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg: Any, model: Optional[Any] = None) -> Union[int, tuple]:
    """
    Get image resize dimensions for a specific model.

    If returned value is an int, the resized image will be a square.
    If returned value is a tuple, the resized image will be a rectangle.

    Args:
        cfg: Configuration object with model parameters
        model: Optional loaded model to introspect exact expected HxW

    Returns:
        Union[int, tuple]: Image resize dimensions

    Raises:
        ValueError: If model family is not supported
    """
    # Support both dataclass-like and dict cfg
    model_family = cfg.get("model_family") if isinstance(cfg, dict) else getattr(cfg, "model_family", None)
    if model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {model_family}")

    # If we have a loaded Octo model, infer exact expected size from example_batch
    if model_family == "octo" and model is not None:
        try:
            ex = model.example_batch["observation"].get("image_primary")
            if ex is not None and ex.ndim >= 5:
                # (B, T, H, W, C)
                return int(ex.shape[2]), int(ex.shape[3])
        except Exception:
            pass
        # fallback to 256 if present in config, else default mapping
        try:
            # Some finetune scripts add top-level window_size; image size lives in tokenizer but not always present
            size = int(MODEL_IMAGE_SIZES.get("octo", 224))
            return size
        except Exception:
            return MODEL_IMAGE_SIZES[model_family]

    return MODEL_IMAGE_SIZES[model_family]


def resize_image_for_policy(image: np.ndarray, resize_size: Union[int, tuple]) -> np.ndarray:
    """
    Resize an HxWxC image to the expected policy input resolution.

    Uses OpenCV if available; falls back to NumPy/PIL-style nearest if not.
    """
    try:
        import cv2  # type: ignore

        if isinstance(resize_size, int):
            target = (resize_size, resize_size)
        else:
            target = (int(resize_size[1]), int(resize_size[0])) if len(resize_size) == 2 else tuple(resize_size)
        return cv2.resize(image, target, interpolation=cv2.INTER_LINEAR)
    except Exception:
        # Fallback: simple PIL resize
        from PIL import Image

        if isinstance(resize_size, int):
            target = (resize_size, resize_size)
        else:
            target = (int(resize_size[1]), int(resize_size[0])) if len(resize_size) == 2 else tuple(resize_size)
        return np.array(Image.fromarray(image).resize(target, Image.BILINEAR))


def get_action(
    cfg: Any,
    model: torch.nn.Module,
    obs: Dict[str, Any],
    task_label: str,
    processor: Optional[Any] = None,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    capture_spec: Optional[Dict[str, Any]] = None,
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Query the model to get action predictions.

    Args:
        cfg: Configuration object with model parameters
        model: The loaded model
        obs: Observation dictionary
        task_label: Text description of the task
        processor: Model processor for inputs
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        List[np.ndarray] or tuple: Predicted actions. When ``capture_spec`` is provided,
        returns ``(actions, payload)`` where ``payload`` contains intermediate vision tokens.

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family == "openvla":
        from evaluation.supporting_files.openvla_utils import get_vla_action

        with torch.no_grad():
            action = get_vla_action(
                cfg=cfg,
                vla=model,
                processor=processor,
                obs=obs,
                task_label=task_label,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=use_film,
            )
        return (action, None) if capture_spec is not None else action
    elif cfg.model_family == "octo":
        # Build Octo observation and sample a single-step action
        import jax

        # Infer window and whether the checkpoint expects images
        expected_window = 1
        try:
            ex_obs = model.example_batch["observation"]
            expected_window = int(ex_obs["timestep_pad_mask"].shape[1])
        except Exception:
            ex_obs = {}
        target_h, target_w = None, None
        try:
            ex_img = ex_obs.get("image_primary")
            if ex_img is not None and ex_img.ndim >= 5:
                target_h, target_w = int(ex_img.shape[2]), int(ex_img.shape[3])
        except Exception:
            pass

        # VGGT-only if checkpoint has no image_primary OR user forces it
        force_vggt_only = bool(getattr(cfg, "vggt_only_eval", False))
        expect_images = ("image_primary" in ex_obs) and (not force_vggt_only)
        
        # Print once, flush to bypass buffering
        if not hasattr(get_action, "_printed"):
            print(f"[EVAL] expect_images={expect_images}, vggt_only_eval={force_vggt_only}, has_img_in_ckpt={'image_primary' in ex_obs}", flush=True)
            get_action._printed = True

        # Maintain histories sized to the model’s expected window
        from collections import deque
        global IMAGE_HISTORY, PROPRIO_HISTORY, VGGT_HISTORY, POINTMAP_HISTORY
        if getattr(IMAGE_HISTORY, "maxlen", None) != expected_window:
            IMAGE_HISTORY = deque(list(IMAGE_HISTORY), maxlen=expected_window)
        if getattr(PROPRIO_HISTORY, "maxlen", None) != expected_window:
            PROPRIO_HISTORY = deque(list(PROPRIO_HISTORY), maxlen=expected_window)
        if getattr(VGGT_HISTORY, "maxlen", None) != expected_window:
            VGGT_HISTORY = deque(list(VGGT_HISTORY), maxlen=expected_window)
        if getattr(POINTMAP_HISTORY, "maxlen", None) != expected_window:
            POINTMAP_HISTORY = deque(list(POINTMAP_HISTORY), maxlen=expected_window)

        # Images (only if expected)
        image = obs["full_image"]
        if expect_images:
            IMAGE_HISTORY.append(image)
            while len(IMAGE_HISTORY) < expected_window:
                IMAGE_HISTORY.append(image)
            image_stack = np.stack(list(IMAGE_HISTORY), axis=0)  # (T, H, W, 3)

        # Proprio (7-D)
        proprio_dim = 7
        if "state" in obs and obs["state"] is not None:
            state_vec = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
            state_vec = state_vec[:proprio_dim] if state_vec.shape[-1] >= proprio_dim else np.pad(state_vec, (0, proprio_dim - state_vec.shape[-1]))
            PROPRIO_HISTORY.append(state_vec)
        else:
            PROPRIO_HISTORY.append(np.zeros((proprio_dim,), dtype=np.float32))
        while len(PROPRIO_HISTORY) < expected_window:
            PROPRIO_HISTORY.append(PROPRIO_HISTORY[-1])
        proprio_stack = np.stack(list(PROPRIO_HISTORY), axis=0)  # (T, D)

        # Optional VGGT tokens
        vggt_stack = None
        if "vggt_tokens" in obs and obs["vggt_tokens"] is not None:
            current_tokens = np.asarray(obs["vggt_tokens"])  # (H, W)
            VGGT_HISTORY.append(current_tokens)
            while len(VGGT_HISTORY) < expected_window:
                VGGT_HISTORY.append(VGGT_HISTORY[-1])
            vggt_stack = np.stack(list(VGGT_HISTORY), axis=0)  # (T, H, W)

        # Optional PointMap
        pointmap_stack = None
        pm_key = getattr(cfg, "pointmap_key", "pointmap")
        if pm_key in obs and obs[pm_key] is not None:
            current_pm = np.asarray(obs[pm_key])  # (H, W, 4)
            if current_pm.ndim == 3 and current_pm.shape[-1] in (1, 3, 4):
                POINTMAP_HISTORY.append(current_pm)
                while len(POINTMAP_HISTORY) < expected_window:
                    POINTMAP_HISTORY.append(POINTMAP_HISTORY[-1])
                pointmap_stack = np.stack(list(POINTMAP_HISTORY), axis=0)  # (T, H, W, C)

        # Build observation dict to match the checkpoint schema
        T = expected_window
        observation = {
            "timestep": np.arange(T, dtype=np.int32)[np.newaxis, ...],
            "task_completed": np.zeros((1, T, 4), dtype=bool),
            "timestep_pad_mask": np.ones((1, T), dtype=bool),
            "pad_mask_dict": {
                "timestep": np.ones((1, T), dtype=bool),
                "proprio": np.ones((1, T), dtype=bool),
            },
            "proprio": proprio_stack[np.newaxis, ...],
        }

        # Conditionally add image_primary
        if expect_images:
            observation["image_primary"] = image_stack[np.newaxis, ...]
            observation["pad_mask_dict"]["image_primary"] = np.ones((1, T), dtype=bool)

        # Conditionally add vggt_tokens
        if vggt_stack is not None:
            observation["vggt_tokens"] = vggt_stack[np.newaxis, ...]  # (1, T, 64, 512) or (1, T, H, W)
            observation["pad_mask_dict"]["vggt_tokens"] = np.ones((1, T), dtype=bool)

        # Conditionally add pointmap
        if pointmap_stack is not None:
            observation[pm_key] = pointmap_stack[np.newaxis, ...]  # (1, T, H, W, C)
            # Pointmaps are readouts injected post-tokenizers (no tokenizer name to mask),
            # so we do NOT add to pad_mask_dict. OctoModule consumes this directly.

        # Construct task from text without touching its representation
        task = model.create_tasks(texts=[task_label])

        capture_payload: Optional[Dict[str, Optional[np.ndarray]]] = None
        if capture_spec is not None and capture_spec.get("request_tokens"):
            try:
                timestep_idx = int(capture_spec.get("timestep_index", T - 1))
            except Exception:
                timestep_idx = T - 1
            capture_payload = _extract_vision_tokens_for_snapshot(
                model,
                observation,
                task,
                timestep_index=timestep_idx,
                capture_spec=capture_spec,
            )

        # Build un-normalization stats from the checkpoint
        ds = getattr(model, "dataset_statistics", {}).get("action")
        unnorm_stats = None
        if ds is not None:
            action_mean = np.array(ds["mean"], dtype=np.float32)
            action_std = np.array(ds["std"], dtype=np.float32)
            unnorm_stats = {"mean": action_mean, "std": action_std}
            if "mask" in ds:
                unnorm_stats["mask"] = np.array(ds["mask"], dtype=bool)

        # Sample action; model applies un-normalization
        action = model.sample_actions(
            observation,
            task,
            unnormalization_statistics=unnorm_stats,
            rng=jax.random.PRNGKey(0),
        )
        # Convert to numpy and normalize output to a list of 7D steps
        arr = np.array(action)
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]

        steps: List[np.ndarray] = []
        if arr.ndim == 2:
            for i in range(arr.shape[0]):
                vec = np.asarray(arr[i]).ravel()
                if vec.size >= 7:
                    steps.append(vec[:7].astype(np.float32))
                else:
                    steps.append(np.pad(vec.astype(np.float32), (0, 7 - vec.size)))
        else:
            vec = np.asarray(arr).ravel()
            if vec.size >= 7:
                steps.append(vec[:7].astype(np.float32))
            else:
                steps.append(np.pad(vec.astype(np.float32), (0, 7 - vec.size)))

        if capture_spec is not None:
            return steps, capture_payload
        return steps
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action