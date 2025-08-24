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
        Union[List[np.ndarray], np.ndarray]: Predicted actions

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
        return action
    elif cfg.model_family == "octo":
        # Build Octo observation and sample a single-step action
        import jax

        # Infer expected window size and image resolution from the model's example batch
        expected_window = 1
        target_h, target_w = None, None
        try:
            ex = model.example_batch["observation"]["image_primary"]
            if ex.ndim >= 5:
                expected_window = int(ex.shape[1])
                target_h, target_w = int(ex.shape[2]), int(ex.shape[3])
        except Exception:
            pass
        if target_h is None or target_w is None:
            # Fallback to helper (may use default mapping)
            rs = get_image_resize_size(cfg, model)
            if isinstance(rs, int):
                target_h = target_w = int(rs)
            else:
                target_h, target_w = int(rs[0]), int(rs[1])

        # Ensure global histories match expected window size
        global IMAGE_HISTORY, PROPRIO_HISTORY
        if getattr(IMAGE_HISTORY, "maxlen", None) != expected_window:
            IMAGE_HISTORY = deque(list(IMAGE_HISTORY), maxlen=expected_window)
        if getattr(PROPRIO_HISTORY, "maxlen", None) != expected_window:
            PROPRIO_HISTORY = deque(list(PROPRIO_HISTORY), maxlen=expected_window)

        # Prepare image and resize to target
        image = obs["full_image"]
        if image.shape[0] != target_h or image.shape[1] != target_w:
            image = resize_image_for_policy(image, (target_h, target_w))

        # Update image history and build stack of length expected_window
        IMAGE_HISTORY.append(image)
        while len(IMAGE_HISTORY) < expected_window:
            # duplicate last frame until filled
            IMAGE_HISTORY.append(image)
        image_stack = np.stack(list(IMAGE_HISTORY), axis=0)  # (T, H, W, 3)

        # Optional proprioception (7-dim expected by this checkpoint unless otherwise specified)
        proprio_dim = 7
        if "state" in obs and obs["state"] is not None:
            state_vec = np.asarray(obs["state"], dtype=np.float32)
            if state_vec.shape[-1] < proprio_dim:
                pad = np.zeros((proprio_dim - state_vec.shape[-1],), dtype=np.float32)
                state_vec = np.concatenate([state_vec, pad], axis=-1)
            elif state_vec.shape[-1] > proprio_dim:
                state_vec = state_vec[:proprio_dim]
            PROPRIO_HISTORY.append(state_vec)
        else:
            PROPRIO_HISTORY.append(np.zeros((proprio_dim,), dtype=np.float32))
        while len(PROPRIO_HISTORY) < expected_window:
            PROPRIO_HISTORY.append(PROPRIO_HISTORY[-1])
        proprio_stack = np.stack(list(PROPRIO_HISTORY), axis=0)  # (T, D)

        # Build observation dict matching model.example_batch shapes
        observation = {
            "image_primary": image_stack[np.newaxis, ...],  # (1, T, H, W, 3)
            "timestep": np.arange(expected_window, dtype=np.int32)[np.newaxis, ...],
            # Shape (1, T, 4): binary flags for per-step done signals; we set all False
            "task_completed": np.zeros((1, expected_window, 4), dtype=bool),
            "timestep_pad_mask": np.ones((1, expected_window), dtype=bool),
            "pad_mask_dict": {
                "image_primary": np.ones((1, expected_window), dtype=bool),
                "timestep": np.ones((1, expected_window), dtype=bool),
                "proprio": np.ones((1, expected_window), dtype=bool),
            },
            "proprio": proprio_stack[np.newaxis, ...],  # (1, T, D)
        }

        # Task construction (language-conditioned). Use input_ids directly and add pad mask
        _raw_task = model.create_tasks(texts=[task_label])
        task = dict(_raw_task)
        ids = None
        lang = task.get("language_instruction")
        if isinstance(lang, dict) and "input_ids" in lang:
            ids = lang["input_ids"]
        elif "language_instruction/input_ids" in task:
            ids = task["language_instruction/input_ids"]
        elif isinstance(lang, (np.ndarray, list)):
            ids = lang
        if ids is not None:
            ids = np.asarray(ids, dtype=np.int32)
            task["language_instruction"] = ids
            # Build a pad mask for language if not present
            pad_mask = np.ones(ids.shape[:-1] if ids.ndim > 1 else (1,), dtype=bool)
            pad_dict = task.get("pad_mask_dict", {})
            pad_dict["language_instruction"] = pad_mask
            task["pad_mask_dict"] = pad_dict
            # Drop flattened extras to avoid conflicting paths
            task.pop("language_instruction/input_ids", None)
            task.pop("language_instruction/attention_mask", None)

        # Sample action
        action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))

        # Convert to numpy and squeeze leading singleton dims
        arr = np.array(action)
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]

        steps: List[np.ndarray] = []
        # Case: 2D array (horizon, dim)
        if arr.ndim == 2:
            horizon, dim = arr.shape
            if dim == 7:
                # Already 7D per step
                for i in range(horizon):
                    steps.append(arr[i].astype(np.float32))
            elif dim == 4:
                # Map each 4D step -> 7D by zero-filling rotations
                for i in range(horizon):
                    dx, dy, dz, grip = float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]), float(arr[i, 3])
                    steps.append(np.array([dx, dy, dz, 0.0, 0.0, 0.0, grip], dtype=np.float32))
            else:
                # Fallback: try to slice/pad to 7 per step
                for i in range(horizon):
                    vec = np.asarray(arr[i]).ravel()
                    if vec.size >= 7:
                        steps.append(vec[:7].astype(np.float32))
                    else:
                        pad = np.zeros((7 - vec.size,), dtype=np.float32)
                        steps.append(np.concatenate([vec.astype(np.float32), pad], axis=0))
        else:
            # Case: 1D vector
            vec = arr.ravel()
            if vec.size == 7:
                steps.append(vec.astype(np.float32))
            elif vec.size == 4:
                dx, dy, dz, grip = float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3])
                steps.append(np.array([dx, dy, dz, 0.0, 0.0, 0.0, grip], dtype=np.float32))
            else:
                if vec.size >= 7:
                    steps.append(vec[:7].astype(np.float32))
                else:
                    pad = np.zeros((7 - vec.size,), dtype=np.float32)
                    steps.append(np.concatenate([vec.astype(np.float32), pad], axis=0))

        # Return the full action chunk to be consumed open-loop
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