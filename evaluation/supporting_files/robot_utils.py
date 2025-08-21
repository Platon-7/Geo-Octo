"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

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
    "octo": 224,
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
    if cfg.model_family == "openvla":
        from evaluation.supporting_files.openvla_utils import get_vla

        model = get_vla(cfg)
    elif cfg.model_family == "octo":
        # Lazy import to avoid hard dependency if unused
        from octo.model.octo_model import OctoModel

        model = OctoModel.load_pretrained(str(cfg.pretrained_checkpoint))
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg: Any) -> Union[int, tuple]:
    """
    Get image resize dimensions for a specific model.

    If returned value is an int, the resized image will be a square.
    If returned value is a tuple, the resized image will be a rectangle.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        Union[int, tuple]: Image resize dimensions

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return MODEL_IMAGE_SIZES[cfg.model_family]


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

        # Prepare observation dict
        image = obs["full_image"]
        resize_size = MODEL_IMAGE_SIZES.get("octo", 224)
        if image.shape[0] != resize_size or image.shape[1] != resize_size:
            image = resize_image_for_policy(image, resize_size)

        # Stack to expected history length (2)
        image_stack = np.stack([image, image], axis=0)  # (2, H, W, 3)

        # Optional proprioception (7-dim expected by this checkpoint)
        proprio_dim = 7
        if "state" in obs and obs["state"] is not None:
            state_vec = np.asarray(obs["state"], dtype=np.float32)
            if state_vec.shape[-1] < proprio_dim:
                pad = np.zeros((proprio_dim - state_vec.shape[-1],), dtype=np.float32)
                state_vec = np.concatenate([state_vec, pad], axis=-1)
            elif state_vec.shape[-1] > proprio_dim:
                state_vec = state_vec[:proprio_dim]
            proprio_stack = np.stack([state_vec, state_vec], axis=0)  # (2, D)
        else:
            proprio_stack = np.zeros((2, proprio_dim), dtype=np.float32)

        observation = {
            "image_primary": image_stack[np.newaxis, ...],  # (1, 2, H, W, 3)
            "timestep": np.array([[0, 1]], dtype=np.int32),
            # Shape (1, 2, 4): binary flags for per-step done signals; we set all False
            "task_completed": np.zeros((1, 2, 4), dtype=bool),
            "timestep_pad_mask": np.array([[True, True]], dtype=bool),
            "pad_mask_dict": {
                "image_primary": np.array([[True, True]], dtype=bool),
                "timestep": np.array([[True, True]], dtype=bool),
                "proprio": np.array([[True, True]], dtype=bool),
            },
            "proprio": proprio_stack[np.newaxis, ...],  # (1, 2, D)
        }

        # Task construction (language-conditioned). Convert to plain token ids if needed.
        task = model.create_tasks(texts=[task_label])

        # Handle both nested and flattened representations from create_tasks
        # 1) Nested dict case: { 'language_instruction': {'input_ids': ..., 'attention_mask': ...}, ... }
        lang = task.get("language_instruction")
        if isinstance(lang, dict) and "input_ids" in lang:
            task["language_instruction"] = np.asarray(lang["input_ids"], dtype=np.int32)
        # 2) Flattened keys case: { 'language_instruction/input_ids': ..., 'language_instruction/attention_mask': ... }
        if "language_instruction" not in task and "language_instruction/input_ids" in task:
            task["language_instruction"] = np.asarray(task["language_instruction/input_ids"], dtype=np.int32)
        # Remove auxiliary fields not needed by sample_actions
        if "language_instruction/attention_mask" in task:
            try:
                del task["language_instruction/attention_mask"]
            except Exception:
                pass
        if "language_instruction/input_ids" in task:
            try:
                del task["language_instruction/input_ids"]
            except Exception:
                pass
        # Some create_tasks variants add image pad mask to tasks; drop to match example batch
        if "pad_mask_dict/image_primary" in task:
            try:
                del task["pad_mask_dict/image_primary"]
            except Exception:
                pass

        # Sample action
        action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))

        # Convert to numpy and squeeze batch
        action = np.array(action)[0]

        # Return as a list to match action chunk interface
        return [action]
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