import sys
import warnings
import json
import numpy as np

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
stats_path = "/home/pkarageorgis/geo_octo/libero_datasets/unified_stats/unified_dataset_statistics_libero_spatial_no_vggt.json"
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
from typing import Optional, Union

import draccus
import tqdm
from libero.libero import benchmark

import wandb


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

# New imports for VGGT ONNX + compression
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image
# Ensure we can import compressor saved as top-level module `vggt_compression_analysis`
sys.path.append("/workspace/octo/scripts")
from vggt_compression_analysis import VGGTCompressor


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

# After logging.basicConfig(...)

# 1) Silence noisy libraries
for name in ("octo", "octo.octo", "octo.utils", "octo.model", "flax", "transformers"):
    lg = logging.getLogger(name)
    lg.setLevel(logging.ERROR)
    lg.propagate = False  # prevent bubbling to root

# 2) Drop specific spam lines at root
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

# 3) Reduce absl (used by JAX/Flax) verbosity
try:
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# 4) Route warnings -> logging so they can be filtered
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
    # VGGT-specific parameters
    #################################################################################################################
    use_vggt_tokens: bool = True                     # Whether to compute and feed VGGT tokens online
    vggt_onnx_path: str = "/home/pkarageorgis/vggt_onxx/vggt_fp16.onnx"  # Path to VGGT ONNX model
    vggt_input_res: int = 224                        # Input resolution for VGGT model
    vggt_use_cuda: bool = True                       # Whether to use CUDAExecutionProvider when available
    vggt_compressor_path: Optional[str] = None       # Path to saved VGGTCompressor .pkl
    vggt_output_name: Optional[str] = None           # Specific ONNX output name to use (if exported)
    vggt_output_index: int = 0                       # Fallback output index when name not given
    vggt_raw_tokens: int = 261                       # Expected number of VGGT tokens before compression
    vggt_raw_dim: int = 2048                         # Expected token embedding dimension before compression

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    # Only applicable for OpenVLA
    if cfg.model_family == "openvla" and cfg.use_proprio:
        from evaluation.supporting_files.openvla_utils import get_proprio_projector

        proprio_projector = get_proprio_projector(
            cfg,
            getattr(model, "llm_dim", None),
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    # Only applicable for OpenVLA
    if cfg.model_family == "openvla" and (cfg.use_l1_regression or cfg.use_diffusion):
        from evaluation.supporting_files.openvla_utils import get_action_head

        action_head = get_action_head(cfg, getattr(model, "llm_dim", None))

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    # Only applicable for OpenVLA
    if cfg.model_family == "openvla" and cfg.use_diffusion:
        from evaluation.supporting_files.openvla_utils import get_noisy_action_projector

        noisy_action_projector = get_noisy_action_projector(cfg, getattr(model, "llm_dim", None))

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        from evaluation.supporting_files.openvla_utils import get_processor

        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Prepare observations dict
    observation = {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]) 
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family, action_mean=None, action_std=None):
    """Process action before sending to environment."""
    # For OpenVLA, the dataset gripper is [0,1] so normalize and invert
    if model_family == "openvla":
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
    elif model_family == "octo":
        if action_mean is None or action_std is None:
            raise ValueError("Action statistics (mean, std) must be provided for Octo model evaluation!")
            
        # The model outputs a normalized action. Let's un-normalize it.
        # Make sure the shapes match. The stats are for 7-dim actions.
        action_mean = action_mean[:action.shape[-1]]
        action_std = action_std[:action.shape[-1]]
        
        unnormalized_action = (action * action_std) + action_mean
        return unnormalized_action
    
    else:
        # Fallback for other models if needed
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


# ===== VGGT helpers =====

def _resize_to_vggt(image: np.ndarray, size: int) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if cv2 is not None:
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    else:
        resized = np.array(Image.fromarray(image).resize((size, size), Image.BICUBIC))
    return resized


def _prepare_onnx_input(image: np.ndarray, size: int) -> np.ndarray:
    # Expect HxWxC RGB uint8; normalize to [0,1] float32; CHW; add (B=1, V=1)
    resized = _resize_to_vggt(image, size)
    arr = resized.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    batched = chw[None, None, ...]  # (1, 1, 3, H, W)
    return batched


def _extract_tokens_from_outputs(outputs: list, cfg: GenerateConfig) -> np.ndarray:
    # If a specific output name/index was provided, select accordingly
    # Here `outputs` is a list as returned by session.run
    if cfg.vggt_output_name is not None:
        # session.run with output_names ensures ordering; user must ensure correct name order outside
        # We cannot map name->array here, so rely on index instead when names are specified during session.run
        pass
    # Choose output by index if provided
    if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        idx = int(cfg.vggt_output_index) if 0 <= int(cfg.vggt_output_index) < len(outputs) else 0
        chosen = outputs[idx]
        arr = np.asarray(chosen)
    else:
        # Fallback: pick the largest dimensional candidate
        candidates = []
        for out in outputs:
            try:
                candidates.append(np.asarray(out))
            except Exception:
                continue
        if not candidates:
            raise RuntimeError("ONNX outputs did not contain any ndarray candidates")
        candidates.sort(key=lambda x: (x.ndim, x.size), reverse=True)
        arr = candidates[0]
    # Squeeze batch/view dims
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    return arr


def compute_compressed_vggt_tokens(image: np.ndarray, vggt_ctx: dict) -> Optional[np.ndarray]:
    if vggt_ctx is None:
        return None
    session = vggt_ctx.get("session")
    input_name = vggt_ctx.get("input_name")
    output_names = vggt_ctx.get("output_names")
    input_res = vggt_ctx.get("input_res", 224)
    compressor: Optional[VGGTCompressor] = vggt_ctx.get("compressor")
    cfg: GenerateConfig = vggt_ctx.get("cfg")
    if session is None or input_name is None:
        return None
    x = _prepare_onnx_input(image, input_res)
    # If user specified output name, pass that to session.run to ensure ordering
    outs = output_names if (cfg and cfg.vggt_output_name is None) else output_names
    outputs = session.run(outs or None, {input_name: x})
    tokens = _extract_tokens_from_outputs(outputs, cfg)  # e.g., (L, D)
    # Reshape to expected raw shape before compression
    num_tokens = (cfg.vggt_raw_tokens if cfg else 261)
    token_dim = (cfg.vggt_raw_dim if cfg else 2048)
    tokens = tokens.reshape(num_tokens, token_dim)
    tokens_batched = tokens.reshape(1, num_tokens, token_dim)
    if compressor is None:
        raise RuntimeError("VGGT compressor not loaded but use_vggt_tokens=True; provide vggt_compressor_path")
    compressed = compressor.compress(tokens_batched)  # (1, H, W)
    return compressed[0]


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
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs)

            # If enabled, compute VGGT tokens online and attach to observation
            if cfg.model_family == "octo" and cfg.use_vggt_tokens and vggt_ctx is not None:
                try:
                    compressed_tokens = compute_compressed_vggt_tokens(img, vggt_ctx)
                    observation["vggt_tokens"] = compressed_tokens  # (H, W)
                except Exception as _e:
                    # Log once per episode if VGGT fails, and continue without VGGT tokens
                    log_message(f"[VGGT] Failed to compute tokens at t={t}: {_e}", log_file)

            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
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

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family, action_mean, action_std)

            # Execute action in environment
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
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
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
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    # =========================================================================
    # --- PART 1: SETUP ONNX RUNTIME FOR VGGT ---
    # =========================================================================
    vggt_ctx: Optional[dict] = None
    if cfg.model_family == "octo" and cfg.use_vggt_tokens:
        if ort is None:
            print("[VGGT] onnxruntime not available; continuing without VGGT tokens")
        elif not os.path.exists(cfg.vggt_onnx_path):
            print(f"[VGGT] ONNX model not found at {cfg.vggt_onnx_path}; continuing without VGGT tokens")
        elif cfg.vggt_compressor_path is None or not os.path.exists(cfg.vggt_compressor_path):
            print("[VGGT] Compressor path missing or not found; continuing without VGGT tokens")
        else:
            print("Loading ONNX VGGT model for fast inference...")
            providers = ['CUDAExecutionProvider'] if cfg.vggt_use_cuda else ['CPUExecutionProvider']
            try:
                session = ort.InferenceSession(cfg.vggt_onnx_path, providers=providers)
                input_name = session.get_inputs()[0].name
                output_names = [output.name for output in session.get_outputs()]
                print(f"ONNX VGGT model loaded. Input: '{input_name}', Outputs: {output_names}")
                compressor = VGGTCompressor.load_compressor(cfg.vggt_compressor_path)
                vggt_ctx = {
                    "session": session,
                    "input_name": input_name,
                    "output_names": output_names,
                    "input_res": cfg.vggt_input_res,
                    "compressor": compressor,
                    "cfg": cfg,
                }
            except Exception as e:
                print(f"[VGGT] Failed to initialize ONNX session: {e}; continuing without VGGT tokens")
                vggt_ctx = None

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    
    try:
        mcfg = model.config  # <- don't overwrite cfg
        print("[DEBUG] top keys:", list(mcfg.keys()))
        obs_tok = mcfg.get("model", {}).get("observation_tokenizers") or mcfg.get("observation_tokenizers")
        print("[DEBUG] obs tokenizers:", list((obs_tok or {}).keys()))
        heads = mcfg.get("heads") or mcfg.get("model", {}).get("heads")
        act_dim = None
        if isinstance(heads, dict):
            act = heads.get("action", {})
            if isinstance(act, dict):
                act_dim = act.get("dim") or act.get("readout_dim")
        print("[DEBUG] action head dim (from config):", act_dim)
    except Exception as e:
        print("[DEBUG] config introspection error:", e)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
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

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()