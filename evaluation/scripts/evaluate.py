import sys
import warnings

# Add compatibility shim before importing anything else
try:
    import jax.numpy as jnp
    if not hasattr(jnp, 'DeviceArray'):
        jnp.DeviceArray = jnp.ndarray
        print("[FIX] Added DeviceArray compatibility shim")
except ImportError:
    print("[WARNING] Could not import JAX")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")

import os
import cv2
import numpy as np
import jax
import tensorflow as tf
from typing import Optional

# Disable tokenizer parallelism to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF warnings

# Octo / LIBERO Imports
from octo.model.octo_model import OctoModel
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_PATH = "/home/pkarageorgis/geo_octo/octo/my_octo_vggt_model_offline/octo_vggt_finetune_staged/experiment_20250808_130401_BASELINE_RUN"
TASK_SUITE_NAME = "libero_spatial"
EVAL_TASK_ID = 6
NUM_TIMESTEPS = 400
WINDOW_SIZE = 2
OUTPUT_DIR = "evaluation/test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BASE_DATA_DIR = "/scratch-shared/tmp.cwkV8vOvfY/libero_datasets"
DATASET_STATISTICS_KEY = "libero_spatial_no_noops"
LIBERO_DIR = "LIBERO"
# Ablation: one of {"multimodal", "image_conditioned", "language_conditioned"}
EVAL_MODE = "multimodal"

# Optional control ablations / safety
ZERO_ROTATION = True         # Zero out orientation deltas to test position-only control
MAP_GRIPPER_ABS_TO_REL = True  # Map gripper from [0,1] -> [-1,1]
LPF_ALPHA = 0.3             # Exponential moving average for action smoothing (0 disables if <=0)
MANUAL_CLAMP_IF_NO_SPACE = True
TRANS_MAX = 0.01            # meters per step cap if no env.action_space (tighter for stability)
ROT_MAX = 0.15              # rad per step cap if no env.action_space
GRIP_MAX = 1.0

# Axis/sign remap toggles
FLIP_X = True    # invert X if robot moves right instead of left
FLIP_Y = False
FLIP_Z = False

# Gripper handling
# If your model outputs relative [-1,1], set GRIPPER_MODE='rel'; if absolute [0,1], set 'abs'.
GRIPPER_MODE = 'rel'
GRIPPER_SIGN = -1.0   # invert if open/close seems reversed

# ==============================================================================
# Helper function to correctly load the goal image from TFRecord files
# ==============================================================================
def get_goal_image_from_tfrecord(dataset_dir, episode_index):
    """Parses a TFRecord file to extract the final image of a specific episode."""
    tfrecord_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if ".tfrecord" in f]
    if not tfrecord_files:
        raise FileNotFoundError(f"No .tfrecord files found in '{dataset_dir}'.")
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    try:
        episode_proto = next(iter(raw_dataset.skip(episode_index).take(1)))
    except tf.errors.OutOfRangeError:
        raise ValueError(f"episode_index {episode_index} is out of range for the dataset in {dataset_dir}") from None
    episode_feature_description = {
        'steps/observation/image': tf.io.VarLenFeature(tf.string),
    }
    parsed_episode = tf.io.parse_single_example(episode_proto, episode_feature_description)
    image_tensors = parsed_episode['steps/observation/image'].values
    goal_image = tf.io.decode_jpeg(image_tensors[-1]).numpy()
    return goal_image

# ==============================================================================
# Main evaluation logic
# ==============================================================================
print("="*50)
print("Corrected OCTO + LIBERO EVALUATION")
print("="*50)

# 1. Load Model
print(f"\n[INFO] Loading Octo model from: {MODEL_PATH}")
model = OctoModel.load_pretrained(MODEL_PATH)
print("[SUCCESS] Octo model loaded.\n")

# Print model spec and config summary
try:
    print("[DEBUG] Model spec:")
    print(model.get_pretty_spec())
    if "model" in model.config:
        mcfg = model.config["model"]
        print("[DEBUG] Observation tokenizers:", list(mcfg.get("observation_tokenizers", {}).keys()))
        print("[DEBUG] Task tokenizers:", list(mcfg.get("task_tokenizers", {}).keys()))
except Exception:
    pass

if DATASET_STATISTICS_KEY not in model.dataset_statistics:
    print(f"[ERROR] Statistics key '{DATASET_STATISTICS_KEY}' not found in model.")
    print(f"Available keys: {list(model.dataset_statistics.keys())}")
    exit()

try:
    # 2. Initialize LIBERO Environment
    print(f"[INFO] Accessing benchmark suite: {TASK_SUITE_NAME}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME]()
    task = task_suite.get_task(EVAL_TASK_ID)
    task_name = task.name
    language_instruction = task.language
    print(f"[SUCCESS] Retrieved task '{task_name}': '{language_instruction}'")

    bddl_file_path = os.path.join(LIBERO_DIR, "libero", "libero", "bddl_files", task.problem_folder, task.bddl_file)
    print(f"[INFO] Using BDDL file path: {bddl_file_path}")
    
    env_args = {"bddl_file_name": bddl_file_path, "camera_heights": 224, "camera_widths": 224}
    env = OffScreenRenderEnv(**env_args)

    # === Env action diagnostics and mapping ===
    try:
        env_action_dim = getattr(env.env, "action_dim", None)
    except Exception:
        env_action_dim = None
    print(f"[DEBUG] Env action_space: {getattr(env, 'action_space', None)}")
    print(f"[DEBUG] Env action_dim (if available): {env_action_dim}")

    def get_target_dim() -> int:
        aspace = getattr(env, "action_space", None)
        if aspace is None and hasattr(env, "env"):
            aspace = getattr(env.env, "action_space", None)
        if env_action_dim is not None:
            return int(env_action_dim)
        if aspace is not None and hasattr(aspace, "shape") and aspace.shape is not None:
            return int(aspace.shape[0])
        return 7

    def map_action_for_env(action7: np.ndarray, target_dim: Optional[int]):
        a = np.array(action7, dtype=np.float32)
        if target_dim == 4:
            # Map [dx, dy, dz, dRx, dRy, dRz, g] -> [dx, dy, dz, g]
            return np.array([a[0], a[1], a[2], a[6]], dtype=np.float32)
        if target_dim is None:
            aspace = getattr(env, "action_space", None)
            if aspace is None and hasattr(env, "env"):
                aspace = getattr(env.env, "action_space", None)
            if aspace is not None and hasattr(aspace, "shape"):
                return a[: int(aspace.shape[0])]
            if target_dim is not None:
                return a[:target_dim]
            return a
        # target_dim provided
        return a[:target_dim]

    def prepare_action(action7: np.ndarray) -> np.ndarray:
        td = get_target_dim()
        try:
            mapped = map_action_for_env(action7, td)
            arr = np.asarray(mapped, dtype=np.float32).reshape(-1)
        except Exception:
            arr = np.zeros(td, dtype=np.float32)
        # pad / trim
        if arr.shape[0] < td:
            arr = np.pad(arr, (0, td - arr.shape[0]))
        elif arr.shape[0] > td:
            arr = arr[:td]
        # finite check
        if not np.all(np.isfinite(arr)):
            arr = np.zeros(td, dtype=np.float32)
        return arr

    def get_eef_pos(obs_dict):
        return np.array(obs_dict.get("robot0_eef_pos", np.zeros(3, dtype=np.float32)))

    def probe_env_response(env_obj, steps=3, delta=0.03):
        print("[INFO] Probing env response to small action impulses...")
        try:
            base_obs, _, _, _ = env_obj.step(np.zeros(get_target_dim(), dtype=np.float32))
        except Exception:
            base_obs = {}
        base_eef = get_eef_pos(base_obs)
        # Build 7D basis; gripper as 1.0 toggle
        for i in range(7):
            v = np.zeros(7, dtype=np.float32)
            v[i] = delta if i < 6 else 1.0
            try:
                mapped = prepare_action(v)
                obs_before = base_obs
                eef_before = get_eef_pos(obs_before)
                last_obs = obs_before
                for _ in range(steps):
                    last_obs, _, _, _ = env_obj.step(mapped)
                eef_after = get_eef_pos(last_obs)
                print(f"[PROBE] Axis {i}: action={mapped} -> Δeef={eef_after - eef_before}")
            except Exception as e:
                print(f"[PROBE] Axis {i} probing failed: {e}")
                break
        print("[INFO] Probe complete.")

    # 3. Create Task - Using YOUR structure with the CORRECT goal image
    print(f"\n[INFO] Creating task specification...")
    try:
        dataset_dir_for_eval = os.path.join(BASE_DATA_DIR, DATASET_STATISTICS_KEY, "1.0.0")
        print(f"[INFO] Loading goal image from: {dataset_dir_for_eval}")
        goal_image = get_goal_image_from_tfrecord(dataset_dir_for_eval, EVAL_TASK_ID)
        goal_image_resized = cv2.resize(goal_image, (224, 224))
        print("[SUCCESS] Correct goal image loaded from demonstration.")
    except Exception as e:
        print(f"[FATAL] Could not load goal image: {e}")
        import traceback
        traceback.print_exc()
        exit()
    
    # Create tasks using both goal image and language so tokenization matches training
    if EVAL_MODE == "multimodal":
        tasks = model.create_tasks(goals={"image_primary": goal_image_resized[None]}, texts=[language_instruction])
    elif EVAL_MODE == "image_conditioned":
        tasks = model.create_tasks(goals={"image_primary": goal_image_resized[None]})
    elif EVAL_MODE == "language_conditioned":
        tasks = model.create_tasks(texts=[language_instruction])
    else:
        raise ValueError(f"Unknown EVAL_MODE: {EVAL_MODE}")
    # Ensure language is passed as numeric token ids like during finetuning
    if isinstance(tasks.get("language_instruction"), dict) and "input_ids" in tasks["language_instruction"]:
        tasks["language_instruction"] = np.asarray(tasks["language_instruction"]["input_ids"], dtype=np.int32)
    print(f"[SUCCESS] Task prompt created (mode={EVAL_MODE}).")

    # 4. Setup for Inference Loop
    print("\n[INFO] Setting up inference...")
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(EVAL_TASK_ID)
    env.set_init_state(init_states[0])
    obs, _, _, _ = env.step(np.zeros(7))

    # Probe once before the loop
    probe_env_response(env, steps=3, delta=0.03)

    def extract_proprio(obs_dict):
        proprio = obs_dict.get("robot0_joint_pos", np.zeros(7))
        return proprio[:7] if len(proprio) >= 7 else np.zeros(7)

    images = []
    proprios = []
    frames = []
    prev_action_exec = None

    print(f"[INFO] Starting evaluation loop with {WINDOW_SIZE}-frame window...")

    # 5. Inference Loop
    for step in range(NUM_TIMESTEPS):
        current_image = obs["agentview_image"]
        current_proprio = extract_proprio(obs)
        images.append(current_image)
        proprios.append(current_proprio)

        if len(images) > WINDOW_SIZE:
            images.pop(0)
            proprios.pop(0)

        if len(images) == WINDOW_SIZE:
            ### CHANGED ### - Restored your original, fully-detailed observation dictionary.
            # This directly addresses all previous errors (KeyError, AssertionError, and ScopeParamNotFoundError).
            observation = {
                'image_primary': np.stack(images)[None],
                'proprio': np.stack(proprios)[None],
                'timestep_pad_mask': np.full((1, WINDOW_SIZE), True, dtype=bool),
                'timestep': np.array([[step-(WINDOW_SIZE-1), step]], dtype=np.int32),
                'task_completed': np.zeros((1, WINDOW_SIZE, 4), dtype=bool),
                'pad_mask_dict': {
                    'image_primary': np.full((1, WINDOW_SIZE), True, dtype=bool),
                    'proprio': np.full((1, WINDOW_SIZE), True, dtype=bool),
                    'timestep': np.full((1, WINDOW_SIZE), True, dtype=bool),
                }
            }
            
            actions = model.sample_actions(
                observation,
                tasks,
                unnormalization_statistics=model.dataset_statistics[DATASET_STATISTICS_KEY]["action"],
                rng=jax.random.PRNGKey(step)
            )
            predicted_action = actions[0]
            action_to_execute = predicted_action[0] if predicted_action.ndim == 2 else predicted_action
        else:
            action_to_execute = np.zeros(7)

        # Control ablations / preprocessing
        a = np.array(action_to_execute, dtype=np.float32)
        if a.shape[0] < 7:
            a = np.pad(a, (0, 7 - a.shape[0]))
        if ZERO_ROTATION:
            a[3:6] = 0.0
        # Axis flips
        if FLIP_X:
            a[0] = -a[0]
        if FLIP_Y:
            a[1] = -a[1]
        if FLIP_Z:
            a[2] = -a[2]
        # Gripper mapping
        if GRIPPER_MODE == 'rel':
            # Ensure in [-1,1] and apply sign
            a[6] = np.clip(a[6], -1.0, 1.0) * GRIPPER_SIGN
        else:
            # Treat as absolute [0,1], with possible inversion
            g = np.clip(a[6], 0.0, 1.0)
            a[6] = (GRIPPER_SIGN * (g * 2.0 - 1.0))  # convert to rel for controller
        action_to_execute = a

        # Map to env action dimension and sanitize
        action_to_execute = prepare_action(action_to_execute)

        # Optional smoothing
        if LPF_ALPHA and LPF_ALPHA > 0:
            if prev_action_exec is None or prev_action_exec.shape != action_to_execute.shape:
                prev_action_exec = np.zeros_like(action_to_execute)
            action_to_execute = LPF_ALPHA * action_to_execute + (1.0 - LPF_ALPHA) * prev_action_exec
            prev_action_exec = action_to_execute.copy()

        # Clamp to environment bounds for safety and log clipping
        try:
            aspace = getattr(env, "action_space", None)
            if aspace is None and hasattr(env, "env"):
                aspace = getattr(env.env, "action_space", None)
            if aspace is not None:
                low, high = aspace.low, aspace.high
                pre_clip = action_to_execute.copy()
                action_to_execute = np.clip(action_to_execute, low, high)
                clipped_frac = float(np.mean(np.abs(pre_clip - action_to_execute) > 1e-6))
                if (step % 25) == 0:
                    mu, sd = float(np.mean(action_to_execute)), float(np.std(action_to_execute))
                    print(f"[STEP {step}] clipped_frac={clipped_frac:.2f} action(mean,std)={mu:.3f},{sd:.3f}")
            elif MANUAL_CLAMP_IF_NO_SPACE:
                # Apply manual caps if env doesn't expose action_space
                td = get_target_dim()
                pre_clip = action_to_execute.copy()
                if td >= 3:
                    action_to_execute[0:3] = np.clip(action_to_execute[0:3], -TRANS_MAX, TRANS_MAX)
                if td >= 6:
                    action_to_execute[3:6] = np.clip(action_to_execute[3:6], -ROT_MAX, ROT_MAX)
                action_to_execute[-1] = np.clip(action_to_execute[-1], -GRIP_MAX, GRIP_MAX)
                clipped_frac = float(np.mean(np.abs(pre_clip - action_to_execute) > 1e-6))
                if (step % 25) == 0:
                    mu, sd = float(np.mean(action_to_execute)), float(np.std(action_to_execute))
                    print(f"[STEP {step}] manual_clamp clipped_frac={clipped_frac:.2f} action(mean,std)={mu:.3f},{sd:.3f}")
        except Exception:
            pass

        obs, reward, done, info = env.step(action_to_execute)
        
        # flip the video because it is upside-down for some reason
        flipped_image = cv2.flip(current_image, 0)
        frames.append(cv2.cvtColor(flipped_image, cv2.COLOR_RGB2BGR))
        
        if (step + 1) % 50 == 0:
            print(f"    - Step {step+1}/{NUM_TIMESTEPS}: Reward={reward}, Done={done}")
        if done:
            print("\n[INFO] Task completed!")
            break

    print("\n[SUCCESS] Evaluation completed.")

    # 6. Save Video
    if frames:
        print("\n[INFO] Saving video...")
        video_path = os.path.join(OUTPUT_DIR, f"eval_final_{task_name}.mp4")
        height, width, layers = frames[0].shape
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"[SUCCESS] Video saved to: {video_path}")

except Exception as e:
    print(f"\n[ERROR] An unhandled error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'env' in locals():
        env.close()
        print("[INFO] Environment closed.")
    print("\n" + "="*50)
    print("SIMPLE EVALUATION FINISHED")
    print("="*50)