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
import json
import difflib

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
NUM_TIMESTEPS = 800
WINDOW_SIZE = 2
OUTPUT_DIR = "evaluation/test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BASE_DATA_DIR = "/scratch-shared/tmp.cwkV8vOvfY/libero_datasets"
DATASET_STATISTICS_KEY = "libero_spatial_no_noops"
LIBERO_DIR = "LIBERO"
# Ablation: one of {"multimodal", "image_conditioned", "language_conditioned"}
EVAL_MODE = "multimodal"
MAPPING_VERSION = "1.0.0"
MAPPING_DIR = "/gpfs/home4/pkarageorgis/geo_octo/evaluation/task_to_episode_map"

# Optional control ablations / safety
ZERO_ROTATION = True         # Zero out orientation deltas to test position-only control
MAP_GRIPPER_ABS_TO_REL = True  # Map gripper from [0,1] -> [-1,1]
LPF_ALPHA = 0.0             # Exponential moving average; set 0.0 to disable smoothing for snappier motion
MANUAL_CLAMP_IF_NO_SPACE = True
TRANS_MAX = 0.03            # meters per step cap if no env.action_space (raise for faster motion)
ROT_MAX = 0.15              # rad per step cap if no env.action_space
GRIP_MAX = 1.0

# Axis/sign remap toggles
FLIP_X = False   # start neutral; calibration will correct mapping
FLIP_Y = False
FLIP_Z = False

# Gripper handling
# If your model outputs relative [-1,1], set GRIPPER_MODE='rel'; if absolute [0,1], set 'abs'.
GRIPPER_MODE = 'rel'
GRIPPER_SIGN = -1.0   # invert if open/close seems reversed

# Optional translation calibration
CALIBRATE_TRANSLATION = True
CALIB_DELTA = 0.02
CALIB_STEPS = 2

# Gains to amplify model outputs (apply before clamping)
TRANS_GAIN = 5.0
ROT_GAIN = 1.0
GRIP_GAIN = 1.0

# Gripper gating / debounce
GRIPPER_HOLD_ENABLED = True
GRIPPER_HOLD_PROPORTION = 0.6    # hold for first 60% of steps
GRIPPER_HOLD_VALUE = +1.0        # relative: +1 open, -1 close (depending on controller)
GRIPPER_DEBOUNCE_STEPS = 8

# Image orientation correction (apply to both goal and observation)
IMG_VFLIP = True    # vertical flip for model inputs
IMG_HFLIP = False   # horizontal flip off to preserve left-right

def apply_image_orientation(img: np.ndarray) -> np.ndarray:
    out = img
    if IMG_VFLIP and IMG_HFLIP:
        out = cv2.flip(out, -1)
    elif IMG_VFLIP:
        out = cv2.flip(out, 0)
    elif IMG_HFLIP:
        out = cv2.flip(out, 1)
    return out

# Video rendering orientation (separate from model)
VIDEO_USE_MODEL_ORIENTATION = False
VIDEO_VFLIP = True
VIDEO_HFLIP = False

def apply_video_orientation(img: np.ndarray, model_img: np.ndarray) -> np.ndarray:
    frame = model_img if VIDEO_USE_MODEL_ORIENTATION else img
    if VIDEO_USE_MODEL_ORIENTATION:
        return frame
    if VIDEO_VFLIP and VIDEO_HFLIP:
        return cv2.flip(frame, -1)
    if VIDEO_VFLIP:
        return cv2.flip(frame, 0)
    if VIDEO_HFLIP:
        return cv2.flip(frame, 1)
    return frame

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

    def calibrate_translation_mapping(env_obj, delta=0.02, steps=2):
        """Estimate 3x3 Jacobian J s.t. Δeef ≈ J * a_xyz for small commands, then return P ≈ pinv(J)."""
        print("[INFO] Calibrating translation mapping (3x3)...")
        J = np.zeros((3, 3), dtype=np.float32)
        try:
            # zero action to get baseline
            base_obs, _, _, _ = env_obj.step(np.zeros(get_target_dim(), dtype=np.float32))
            base_eef = get_eef_pos(base_obs)
            for k in range(3):
                cmd = np.zeros(7, dtype=np.float32)
                cmd[k] = delta
                mapped = prepare_action(cmd)
                last_obs = base_obs
                for _ in range(steps):
                    last_obs, _, _, _ = env_obj.step(mapped)
                eef_after = get_eef_pos(last_obs)
                d = (eef_after - base_eef) / max(delta, 1e-6)
                J[:, k] = d
                # small settle back
                settle = prepare_action(-cmd)
                for _ in range(steps):
                    env_obj.step(settle)
            # Pseudoinverse for stability
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            S_inv = np.diag([1/s if s > 1e-6 else 0.0 for s in S])
            P = Vt.T @ S_inv @ U.T
            print(f"[CALIB] J=\n{J}\n[CALIB] P=\n{P}")
            return P
        except Exception as e:
            print(f"[WARN] Calibration failed: {e}")
            return None

    # 3. Create Task - Using YOUR structure with the CORRECT goal image
    print(f"\n[INFO] Creating task specification...")
    try:
        dataset_dir_for_eval = os.path.join(BASE_DATA_DIR, DATASET_STATISTICS_KEY, MAPPING_VERSION)
        map_path = os.path.join(MAPPING_DIR, f"{DATASET_STATISTICS_KEY}_{MAPPING_VERSION}.json")
        print(f"[INFO] Loading mapping: {map_path}")
        with open(map_path, "r") as f:
            mapping = json.load(f)
        lang_key = language_instruction.strip()
        # Build candidate episodes for this language
        by_lang = mapping.get("by_language", {})
        candidates = list(by_lang.get(lang_key, []))
        # Soft filter: prefer non-drawer/non-stove variants
        if mapping.get("episodes"):
            ep_lang_map = {int(ep["index"]): (ep.get("language") or "") for ep in mapping["episodes"]}
            filtered = [idx for idx in candidates if not any(k in ep_lang_map.get(int(idx), "").lower() for k in ["drawer", "stove"])]
            if filtered:
                candidates = filtered
        if not candidates:
            # fallback: take all with fuzzy contain
            print(f"[WARN] No exact candidates for language; using all episodes containing keywords")
            episodes_list = mapping.get("episodes", [])
            for ep in episodes_list:
                text = (ep.get("language") or "").lower()
                if "bowl" in text and "plate" in text and ("drawer" not in text and "stove" not in text):
                    candidates.append(int(ep["index"]))
        if not candidates:
            raise RuntimeError(f"No episode indices found for language: {lang_key!r} in mapping {map_path}")
        print(f"[INFO] Candidate episodes: {len(candidates)} (showing first 10): {candidates[:10]}")
        # Precompute init images
        env.seed(0)
        env.reset()
        init_states = task_suite.get_task_init_states(EVAL_TASK_ID)
        init_images = []
        for i, st in enumerate(init_states):
            try:
                env.set_init_state(st)
                tmp_obs, _, _, _ = env.step(np.zeros(get_target_dim(), dtype=np.float32))
                tmp_img = apply_image_orientation(tmp_obs.get("agentview_image", np.zeros((224, 224, 3), dtype=np.uint8)))
                if tmp_img.shape[:2] != (224, 224):
                    tmp_img = cv2.resize(tmp_img, (224, 224))
                init_images.append((i, tmp_img))
            except Exception:
                continue
        if not init_images:
            raise RuntimeError("Could not capture any init_state images")
        # Search for best (episode, init_state) by MSE
        best_pair = (None, None)
        best_mse = float("inf")
        search_cap = min(len(candidates), 50)
        for idx in candidates[:search_cap]:
            try:
                gimg = get_goal_image_from_tfrecord(dataset_dir_for_eval, int(idx))
                gimg = cv2.resize(gimg, (224, 224))
                gimg = apply_image_orientation(gimg)
                gimg_f = gimg.astype(np.float32)
                for init_idx, im in init_images:
                    mse = float(np.mean((im.astype(np.float32) - gimg_f) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best_pair = (int(idx), int(init_idx))
            except Exception:
                continue
        if best_pair[0] is None:
            raise RuntimeError("Failed to find a best (episode, init_state) pair")
        episode_index, chosen_init_idx = best_pair
        print(f"[INFO] Chosen episode {episode_index} and init_state {chosen_init_idx} with MSE={best_mse:.2f}")
        # Load the chosen goal image
        goal_image = get_goal_image_from_tfrecord(dataset_dir_for_eval, episode_index)
        goal_image_resized = cv2.resize(goal_image, (224, 224))
        goal_image_resized = apply_image_orientation(goal_image_resized)
        # Set the chosen init state
        env.set_init_state(init_states[chosen_init_idx])
        # Save for confirmation
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"chosen_goal_ep{episode_index}.png"), cv2.cvtColor(goal_image_resized, cv2.COLOR_RGB2BGR))
            # Capture the oriented init image
            tmp_obs, _, _, _ = env.step(np.zeros(get_target_dim(), dtype=np.float32))
            init_img = apply_image_orientation(tmp_obs.get("agentview_image", np.zeros((224, 224, 3), dtype=np.uint8)))
            init_img = cv2.resize(init_img, (224, 224))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"chosen_init_{chosen_init_idx}.png"), cv2.cvtColor(init_img, cv2.COLOR_RGB2BGR))
        except Exception:
            pass
        print("[SUCCESS] Correct goal image loaded from demonstration (via mapping+MSE search).")
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
    # After setting chosen init above, take a fresh obs
    obs, _, _, _ = env.step(np.zeros(get_target_dim(), dtype=np.float32))

    # Probe once before the loop
    probe_env_response(env, steps=3, delta=0.03)

    calib_P = None
    if CALIBRATE_TRANSLATION:
        calib_P = calibrate_translation_mapping(env, delta=CALIB_DELTA, steps=CALIB_STEPS)

    def extract_proprio(obs_dict):
        proprio = obs_dict.get("robot0_joint_pos", np.zeros(7))
        return proprio[:7] if len(proprio) >= 7 else np.zeros(7)

    images = []
    proprios = []
    frames = []
    prev_action_exec = None
    grip_last_value = 0.0
    grip_stable_count = 0

    print(f"[INFO] Starting evaluation loop with {WINDOW_SIZE}-frame window...")

    # 5. Inference Loop
    for step in range(NUM_TIMESTEPS):
        raw_image = obs["agentview_image"]
        current_image = apply_image_orientation(raw_image)
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
        # Apply calibrated mapping for translation
        if CALIBRATE_TRANSLATION and calib_P is not None:
            try:
                a[:3] = (calib_P @ a[:3]).astype(np.float32)
            except Exception:
                pass
        # Apply gains
        a[:3] *= TRANS_GAIN
        a[3:6] *= ROT_GAIN
        a[6] *= GRIP_GAIN
        # Gripper mapping
        if GRIPPER_MODE == 'rel':
            # Ensure in [-1,1] and apply sign
            a[6] = np.clip(a[6], -1.0, 1.0) * GRIPPER_SIGN
        else:
            # Treat as absolute [0,1], with possible inversion
            g = np.clip(a[6], 0.0, 1.0)
            a[6] = (GRIPPER_SIGN * (g * 2.0 - 1.0))  # convert to rel for controller
        # Gripper gating
        if GRIPPER_HOLD_ENABLED:
            if step < int(NUM_TIMESTEPS * GRIPPER_HOLD_PROPORTION):
                a[6] = GRIPPER_HOLD_VALUE
            else:
                # debounce small toggles
                if np.sign(a[6]) == np.sign(grip_last_value) or abs(a[6]) < 0.3:
                    grip_stable_count += 1
                else:
                    grip_stable_count = 0
                if grip_stable_count < GRIPPER_DEBOUNCE_STEPS:
                    a[6] = grip_last_value
                else:
                    grip_last_value = float(np.clip(a[6], -1.0, 1.0))
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
        
        # Use separate orientation settings for video
        video_frame = apply_video_orientation(raw_image, current_image)
        frames.append(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
        
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