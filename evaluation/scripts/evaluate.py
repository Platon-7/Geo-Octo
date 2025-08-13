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

# ==== Optional image crop to approximate training aug average ====
USE_CENTER_CROP = False
CROP_SCALE = 0.9
CROP_RATIO = 1.0

def average_crop_resize(img, out=(224, 224), scale=0.9, ratio=1.0):
    h, w = img.shape[:2]
    new_h = int(h * np.sqrt(scale / max(ratio, 1e-6)))
    new_w = int(w * np.sqrt(scale * ratio))
    new_h = max(1, min(new_h, h))
    new_w = max(1, min(new_w, w))
    y0 = max(0, (h - new_h) // 2)
    x0 = max(0, (w - new_w) // 2)
    crop = img[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(crop, out, interpolation=cv2.INTER_LINEAR)

# ==== Action remap to align dataset semantics with env controller ====
USE_ACTION_REMAP = True
# Choose between modes: "ee_delta" (xyz+rpy+gripper) or "joint_delta" (7 joint deltas)
ACTION_MODE = "ee_delta"  # alternatives: "joint_delta", "none"

# Scales for ee_delta mode
TRANS_SCALE = 0.15   # try 0.05–0.25
ROT_SCALE = 0.7      # try 0.3–1.0
# Per-axis permutation/signs (applied to first 3 for translation, next 3 for rotation)
TRANS_PERM = (0, 1, 2)
ROT_PERM = (0, 1, 2)
TRANS_SIGN = ( -1.0, 1.0, 1.0 )  # flip X by default since motion drifts right
ROT_SIGN_VEC = ( 1.0, 1.0, 1.0 )

# Scale for joint_delta mode
JOINT_SCALE = 0.2    # try 0.1–0.5

# Sampling controls
USE_ARGMAX = False    # deterministic often collapses to tiny actions
SAMPLE_TEMPERATURE = 0.7

# Task modality toggles
USE_LANGUAGE = True
USE_GOAL_IMAGE = True


def remap_action(a: np.ndarray) -> np.ndarray:
    if not USE_ACTION_REMAP or ACTION_MODE == "none":
        return a
    arr = np.array(a, dtype=np.float32, copy=True)
    if ACTION_MODE == "ee_delta":
        # apply permutation + sign on translation and rotation separately, then scale
        def map_ee(vec):
            t = vec[:3]
            r = vec[3:6]
            t = t[list(TRANS_PERM)] * np.array(TRANS_SIGN, dtype=np.float32)
            r = r[list(ROT_PERM)] * np.array(ROT_SIGN_VEC, dtype=np.float32)
            t = t * TRANS_SCALE
            r = r * ROT_SCALE
            out = np.concatenate([t, r, vec[6:7]], axis=0)
            return out
        if arr.ndim == 2:
            return np.stack([map_ee(v) for v in arr], axis=0)
        else:
            return map_ee(arr)
    elif ACTION_MODE == "joint_delta":
        if arr.ndim == 2:
            arr[:, :6] *= JOINT_SCALE
        else:
            arr[:6] *= JOINT_SCALE
        return arr
    else:
        return arr

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

# Print model spec
try:
    print("[DEBUG] Model spec:")
    print(model.get_pretty_spec())
except Exception as _:
    pass

# Print dataset action stats
if DATASET_STATISTICS_KEY not in model.dataset_statistics:
    print(f"[ERROR] Statistics key '{DATASET_STATISTICS_KEY}' not found in model.")
    print(f"Available keys: {list(model.dataset_statistics.keys())}")
    exit()
else:
    ds_stats = model.dataset_statistics[DATASET_STATISTICS_KEY]
    a_min = np.array(ds_stats["action"]["min"])[:7]
    a_max = np.array(ds_stats["action"]["max"])[:7]
    print("[DEBUG] Dataset action min/max (first 7 dims):", a_min, a_max)

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
    
    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": 224,
        "camera_widths": 224,
    }
    env = OffScreenRenderEnv(**env_args)
    # Print env action space for comparison
    try:
        print("[DEBUG] Env action space low/high (first 7 dims):",
              env.action_space.low[:7], env.action_space.high[:7])
    except Exception:
        pass

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
    
    # Restoring your original, successful method for creating the task dictionary
    # Build tasks per toggles
    goals = {"image_primary": goal_image_resized[None]} if USE_GOAL_IMAGE else None
    texts = [language_instruction] if USE_LANGUAGE else None
    tasks = model.create_tasks(goals=goals, texts=texts)
    # Ensure we pass token ids (like during finetuning), not HF model dict
    if USE_LANGUAGE and isinstance(tasks.get("language_instruction"), dict) and "input_ids" in tasks["language_instruction"]:
        tasks["language_instruction"] = np.array(tasks["language_instruction"]["input_ids"]).astype(np.int32)
        # ensure pad mask marks language present
        if "pad_mask_dict" not in tasks:
            tasks["pad_mask_dict"] = {}
        tasks["pad_mask_dict"]["language_instruction"] = np.ones(tasks["language_instruction"].shape[0], dtype=bool)
    print("[SUCCESS] Task prompt created. USE_GOAL_IMAGE=", USE_GOAL_IMAGE, " USE_LANGUAGE=", USE_LANGUAGE)
    print("[DEBUG] Action mode:", ACTION_MODE, "TRANS_SCALE=", TRANS_SCALE, "ROT_SCALE=", ROT_SCALE, "TRANS_SIGN=", TRANS_SIGN)

    # 4. Setup for Inference Loop
    print("\n[INFO] Setting up inference...")
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(EVAL_TASK_ID)
    env.set_init_state(init_states[0])
    obs, _, _, _ = env.step(np.zeros(7))

    def extract_proprio(obs_dict):
        proprio = obs_dict.get("robot0_joint_pos", np.zeros(7))
        return proprio[:7] if len(proprio) >= 7 else np.zeros(7)

    images = []
    frames = []

    print(f"[INFO] Starting evaluation loop with {WINDOW_SIZE}-frame window...")

    # 5. Inference Loop
    for step in range(NUM_TIMESTEPS):
        current_image = obs["agentview_image"]
        if USE_CENTER_CROP:
            current_image = average_crop_resize(current_image, out=(224, 224), scale=CROP_SCALE, ratio=CROP_RATIO)
        images.append(current_image)

        if len(images) > WINDOW_SIZE:
            images.pop(0)

        if len(images) == WINDOW_SIZE:
            observation = {
                'image_primary': np.stack(images)[None],
                'timestep_pad_mask': np.full((1, WINDOW_SIZE), True, dtype=bool),
                'timestep': np.array([[step-(WINDOW_SIZE-1), step]], dtype=np.int32),
                'task_completed': np.zeros((1, WINDOW_SIZE, 4), dtype=bool),
                'pad_mask_dict': {
                    'image_primary': np.full((1, WINDOW_SIZE), True, dtype=bool),
                    'timestep': np.full((1, WINDOW_SIZE), True, dtype=bool),
                }
            }

            # Print observation diagnostics
            print("[STEP", step, "] obs image shape/dtype:", observation['image_primary'].shape, observation['image_primary'].dtype)

            actions = model.sample_actions(
                observation,
                tasks,
                unnormalization_statistics=model.dataset_statistics[DATASET_STATISTICS_KEY]["action"],
                rng=jax.random.PRNGKey(step),
                argmax=USE_ARGMAX,
                temperature=SAMPLE_TEMPERATURE
            )
            predicted_action = actions[0]
            action_to_execute = predicted_action[0] if predicted_action.ndim == 2 else predicted_action
        else:
            action_to_execute = np.zeros(7)

        # Optional action remap (scale/flip) before clamping
        if USE_ACTION_REMAP:
            pre_remap = action_to_execute.copy()
            action_to_execute = remap_action(action_to_execute)
            if (step % 25) == 0:
                print("[STEP", step, "] pre-remap first6:", pre_remap[:6])
                print("[STEP", step, "] post-remap first6:", action_to_execute[:6])

        # Clamp to env bounds
        try:
            low, high = env.action_space.low, env.action_space.high
            unclipped = action_to_execute.copy()
            action_to_execute = np.clip(action_to_execute, low, high)
            if (step % 25) == 0:
                print("[STEP", step, "] unclipped first6:", unclipped[:6])
                print("[STEP", step, "] clipped first6:", action_to_execute[:6])
        except Exception:
            pass

        obs, reward, done, info = env.step(action_to_execute)

        # flip the video because it is upside-down for some reason
        flipped_image = cv2.flip(current_image, 0)
        frames.append(cv2.cvtColor(flipped_image, cv2.COLOR_RGB2BGR))

        if (step + 1) % 25 == 0:
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