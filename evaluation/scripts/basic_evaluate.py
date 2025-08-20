import warnings

# Minimal JAX compatibility shim
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

# Disable tokenizer parallelism to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Octo / LIBERO Imports
from octo.model.octo_model import OctoModel
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

"""
A simplified evaluation script:
- loads a finetuned Octo model
- creates a LIBERO environment for a single task
- optionally loads a goal image using a prebuilt language→episode mapping
- runs a short inference loop with minimal post-processing

Removed for simplicity:
- action-space probes and calibration
- rotation gating and gripper debouncing
- MSE search over init states / episodes
- separate video vs model image orientation
"""

# =============================
# Configuration (keep minimal)
# =============================
MODEL_PATH = "/home/pkarageorgis/geo_octo/octo/my_octo_vggt_model_offline/octo_vggt_finetune_staged/experiment_20250817_162546_SPATIAL_NO_VGGT"
TASK_SUITE_NAME = "libero_spatial"
EVAL_TASK_ID = 9
NUM_TIMESTEPS = 600
WINDOW_SIZE = 2
OUTPUT_DIR = "evaluation/test_outputs"
BASE_DATA_DIR = "/scratch-shared/tmp.cwkV8vOvfY/libero_datasets"
DATASET_STATISTICS_KEY = "libero_spatial_no_noops"
LIBERO_DIR = "LIBERO"
EVAL_MODE = "multimodal"  # {"multimodal","image_conditioned","language_conditioned"}
MAPPING_VERSION = "1.0.0"
MAPPING_DIR = "/gpfs/home4/pkarageorgis/geo_octo/evaluation/task_to_episode_map"

# Minimal control knobs
IMG_VFLIP = True
ZERO_ROTATION = True
TRANS_GAIN = 5.0
GRIPPER_MODE = 'rel'  # {'rel','abs'}
GRIPPER_SIGN = -1.0
MANUAL_CLAMP_IF_NO_SPACE = True
TRANS_MAX = 0.05
ROT_MAX = 0.15
GRIP_MAX = 1.0


def maybe_flip(img: np.ndarray) -> np.ndarray:
	if IMG_VFLIP:
		return cv2.flip(img, 0)
	return img


def get_goal_image_from_tfrecord(dataset_dir: str, episode_index: int) -> np.ndarray:
	"""Load the last (goal) image of an episode from TFRecords."""
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


def get_target_dim(env) -> int:
	aspace = getattr(env, "action_space", None)
	if aspace is None and hasattr(env, "env"):
		aspace = getattr(env.env, "action_space", None)
	if aspace is not None and getattr(aspace, 'shape', None) is not None:
		return int(aspace.shape[0])
	return 7


def prepare_action_for_env(env, action7: np.ndarray) -> np.ndarray:
	"""Map 7D action to env's expected dim (4D if OSC: [dx,dy,dz,gripper])."""
	a = np.asarray(action7, dtype=np.float32).reshape(-1)
	td = get_target_dim(env)
	if td == 4:
		return np.array([a[0], a[1], a[2], a[6]], dtype=np.float32)
	return a[:td] if a.shape[0] >= td else np.pad(a, (0, td - a.shape[0]))


print("=" * 50)
print("Simple OCTO + LIBERO Evaluation")
print("=" * 50)

# 1) Load model
print(f"[INFO] Loading Octo model from: {MODEL_PATH}")
model = OctoModel.load_pretrained(MODEL_PATH)
if DATASET_STATISTICS_KEY not in model.dataset_statistics:
	print(f"[ERROR] Statistics key '{DATASET_STATISTICS_KEY}' not found. Available: {list(model.dataset_statistics.keys())}")
	raise SystemExit(1)
print("[OK] Model loaded.")

# 2) Create environment and task
bench = benchmark.get_benchmark_dict()[TASK_SUITE_NAME]()
task = bench.get_task(EVAL_TASK_ID)
lang = task.language
print(f"[INFO] Task: {task.name} | Language: {lang}")

bddl_path = os.path.join(LIBERO_DIR, "libero", "libero", "bddl_files", task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=224, camera_widths=224)
obs, _, _, _ = env.step(np.zeros(get_target_dim(env), dtype=np.float32))

# 3) Optionally load goal image via mapping (simple: first candidate)
goal_image_resized = None
if EVAL_MODE in ("multimodal", "image_conditioned"):
	try:
		map_path = os.path.join(MAPPING_DIR, f"{DATASET_STATISTICS_KEY}_{MAPPING_VERSION}.json")
		with open(map_path, "r") as f:
			mapping = json.load(f)
		candidates = list(mapping.get("by_language", {}).get(lang.strip(), []))
		if not candidates:
			raise RuntimeError("No candidates for language in mapping")
		dataset_dir = os.path.join(BASE_DATA_DIR, DATASET_STATISTICS_KEY, MAPPING_VERSION)
		gimg = get_goal_image_from_tfrecord(dataset_dir, int(candidates[0]))
		goal_image_resized = maybe_flip(cv2.resize(gimg, (224, 224)))
		print(f"[OK] Loaded goal image from episode {candidates[0]}")
	except Exception as e:
		print(f"[WARN] Goal image not loaded ({e}); falling back to language-only")
		goal_image_resized = None
		
# 4) Build tasks for conditioning
if EVAL_MODE == "multimodal" and goal_image_resized is not None:
	tasks = model.create_tasks(goals={"image_primary": goal_image_resized[None]}, texts=[lang])
elif EVAL_MODE == "image_conditioned" and goal_image_resized is not None:
	tasks = model.create_tasks(goals={"image_primary": goal_image_resized[None]})
else:
	# language_conditioned or fallback
	tasks = model.create_tasks(texts=[lang])

# ensure language ids type matches finetuning
if isinstance(tasks.get("language_instruction"), dict) and "input_ids" in tasks["language_instruction"]:
	tasks["language_instruction"] = np.asarray(tasks["language_instruction"]["input_ids"], dtype=np.int32)

print("[OK] Task prompt constructed.")

# 5) Simple inference loop
images, proprios, frames = [], [], []

def extract_proprio(o):
	p = o.get("robot0_joint_pos", np.zeros(7))
	return p[:7] if len(p) >= 7 else np.zeros(7)

for step in range(NUM_TIMESTEPS):
	raw = obs["agentview_image"]
	img = maybe_flip(raw)
	prop = extract_proprio(obs)
	images.append(img)
	proprios.append(prop)
	if len(images) > WINDOW_SIZE:
		images.pop(0)
		proprios.pop(0)

	if len(images) == WINDOW_SIZE:
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
			},
		}
		acts = model.sample_actions(
			observation,
			tasks,
			unnormalization_statistics=model.dataset_statistics[DATASET_STATISTICS_KEY]["action"],
			rng=jax.random.PRNGKey(step),
		)
		a = acts[0]
		a = a[0] if a.ndim == 2 else a
	else:
		a = np.zeros(7, dtype=np.float32)

	# Minimal post-process; make writable copy
	a = np.array(a, dtype=np.float32, copy=True)
	if a.shape[0] < 7:
		a = np.pad(a, (0, 7 - a.shape[0]))
	if ZERO_ROTATION:
		a[3:6] = 0.0
	# gripper mapping
	if GRIPPER_MODE == 'rel':
		a[6] = np.clip(a[6], -1.0, 1.0) * GRIPPER_SIGN
	else:
		g = np.clip(a[6], 0.0, 1.0)
		a[6] = GRIPPER_SIGN * (g * 2.0 - 1.0)
	# translate gain
	a[:3] *= TRANS_GAIN

	# map to env
	exec_action = prepare_action_for_env(env, a)

	# manual clamp if needed
	aspace = getattr(env, "action_space", None)
	if aspace is None and MANUAL_CLAMP_IF_NO_SPACE:
		if exec_action.shape[0] >= 3:
			exec_action[0:3] = np.clip(exec_action[0:3], -TRANS_MAX, TRANS_MAX)
		if exec_action.shape[0] >= 6:
			exec_action[3:6] = np.clip(exec_action[3:6], -ROT_MAX, ROT_MAX)
		exec_action[-1] = np.clip(exec_action[-1], -GRIP_MAX, GRIP_MAX)

	# step env
	obs, reward, done, info = env.step(exec_action)

	# store video frame (use same orientation as model)
	frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
	if done:
		print(f"[DONE] Step {step}")
		break

# 6) Save video
if frames:
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	video_path = os.path.join(OUTPUT_DIR, f"eval_simple_{task.name}.mp4")
	h, w = frames[0].shape[:2]
	vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
	for fr in frames:
		vw.write(fr)
	vw.release()
	print(f"[OK] Video saved to: {video_path}")

# Cleanup
env.close()
print("[INFO] Environment closed. Simple evaluation finished.")