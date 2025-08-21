import warnings

# Minimal JAX compatibility shim (avoid import errors in some envs)
try:
    import jax.numpy as jnp
    if not hasattr(jnp, 'DeviceArray'):
        jnp.DeviceArray = jnp.ndarray
except Exception:
    pass

warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")

import os
import cv2
import json
import time
import argparse
import numpy as np
import jax
from dataclasses import dataclass
from typing import Optional, Tuple

# Disable tokenizer parallelism to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Octo / LIBERO Imports
from octo.model.octo_model import OctoModel
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


# ------------------------------
# Constants and helpers
# ------------------------------

class TaskSuite:
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Max steps per task suite (mirrors OpenVLA evaluator defaults)
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


def get_target_dim(env) -> int:
    aspace = getattr(env, "action_space", None)
    if aspace is None and hasattr(env, "env"):
        aspace = getattr(env.env, "action_space", None)
    if aspace is not None and getattr(aspace, 'shape', None) is not None:
        return int(aspace.shape[0])
    return 7


def prepare_action_for_env(env, action7: np.ndarray) -> np.ndarray:
    """Map 7D action [dx,dy,dz,dRx,dRy,dRz,grip] to env's expected dim.
    If env uses 4D OSC, map to [dx,dy,dz,gripper]. Otherwise pad/trim to match.
    """
    a = np.asarray(action7, dtype=np.float32).reshape(-1)
    td = get_target_dim(env)
    if td == 4:
        return np.array([a[0], a[1], a[2], a[6]], dtype=np.float32)
    if a.shape[0] < td:
        a = np.pad(a, (0, td - a.shape[0]))
    else:
        a = a[:td]
    return a


def save_rollout_video(frames, video_path: str, fps: int = 20) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for fr in frames:
        vw.write(fr)
    vw.release()


def make_observation(images_window, proprios_window, step: int, window_size: int):
    return {
        'image_primary': np.stack(images_window)[None],
        'proprio': np.stack(proprios_window)[None],
        'timestep_pad_mask': np.full((1, window_size), True, dtype=bool),
        'timestep': np.array([[step - (window_size - 1), step]], dtype=np.int32),
        'task_completed': np.zeros((1, window_size, 4), dtype=bool),
        'pad_mask_dict': {
            'image_primary': np.full((1, window_size), True, dtype=bool),
            'proprio': np.full((1, window_size), True, dtype=bool),
            'timestep': np.full((1, window_size), True, dtype=bool),
        },
    }


def extract_proprio(obs_dict) -> np.ndarray:
    p = obs_dict.get("robot0_joint_pos", np.zeros(7, dtype=np.float32))
    return p[:7] if len(p) >= 7 else np.zeros(7, dtype=np.float32)


def apply_image_orientation(img: np.ndarray, vflip: bool, hflip: bool) -> np.ndarray:
    out = img
    if vflip and hflip:
        out = cv2.flip(out, -1)
    elif vflip:
        out = cv2.flip(out, 0)
    elif hflip:
        out = cv2.flip(out, 1)
    return out


# ------------------------------
# Config
# ------------------------------

@dataclass
class EvalConfig:
    # Model / dataset
    model_path: str
    dataset_statistics_key: str

    # LIBERO
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_trials_per_task: int = 50
    num_steps_wait: int = 10
    img_res: int = 224
    libero_dir: str = "LIBERO"

    # Inference window
    window_size: int = 2

    # Limits
    trans_gain: float = 5.0
    zero_rotation: bool = True
    gripper_mode: str = 'rel'   # {'rel','abs'}; 'rel' expects [-1,1]
    gripper_sign: float = -1.0  # invert if open/close reversed

    # Image orientation for model inputs
    img_vflip: bool = True
    img_hflip: bool = True

    # Output
    output_dir: str = "evaluation/test_outputs_alt"


def load_model_and_validate(cfg: EvalConfig) -> OctoModel:
    print(f"[INFO] Loading Octo model from: {cfg.model_path}")
    model = OctoModel.load_pretrained(cfg.model_path)
    if cfg.dataset_statistics_key not in model.dataset_statistics:
        keys = list(model.dataset_statistics.keys())
        raise KeyError(
            f"Statistics key '{cfg.dataset_statistics_key}' not found in model. Available: {keys}"
        )
    print("[OK] Model loaded and statistics key validated.")
    return model


def build_env_for_task(cfg: EvalConfig, task) -> OffScreenRenderEnv:
    bddl_path = os.path.join(
        cfg.libero_dir, "libero", "libero", "bddl_files", task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=cfg.img_res, camera_widths=cfg.img_res)
    return env


def build_tasks_prompt(model: OctoModel, task_language: str):
    tasks = model.create_tasks(texts=[task_language])
    # Ensure language is passed as token ids like during finetuning
    if isinstance(tasks.get("language_instruction"), dict) and "input_ids" in tasks["language_instruction"]:
        tasks["language_instruction"] = np.asarray(tasks["language_instruction"]["input_ids"], dtype=np.int32)
    return tasks


def episode_loop(cfg: EvalConfig, model: OctoModel, env: OffScreenRenderEnv, tasks, max_steps: int,
                 unnorm_stats) -> Tuple[bool, list]:
    images, proprios, frames = [], [], []
    obs, _, _, _ = env.step(np.zeros(get_target_dim(env), dtype=np.float32))

    # Let the scene settle
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(np.zeros(get_target_dim(env), dtype=np.float32))

    success = False
    for step in range(max_steps):
        raw_img = obs.get("agentview_image")
        if raw_img is None:
            raw_img = np.zeros((cfg.img_res, cfg.img_res, 3), dtype=np.uint8)
        model_img = apply_image_orientation(raw_img, cfg.img_vflip, cfg.img_hflip)
        proprio = extract_proprio(obs)

        images.append(model_img)
        proprios.append(proprio)
        if len(images) > cfg.window_size:
            images.pop(0)
            proprios.pop(0)

        if len(images) == cfg.window_size:
            observation = make_observation(images, proprios, step, cfg.window_size)
            acts = model.sample_actions(
                observation,
                tasks,
                unnormalization_statistics=unnorm_stats,
                rng=jax.random.PRNGKey(step),
            )
            a = acts[0]
            a = a[0] if a.ndim == 2 else a
        else:
            a = np.zeros(7, dtype=np.float32)

        a = np.array(a, dtype=np.float32, copy=True)
        if a.shape[0] < 7:
            a = np.pad(a, (0, 7 - a.shape[0]))

        # Simple post-processing
        if cfg.zero_rotation:
            a[3:6] = 0.0

        # Translation gain
        a[:3] *= cfg.trans_gain

        # Gripper mapping
        if cfg.gripper_mode == 'rel':
            a[6] = float(np.clip(a[6], -1.0, 1.0) * cfg.gripper_sign)
        else:
            g = float(np.clip(a[6], 0.0, 1.0))
            a[6] = cfg.gripper_sign * (g * 2.0 - 1.0)

        exec_action = prepare_action_for_env(env, a)
        obs, reward, done, info = env.step(exec_action)

        # Store frame for video (use same orientation as model input)
        frames.append(cv2.cvtColor(model_img, cv2.COLOR_RGB2BGR))
        if done:
            success = True
            break

    return success, frames


def evaluate(cfg: EvalConfig) -> float:
    os.makedirs(cfg.output_dir, exist_ok=True)
    model = load_model_and_validate(cfg)

    bench = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    num_tasks = bench.n_tasks
    print(f"[INFO] Task suite: {cfg.task_suite_name} | #tasks: {num_tasks}")

    total_episodes = 0
    total_successes = 0
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 300)
    unnorm_stats = model.dataset_statistics[cfg.dataset_statistics_key]["action"]

    for task_id in range(num_tasks):
        task = bench.get_task(task_id)
        task_lang = task.language
        print(f"\n[TASK {task_id}] {task.name} :: '{task_lang}'")

        env = build_env_for_task(cfg, task)
        init_states = bench.get_task_init_states(task_id)
        tasks_prompt = build_tasks_prompt(model, task_lang)

        task_episodes = 0
        task_successes = 0

        for episode_idx in range(cfg.num_trials_per_task):
            if episode_idx >= len(init_states):
                print(f"[WARN] No more initial states for task {task_id}; stopping early.")
                break

            print(f"[INFO] Episode {episode_idx + 1}/{cfg.num_trials_per_task}")
            env.reset()
            try:
                env.set_init_state(init_states[episode_idx])
            except Exception:
                pass

            success, frames = episode_loop(cfg, model, env, tasks_prompt, max_steps, unnorm_stats)

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            # Save episode video
            video_name = f"task{task_id:02d}_ep{episode_idx:03d}_{'success' if success else 'fail'}.mp4"
            save_rollout_video(frames, os.path.join(cfg.output_dir, video_name))

            # Log running stats
            print(f"[RESULT] success={success} | task_sr={task_successes}/{task_episodes}"
                  f" ({(task_successes/max(1,task_episodes))*100:.1f}%) | total_sr={total_successes}/{total_episodes}"
                  f" ({(total_successes/max(1,total_episodes))*100:.1f}%)")

        env.close()

    final_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print("\n====================")
    print(f"Final episodes: {total_episodes}")
    print(f"Final successes: {total_successes}")
    print(f"Overall success rate: {final_sr:.4f} ({final_sr*100:.1f}%)")
    print("====================")
    return final_sr


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser("Alternate LIBERO evaluator (Octo-adapted)")
    # Model / stats
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--dataset_statistics_key', type=str, required=True,
                   help="Key in model.dataset_statistics for action unnormalization (e.g., libero_spatial_no_noops)")

    # LIBERO
    p.add_argument('--task_suite_name', type=str, default=TaskSuite.LIBERO_SPATIAL,
                   choices=list(TASK_MAX_STEPS.keys()))
    p.add_argument('--num_trials_per_task', type=int, default=50)
    p.add_argument('--num_steps_wait', type=int, default=10)
    p.add_argument('--img_res', type=int, default=224)
    p.add_argument('--libero_dir', type=str, default='LIBERO')

    # Inference window
    p.add_argument('--window_size', type=int, default=2)

    # Control
    p.add_argument('--trans_gain', type=float, default=5.0)
    p.add_argument('--zero_rotation', type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument('--gripper_mode', type=str, default='rel', choices=['rel', 'abs'])
    p.add_argument('--gripper_sign', type=float, default=-1.0)

    # Orientation
    p.add_argument('--img_vflip', type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument('--img_hflip', type=lambda x: str(x).lower() == 'true', default=True)

    # Output
    p.add_argument('--output_dir', type=str, default='evaluation/test_outputs_alt')

    args = p.parse_args()
    return EvalConfig(
        model_path=args.model_path,
        dataset_statistics_key=args.dataset_statistics_key,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        num_steps_wait=args.num_steps_wait,
        img_res=args.img_res,
        libero_dir=args.libero_dir,
        window_size=args.window_size,
        trans_gain=args.trans_gain,
        zero_rotation=args.zero_rotation,
        gripper_mode=args.gripper_mode,
        gripper_sign=args.gripper_sign,
        img_vflip=args.img_vflip,
        img_hflip=args.img_hflip,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    try:
        evaluate(cfg)
    except KeyboardInterrupt:
        print("[INTERRUPT] Stopped by user.")

