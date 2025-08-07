#!/usr/bin/env python3
"""
Fix JAX compatibility issue with transformers library.
This script adds a compatibility shim for the deprecated DeviceArray.
"""

import sys
import warnings

# Add compatibility shim before importing anything else
try:
    import jax.numpy as jnp
    if not hasattr(jnp, 'DeviceArray'):
        # Add the deprecated DeviceArray as an alias to Array
        jnp.DeviceArray = jnp.ndarray
        print("[FIX] Added DeviceArray compatibility shim")
except ImportError:
    print("[WARNING] Could not import JAX")

# Suppress the specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")

print('--- Python Executable ---')
print(sys.executable)
print('\n--- Python Path ---')
print(sys.path)
print('\n--- Loading NumPy ---')
import numpy
print('NumPy Version:', numpy.__version__)
print('NumPy Path:', numpy.__file__)
print('\n--- Attempting to import cv2 ---')
import cv2
print('cv2 imported successfully!')

import os
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import random
from functools import partial

# Octo / LIBERO Imports
from octo.model.octo_model import OctoModel
from octo.utils.train_callbacks import supply_rng
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# ==============================================================================
# (0) Configuration
# ==============================================================================
# --- REQUIRED PATHS ---
# Path to your fine-tuned Octo model checkpoint directory
MODEL_PATH = "/home/pkarageorgis/geo_octo/octo/my_octo_vggt_model_offline/octo_vggt_finetune_staged/experiment_20250805_112710_BEST_RUN"

# Path to the PARENT directory where you downloaded the datasets
# This should be the path that CONTAINS the "libero_10", "libero_goal", etc. folders
DATASET_DIR = "/scratch-shared/tmp.cwkV8vOvfY/libero_evaluation"

# --- EVALUATION CONFIG ---
# Which task suite to evaluate on. Let's use libero_10 as requested.
TASK_SUITE_NAME = "libero_10" 
# Set to an integer to test a specific task, or None to pick a random one
EVAL_TASK_ID = None 
# How many steps to run the test for.
NUM_TIMESTEPS = 200 # A bit longer to see meaningful behavior

# --- OUTPUT ---
OUTPUT_DIR = "evaluation/test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# The root of the LIBERO library relative to where the script is run from (`geo_octo`)
LIBERO_DIR = "LIBERO"

print("="*50)
print("OCTO MODEL EVALUATION SCRIPT (JAX FIXED)")
print("="*50)

# ==============================================================================
# (1) Load the Fine-Tuned Octo Model
# ==============================================================================
print(f"\n[INFO] Loading Octo model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"FATAL: Model checkpoint not found at {MODEL_PATH}")

# Load the model using the correct API
model = OctoModel.load_pretrained(MODEL_PATH)
print("[SUCCESS] Octo model loaded.\n")

# Main try block to ensure the environment is closed properly
try:
    # ==============================================================================
    # (2) Initialize the LIBERO Environment
    # ==============================================================================
    print(f"[INFO] Accessing benchmark suite: {TASK_SUITE_NAME}")
    benchmark_dict = benchmark.get_benchmark_dict()
    
    # Initialize the task suite (LIBERO handles datasets differently)
    task_suite = benchmark_dict[TASK_SUITE_NAME]()

    # Pick a random task if no specific ID is given
    if EVAL_TASK_ID is None:
        EVAL_TASK_ID = random.randint(0, task_suite.n_tasks - 1)
        print(f"[INFO] No task ID specified. Randomly selected task #{EVAL_TASK_ID}")

    task = task_suite.get_task(EVAL_TASK_ID)
    task_name = task.name
    language_instruction = task.language # Get the language goal for the model

    # Construct BDDL path (same as before)
    bddl_file_path = os.path.join(LIBERO_DIR, "libero", "libero", "bddl_files", task.problem_folder, task.bddl_file)

    print(f"[SUCCESS] Retrieved task '{task_name}'")
    print(f"    - Language instruction: '{language_instruction}'\n")

    # Initialize the simulation environment with model's expected image size
    env_args = {"bddl_file_name": bddl_file_path, "camera_heights": 224, "camera_widths": 224}
    env = OffScreenRenderEnv(**env_args)
    
    # ==============================================================================
    # (3) Create Task Specification and Policy Function
    # ==============================================================================
    # Create task specification using model's utility function
    task_spec = model.create_tasks(texts=[language_instruction])
    print(f"[INFO] Created task specification for: '{language_instruction}'")
    
    # Create policy function with proper normalization
    # We'll try to find appropriate statistics or use default ones
    print("[INFO] Setting up policy function...")
    
    # Get the available dataset statistics
    available_datasets = list(model.dataset_statistics.keys())
    print(f"[INFO] Available dataset statistics: {available_datasets}")
    
    # Try to find appropriate action statistics - use the first available dataset
    if available_datasets:
        action_stats = model.dataset_statistics[available_datasets[0]]["action"]
        print(f"[INFO] Using action statistics from: {available_datasets[0]}")
    else:
        # Fallback: create basic statistics for 7-DOF actions (robot joint + gripper)
        action_stats = {
            "mean": np.zeros(7),
            "std": np.ones(7)
        }
        print("[WARNING] No dataset statistics found, using default normalization")
    
    # Create policy function
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=action_stats,
            train=False,
        ),
    )
    
    # ==============================================================================
    # (4) Run the Evaluation Loop
    # ==============================================================================
    print("[INFO] Starting evaluation loop...")
    
    # Reset env and set a deterministic initial state from the dataset
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(EVAL_TASK_ID)
    env.set_init_state(init_states[0]) # Use the first initial state

    # Get the first observation by stepping with a dummy action
    obs, _, _, _ = env.step([0.0] * 7)
    
    # List to store frames for video
    frames = []
    
    # Initialize observation history for window_size=2
    obs_history = [obs["agentview_image"], obs["agentview_image"]]  # Duplicate first frame

    for step in range(NUM_TIMESTEPS):
        # Prepare observation for the model
        current_image = obs["agentview_image"]
        
        # Update observation history (window_size=2)
        obs_history = obs_history[1:] + [current_image]
        
        # Stack images for window
        image_stack = np.stack(obs_history, axis=0)  # (window_size, H, W, C)
        
        # Create simplified observation format for dummy testing
        # We'll provide minimal required keys and dummy values for the rest
        model_observation = {
            "image_primary": image_stack[None, ...],  # Add batch dimension: (1, window_size, H, W, C)
            "timestep_pad_mask": np.array([[True, True]], dtype=bool),  # No padding for both timesteps
            # Dummy values for required keys (not extracted live for this test)
            "vggt_tokens": np.zeros((1, 2, 512), dtype=np.float32),  # Dummy - would need live extraction for real use
            "proprio": np.zeros((1, 2, 7), dtype=np.float32),  # Dummy - could be extracted from robot state if needed
            "timestep": np.array([[step, step+1]], dtype=np.int32),  # Actual timestep indices
            "task_completed": np.array([[False, False]], dtype=bool),  # Task completion status
            "pad_mask_dict": {
                "image_primary": np.array([[True, True]], dtype=bool),
                "vggt_tokens": np.array([[False, False]], dtype=bool),  # Mark as invalid/dummy
                "proprio": np.array([[False, False]], dtype=bool),     # Mark as invalid/dummy  
                "timestep": np.array([[True, True]], dtype=bool),
            }
        }

        # Get action from the model
        actions = policy_fn(model_observation, task_spec)
        
        # Remove batch dimension and take first action if action chunking is used
        if actions.ndim == 3:  # (batch, action_horizon, action_dim)
            predicted_action = actions[0, 0]  # Take first action from sequence
        else:
            predicted_action = actions[0]  # Just remove batch dim
        
        # Convert to numpy if needed
        if hasattr(predicted_action, 'numpy'):
            predicted_action = predicted_action.numpy()
        
        # Step the environment with the model's action
        obs, reward, done, info = env.step(predicted_action)
        
        # Render the frame for the video
        frames.append(cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)) # Convert to BGR for OpenCV

        print(f"    - Step {step+1}/{NUM_TIMESTEPS}: Reward={reward}, Done={done}")

        if done:
            print("[INFO] Task succeeded! Episode finished early.")
            break
            
    print("[SUCCESS] Evaluation loop completed.\n")

    # ==============================================================================
    # (5) Save Video of the Episode
    # ==============================================================================
    if frames:
        print("[INFO] Saving episode video...")
        video_path = os.path.join(OUTPUT_DIR, f"octo_eval_{TASK_SUITE_NAME}_{EVAL_TASK_ID}.mp4")
        height, width, layers = frames[0].shape
        # Use 'mp4v' codec for MP4 files
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"[SUCCESS] Video saved to: {video_path}\n")

except Exception as e:
    print(f"\n[ERROR] An error occurred during the evaluation: {e}")
    import traceback
    traceback.print_exc()

finally:
    # ==============================================================================
    # (6) Clean Up
    # ==============================================================================
    if 'env' in locals() and 'env' in vars():
        env.close()
        print("[INFO] Environment closed.")
    print("="*50)
    print("EVALUATION SCRIPT FINISHED")
    print("="*50)