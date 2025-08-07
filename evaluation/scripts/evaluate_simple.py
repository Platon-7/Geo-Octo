#!/usr/bin/env python3
"""
Simplified Octo + LIBERO evaluation following the official Octo inference pattern.
Based on the official Octo inference notebook.
"""

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
import random
from functools import partial

# Disable tokenizer parallelism to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Octo / LIBERO Imports
from octo.model.octo_model import OctoModel
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_PATH = "/home/pkarageorgis/geo_octo/octo/my_octo_vggt_model_offline/octo_vggt_finetune_staged/experiment_20250805_112710_BEST_RUN"
TASK_SUITE_NAME = "libero_10" 
EVAL_TASK_ID = None 
NUM_TIMESTEPS = 200
WINDOW_SIZE = 2  # Following the official example

OUTPUT_DIR = "evaluation/test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LIBERO_DIR = "LIBERO"

print("="*50)
print("SIMPLE OCTO + LIBERO EVALUATION")
print("="*50)
print("Following official Octo inference pattern")
print("="*50)

# ==============================================================================
# (1) Load the Fine-Tuned Octo Model
# ==============================================================================
print(f"\n[INFO] Loading Octo model from: {MODEL_PATH}")
model = OctoModel.load_pretrained(MODEL_PATH)
print("[SUCCESS] Octo model loaded.\n")

# Main try block
try:
    # ==============================================================================
    # (2) Initialize LIBERO Environment
    # ==============================================================================
    print(f"[INFO] Accessing benchmark suite: {TASK_SUITE_NAME}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME]()

    if EVAL_TASK_ID is None:
        EVAL_TASK_ID = random.randint(0, task_suite.n_tasks - 1)
        print(f"[INFO] Randomly selected task #{EVAL_TASK_ID}")

    task = task_suite.get_task(EVAL_TASK_ID)
    task_name = task.name
    language_instruction = task.language

    print(f"[SUCCESS] Retrieved task '{task_name}'")
    print(f"    - Language instruction: '{language_instruction}'\n")

    # Initialize environment with 256x256 images (like the official example)
    bddl_file_path = os.path.join(LIBERO_DIR, "libero", "libero", "bddl_files", task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": bddl_file_path, "camera_heights": 256, "camera_widths": 256}
    env = OffScreenRenderEnv(**env_args)
    
    # ==============================================================================
    # (3) Create Task - Following Official Pattern
    # ==============================================================================
    print(f"[INFO] Creating task specification...")
    
    # Create task dict exactly like the official example
    task_dict = model.create_tasks(texts=[language_instruction])
    print(f"[INFO] Created task for: '{language_instruction}'")
    
    # ==============================================================================
    # (4) Setup for Inference Loop - Following Official Pattern
    # ==============================================================================
    print("[INFO] Setting up inference...")
    
    # Reset environment
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(EVAL_TASK_ID)
    env.set_init_state(init_states[0])

    # Get initial observation
    obs, _, _, _ = env.step([0.0] * 7)
    
    # Collect images for window
    images = []
    frames = []  # For video
    
    print(f"[INFO] Starting evaluation loop with {WINDOW_SIZE}-frame window...")
    
    # ==============================================================================
    # (5) Inference Loop - Following Official Pattern
    # ==============================================================================
    for step in range(NUM_TIMESTEPS):
        # Get current image
        current_image = obs["agentview_image"]
        images.append(current_image)
        
        # Keep only the last WINDOW_SIZE images
        if len(images) > WINDOW_SIZE:
            images = images[-WINDOW_SIZE:]
        
        # Only start predicting after we have enough images for the window
        if len(images) == WINDOW_SIZE:
            # Stack images for window - exactly like official example
            input_images = np.stack(images)[None]  # Add batch dimension
            
            # Create observation dict - exactly like official example  
            observation = {
                'image_primary': input_images,
                'timestep_pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool)
            }
            
            # Sample actions - exactly like official example
            actions = model.sample_actions(
                observation, 
                task_dict, 
                unnormalization_statistics=model.dataset_statistics[list(model.dataset_statistics.keys())[0]]["action"], 
                rng=jax.random.PRNGKey(step)
            )
            
            # Extract action - remove batch dimension like official example
            predicted_action = actions[0]  # [action_horizon, action_dim]
            
            # Take first action from the sequence (action chunking)
            if predicted_action.ndim == 2:
                action_to_execute = predicted_action[0]  # First action from chunk
            else:
                action_to_execute = predicted_action
            
            # Convert to numpy if needed
            if hasattr(action_to_execute, 'numpy'):
                action_to_execute = action_to_execute.numpy()
                
        else:
            # Use dummy action until we have enough images
            action_to_execute = np.zeros(7)
        
        # Step environment
        obs, reward, done, info = env.step(action_to_execute)
        
        # Save frame for video
        frames.append(cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
        
        print(f"    - Step {step+1}/{NUM_TIMESTEPS}: Reward={reward}, Done={done}")
        
        if done:
            print("[INFO] Task completed!")
            break
            
    print("[SUCCESS] Evaluation completed.\n")

    # ==============================================================================
    # (6) Save Video
    # ==============================================================================
    if frames:
        print("[INFO] Saving video...")
        video_path = os.path.join(OUTPUT_DIR, f"simple_eval_{TASK_SUITE_NAME}_{EVAL_TASK_ID}.mp4")
        height, width, layers = frames[0].shape
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"[SUCCESS] Video saved to: {video_path}")

except Exception as e:
    print(f"\n[ERROR] An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'env' in locals():
        env.close()
        print("[INFO] Environment closed.")
    print("="*50)
    print("SIMPLE EVALUATION FINISHED")
    print("="*50)