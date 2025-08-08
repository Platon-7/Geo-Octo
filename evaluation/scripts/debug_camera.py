#!/usr/bin/env python3
"""
Debug script to check camera orientation in LIBERO environment.
"""

import os
import cv2
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# Configuration
LIBERO_DIR = "/gpfs/home4/pkarageorgis/geo_octo/LIBERO"
TASK_SUITE_NAME = "libero_10"
OUTPUT_DIR = "evaluation/test_outputs"

print("🎥 CAMERA ORIENTATION DEBUG")
print("=" * 50)

try:
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize environment
    print("[INFO] Setting up LIBERO environment...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME]()
    
    # Get task
    task_id = 0
    task = task_suite.get_task(task_id)
    print(f"[INFO] Using task: {task.name}")
    
    # Setup environment
    bddl_file_path = os.path.join(LIBERO_DIR, "libero", "libero", "bddl_files", task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": bddl_file_path, "camera_heights": 224, "camera_widths": 224}
    env = OffScreenRenderEnv(**env_args)
    
    # Reset and get initial observation
    init_states = task_suite.get_task_init_states(task_id)
    env.seed(0)
    env.reset()
    env.set_init_state(init_states[0])
    
    # Take one step to get observation
    obs, _, _, _ = env.step([0.0] * 7)
    image = obs["agentview_image"]
    
    print(f"[INFO] Image shape: {image.shape}")
    print(f"[INFO] Image dtype: {image.dtype}")
    print(f"[INFO] Image range: [{image.min()}, {image.max()}]")
    
    # Save raw image
    raw_path = os.path.join(OUTPUT_DIR, "debug_camera_raw.png")
    cv2.imwrite(raw_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[SUCCESS] Raw image saved: {raw_path}")
    
    # Save flipped versions for comparison
    flipped_ud = np.flipud(image)
    flipped_lr = np.fliplr(image)
    rotated_180 = np.rot90(image, 2)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_camera_flipped_ud.png"), cv2.cvtColor(flipped_ud, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_camera_flipped_lr.png"), cv2.cvtColor(flipped_lr, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_camera_rotated_180.png"), cv2.cvtColor(rotated_180, cv2.COLOR_RGB2BGR))
    
    print(f"[SUCCESS] Comparison images saved in {OUTPUT_DIR}")
    print("[INFO] Check which orientation looks correct!")
    
    # Clean up
    env.close()
    
except Exception as e:
    print(f"[ERROR] Debug failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 50)
print("🎥 CAMERA DEBUG FINISHED")