import os
import cv2
import numpy as np

import sys
import pprint

print("=============================================")
print("PYTHON ENVIRONMENT DIAGNOSTIC REPORT")
print("=============================================")

print("\n--- Python Executable ---")
print("This is the Python being run:")
print(sys.executable)

print("\n--- Environment Root (sys.prefix) ---")
print("This is the root of the detected Conda environment:")
print(sys.prefix)

print("\n--- Current Working Directory ---")
print("The script is being run from this folder:")
print(os.getcwd())

print("\n--- Python Search Path (sys.path) ---")
print("Python will look for modules in these directories, in this order:")
# Using pprint for clean, readable output
pprint.pprint(sys.path)

print("\n=============================================")
print("DIAGNOSTIC REPORT END")
print("=============================================")

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# ==============================================================================
# (0) Constants & Configuration
# ==============================================================================
# Define the task suite and the specific task we want to test
TASK_SUITE_NAME = "libero_goal"
TASK_NAME = "open_the_top_drawer_and_put_the_bowl_inside"
# Directory to save test outputs (like rendered images)
OUTPUT_DIR = "evaluation/test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The root of the LIBERO library relative to where the script is run from (`geo_octo`)
LIBERO_DIR = "LIBERO"

print("="*50)
print("LIBERO ENVIRONMENT VALIDATION SCRIPT")
print("="*50)

try:
    # ==============================================================================
    # (1) Acessing the Benchmark and a Specific Task
    # ==============================================================================
    print(f"\n[INFO] Accessing benchmark suite: {TASK_SUITE_NAME}")
    
    # Get the dictionary of all available benchmarks
    benchmark_dict = benchmark.get_benchmark_dict()
    
    # Instantiate the specific task suite we want to use
    task_suite = benchmark_dict[TASK_SUITE_NAME]()
    
    # Find the specific task ID by its name
    task_id = -1
    for i in range(task_suite.n_tasks):
        if task_suite.get_task(i).name == TASK_NAME:
            task_id = i
            break
            
    if task_id == -1:
        raise ValueError(f"Task '{TASK_NAME}' not found in suite '{TASK_SUITE_NAME}'.")

    # Retrieve the task object
    task = task_suite.get_task(task_id)
    
    # Construct the path to the BDDL file MANUALLY
    bddl_files_dir = os.path.join(LIBERO_DIR, "libero", "libero", "bddl_files")
    bddl_file_path = os.path.join(bddl_files_dir, task.problem_folder, task.bddl_file)
    
    print(f"[SUCCESS] Retrieved task '{task.name}' (ID: {task_id})")
    print(f"    - Language instruction: '{task.language}'")
    print(f"    - Attempting to use BDDL file: {bddl_file_path}")

    # Add a check to ensure the constructed path is valid
    if not os.path.exists(bddl_file_path):
        raise FileNotFoundError(f"FATAL: BDDL file not found at the constructed path: {bddl_file_path}\n"
                              f"Please ensure the script is run from the 'geo_octo' directory.")
    
    print(f"[SUCCESS] BDDL file found.\n")


    # ==============================================================================
    # (2) Initializing the Simulation Environment
    # ==============================================================================
    print("[INFO] Initializing the offscreen simulation environment...")
    
    # Environment arguments, crucial for headless rendering
    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    
    # Use OffScreenRenderEnv for headless servers
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # for reproducibility
    env.reset()
    
    print("[SUCCESS] Environment initialized.\n")

    # ==============================================================================
    # (3) Setting a Deterministic Initial State
    # ==============================================================================
    print("[INFO] Setting a deterministic initial state for the task...")
    
    init_states = task_suite.get_task_init_states(task_id)
    init_state_id = 0
    
    env.set_init_state(init_states[init_state_id])
    
    print(f"[SUCCESS] Environment reset to initial state #{init_state_id}.\n")

    # ==============================================================================
    # (4) Running a Dummy Simulation Loop
    # ==============================================================================
    print("[INFO] Running a short simulation with dummy actions...")
    
    dummy_action = [0.0] * 7 
    num_steps = 50
    
    for step in range(num_steps):
        obs, reward, done, info = env.step(dummy_action)
        
        if step == 0:
            print(f"    - Observation keys: {obs.keys()}")
            
        print(f"    - Step {step+1}/{num_steps}: Reward={reward}, Done={done}")

    print("[SUCCESS] Simulation loop completed.\n")
    
    # ==============================================================================
    # (5) Verifying Offscreen Rendering
    # ==============================================================================
    print("[INFO] Verifying offscreen rendering by saving the final observation...")
    
    final_image = obs.get("agentview_image")
    
    if final_image is not None:
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        output_image_path = os.path.join(OUTPUT_DIR, "check_libero_render.png")
        cv2.imwrite(output_image_path, final_image_bgr)
        print(f"[SUCCESS] Final rendered image saved to: {output_image_path}\n")
    else:
        print("[WARNING] Could not retrieve 'agentview_image' from observations.\n")

except Exception as e:
    print(f"\n[ERROR] An error occurred during the environment check: {e}")
    import traceback
    traceback.print_exc()

finally:
    # ==============================================================================
    # (6) Cleaning Up
    # ==============================================================================
    if 'env' in locals() and 'env' in vars():
        env.close()
        print("[INFO] Environment closed.")
    print("="*50)
    print("VALIDATION SCRIPT FINISHED")
    print("="*50)