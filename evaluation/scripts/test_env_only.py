#!/usr/bin/env python3
"""
Simplified test script to verify LIBERO environment setup without Octo dependencies.
This script tests the core LIBERO functionality that was working in check_libero.py
"""

import os
import sys
import random

print("="*50)
print("LIBERO ENVIRONMENT TEST (NO OCTO)")
print("="*50)

# Add basic paths - this might help with imports
sys.path.append("/workspace")
sys.path.append("/workspace/LIBERO")

try:
    # Basic LIBERO imports
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    print("[SUCCESS] LIBERO imports successful")
    
    # Configuration
    TASK_SUITE_NAME = "libero_10"
    DATASET_DIR = "/scratch-shared/tmp.cwkV8vOvfY/libero_evaluation"
    LIBERO_DIR = "LIBERO"
    OUTPUT_DIR = "evaluation/test_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize LIBERO environment
    print(f"[INFO] Accessing benchmark suite: {TASK_SUITE_NAME}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME](data_dir=DATASET_DIR)
    
    # Pick a random task
    EVAL_TASK_ID = random.randint(0, task_suite.n_tasks - 1)
    print(f"[INFO] Randomly selected task #{EVAL_TASK_ID}")
    
    task = task_suite.get_task(EVAL_TASK_ID)
    task_name = task.name
    language_instruction = task.language
    
    print(f"[SUCCESS] Retrieved task '{task_name}'")
    print(f"    - Language instruction: '{language_instruction}'")
    
    # Set up environment
    bddl_file_path = os.path.join(LIBERO_DIR, "libero", "libero", "bddl_files", task.problem_folder, task.bddl_file)
    
    if not os.path.exists(bddl_file_path):
        print(f"[ERROR] BDDL file not found: {bddl_file_path}")
        sys.exit(1)
    
    print(f"[SUCCESS] BDDL file found: {bddl_file_path}")
    
    # Initialize environment
    env_args = {"bddl_file_name": bddl_file_path, "camera_heights": 128, "camera_widths": 128}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    
    # Set initial state
    init_states = task_suite.get_task_init_states(EVAL_TASK_ID)
    env.set_init_state(init_states[0])
    
    print("[SUCCESS] Environment initialized and reset")
    
    # Run a few dummy steps
    print("[INFO] Running 10 dummy steps...")
    dummy_action = [0.0] * 7
    
    for step in range(10):
        obs, reward, done, info = env.step(dummy_action)
        print(f"    Step {step+1}: Reward={reward}, Done={done}")
        
        if step == 0:
            print(f"    Observation keys: {list(obs.keys())}")
            if "agentview_image" in obs:
                img_shape = obs["agentview_image"].shape
                print(f"    Image shape: {img_shape}")
    
    env.close()
    print("[SUCCESS] Environment test completed!")
    
    # Print what would be needed for Octo
    print("\n" + "="*50)
    print("OCTO INTEGRATION REQUIREMENTS:")
    print("="*50)
    print(f"1. Model path: {'/home/pkarageorgis/geo_octo/octo/my_octo_vggt_model_offline/octo_vggt_finetune_staged/experiment_20250805_112710_BEST_RUN/150000/default/checkpoint'}")
    print(f"2. Task: {task_name}")
    print(f"3. Language instruction: {language_instruction}")
    print(f"4. Expected observation format:")
    print(f"   - image_primary: shape should be (batch, window, height, width, channels)")
    print(f"   - timestep_pad_mask: shape should be (batch, window)")
    print(f"5. Action space: 7-DOF (robot joints + gripper)")
    print("\nThe evaluation script should work once the conda environment is properly activated!")

except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("\nThis suggests the conda environment is not activated.")
    print("Please run:")
    print("  conda activate octo-eval")
    print("  # or the correct environment name")
    sys.exit(1)
    
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)