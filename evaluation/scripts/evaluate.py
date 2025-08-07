
import sys
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
import tensorflow as tf
import random


# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')

# Octo / LIBERO Imports
from octo.model.octo_model import OctoModel
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# ==============================================================================
# (0) Configuration
# ==============================================================================
# --- REQUIRED PATHS ---
# Path to your fine-tuned Octo model checkpoint directory
MODEL_PATH = "/home/pkarageorgis/geo_octo/octo/my_octo_vggt_model_offline/octo_vggt_finetune_staged/experiment_20250805_112710_BEST_RUN/150000/default/checkpoint"

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
print("OCTO MODEL EVALUATION SCRIPT")
print("="*50)

# ==============================================================================
# (1) Load the Fine-Tuned Octo Model
# ==============================================================================
print(f"\n[INFO] Loading Octo model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"FATAL: Model checkpoint not found at {MODEL_PATH}")

# This loads the model from the checkpoint.
# The 'check_uninitialized_vars=False' is sometimes needed depending on how the model was saved.
model = OctoModel.load_from_checkpoint(MODEL_PATH, check_uninitialized_vars=False)
print("[SUCCESS] Octo model loaded.\n")

# Main try block to ensure the environment is closed properly
try:
    # ==============================================================================
    # (2) Initialize the LIBERO Environment
    # ==============================================================================
    print(f"[INFO] Accessing benchmark suite: {TASK_SUITE_NAME}")
    benchmark_dict = benchmark.get_benchmark_dict()
    
    # CRITICAL: Pass the 'data_dir' to tell LIBERO where your datasets are
    task_suite = benchmark_dict[TASK_SUITE_NAME](data_dir=DATASET_DIR)

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

    # Initialize the simulation environment
    env_args = {"bddl_file_name": bddl_file_path, "camera_heights": 128, "camera_widths": 128}
    env = OffScreenRenderEnv(**env_args)
    
    # ==============================================================================
    # (3) Run the Evaluation Loop
    # ==============================================================================
    print("[INFO] Starting evaluation loop...")
    
    # Reset env and set a deterministic initial state from the dataset
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(EVAL_TASK_ID)
    env.set_init_state(init_states[0]) # Use the first initial state

    # Get the first observation
    obs, _, _, _ = env.step([0.0] * 7)
    
    # List to store frames for video
    frames = []

    for step in range(NUM_TIMESTEPS):
        # Prepare observation for the model
        image = obs["agentview_image"]
        
        # Add a batch dimension to the image (from H,W,C to 1,H,W,C)
        model_observation = {"image_primary": np.expand_dims(image, axis=0)}

        # Prepare the language instruction for the model
        task_payload = {"language_instruction": tf.constant([language_instruction])}

        # Get action from the model
        # The .sample() method is used for inference (i.e., getting a single action)
        predicted_action = model.sample(model_observation, task_payload)
        
        # Convert action to a numpy array and remove the batch dimension
        predicted_action = predicted_action.numpy().squeeze()
        
        # Step the environment with the model's action
        obs, reward, done, info = env.step(predicted_action)
        
        # Render the frame for the video
        # We need to render explicitly to get the most up-to-date image after the step
        current_frame = obs["agentview_image"]
        frames.append(cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)) # Convert to BGR for OpenCV

        print(f"    - Step {step+1}/{NUM_TIMESTEPS}: Reward={reward}, Done={done}")

        if done:
            print("[INFO] Task succeeded! Episode finished early.")
            break
            
    print("[SUCCESS] Evaluation loop completed.\n")

    # ==============================================================================
    # (4) Save Video of the Episode
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
    # (5) Clean Up
    # ==============================================================================
    if 'env' in locals() and 'env' in vars():
        env.close()
        print("[INFO] Environment closed.")
    print("="*50)
    print("EVALUATION SCRIPT FINISHED")
    print("="*50)