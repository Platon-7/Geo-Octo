#!/usr/bin/env python3
"""
Debug script to inspect task creation formats.
"""

import sys
import warnings
import os

# Add compatibility shim
try:
    import jax.numpy as jnp
    if not hasattr(jnp, 'DeviceArray'):
        jnp.DeviceArray = jnp.ndarray
        print("[FIX] Added DeviceArray compatibility shim")
except ImportError:
    print("[WARNING] Could not import JAX")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import cv2
from octo.model.octo_model import OctoModel

MODEL_PATH = "/home/pkarageorgis/geo_octo/octo/my_octo_vggt_model_offline/octo_vggt_finetune_staged/experiment_20250805_112710_BEST_RUN"

print("="*60)
print("TASK CREATION DEBUG")
print("="*60)

try:
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = OctoModel.load_pretrained(MODEL_PATH)
    print("[SUCCESS] Model loaded successfully")
    
    # Create a dummy goal image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    print("\n[DEBUG] Testing Language Task Creation:")
    print("-" * 40)
    lang_task = model.create_tasks(texts=["test instruction"])
    print(f"Language task keys: {list(lang_task.keys())}")
    for key, value in lang_task.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape'):
                    print(f"    {subkey}: {subvalue.shape}")
    
    print("\n[DEBUG] Testing Goal Image Task Creation:")
    print("-" * 40)
    goal_task = model.create_tasks(goals={"image_primary": dummy_image[None]})
    print(f"Goal task keys: {list(goal_task.keys())}")
    for key, value in goal_task.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape'):
                    print(f"    {subkey}: {subvalue.shape}")
    
    print("\n[DEBUG] Model Example Batch (Task):")
    print("-" * 40)
    if hasattr(model, 'example_batch') and 'task' in model.example_batch:
        task_batch = model.example_batch['task']
        print(f"Example task keys: {list(task_batch.keys())}")
        for key, value in task_batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'shape'):
                        print(f"    {subkey}: {subvalue.shape}")
    else:
        print("❌ No example batch found!")

except Exception as e:
    print(f"[ERROR] Failed to debug: {e}")
    import traceback
    traceback.print_exc()

print("="*60)
print("DEBUG FINISHED")
print("="*60)