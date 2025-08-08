#!/usr/bin/env python3
"""
Debug script to inspect the Octo model and understand language tokenizer issues.
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

from octo.model.octo_model import OctoModel
import json

MODEL_PATH = "/home/pkarageorgis/geo_octo/octo/my_octo_vggt_model_offline/octo_vggt_finetune_staged/experiment_20250805_112710_BEST_RUN"

print("="*60)
print("OCTO MODEL DEBUG")
print("="*60)

try:
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = OctoModel.load_pretrained(MODEL_PATH)
    print("[SUCCESS] Model loaded successfully")
    
    print("\n[DEBUG] Model Configuration:")
    print("-" * 40)
    print(f"Config keys: {list(model.config.keys())}")
    
    if 'model' in model.config:
        model_config = model.config['model']
        print(f"Model config keys: {list(model_config.keys())}")
        
        if 'task_tokenizers' in model_config:
            task_tokenizers = model_config['task_tokenizers']
            print(f"Task tokenizers: {list(task_tokenizers.keys())}")
            
            if 'language' in task_tokenizers:
                print(f"Language tokenizer config: {task_tokenizers['language']}")
            else:
                print("❌ No language tokenizer in config!")
        else:
            print("❌ No task_tokenizers in model config!")
    
    print("\n[DEBUG] Dataset Statistics:")
    print("-" * 40)
    if hasattr(model, 'dataset_statistics') and model.dataset_statistics:
        print(f"Available datasets: {list(model.dataset_statistics.keys())}")
        for dataset_name in model.dataset_statistics.keys():
            dataset_stats = model.dataset_statistics[dataset_name]
            print(f"  {dataset_name}: {list(dataset_stats.keys())}")
    else:
        print("❌ No dataset statistics found!")
    
    print("\n[DEBUG] Example Batch Format:")
    print("-" * 40)
    if hasattr(model, 'example_batch') and model.example_batch:
        print(f"Example batch keys: {list(model.example_batch.keys())}")
        
        if 'observation' in model.example_batch:
            obs_keys = list(model.example_batch['observation'].keys())
            print(f"Observation keys: {obs_keys}")
            for key, value in model.example_batch['observation'].items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
        
        if 'task' in model.example_batch:
            task_keys = list(model.example_batch['task'].keys())
            print(f"Task keys: {task_keys}")
            for key, value in model.example_batch['task'].items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
    else:
        print("❌ No example batch found!")
    
    print("\n[DEBUG] Testing Task Creation:")
    print("-" * 40)
    try:
        test_task = model.create_tasks(texts=["test instruction"])
        print("✅ Task creation with text succeeded")
        print(f"Created task keys: {list(test_task.keys())}")
        for key, value in test_task.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
    except Exception as e:
        print(f"❌ Task creation failed: {e}")
    
    print("\n[DEBUG] Model Parameters:")
    print("-" * 40)
    if hasattr(model, 'params') and model.params:
        def print_param_tree(params, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
            for key, value in params.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}/")
                    print_param_tree(value, prefix + "  ", max_depth, current_depth + 1)
                else:
                    if hasattr(value, 'shape'):
                        print(f"{prefix}{key}: {value.shape}")
                    else:
                        print(f"{prefix}{key}: {type(value)}")
        
        print("Parameter structure (first 3 levels):")
        print_param_tree(model.params)
    else:
        print("❌ No model parameters found!")

except Exception as e:
    print(f"[ERROR] Failed to debug model: {e}")
    import traceback
    traceback.print_exc()

print("="*60)
print("DEBUG FINISHED")
print("="*60)