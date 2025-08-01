#!/usr/bin/env python3
"""
Memory testing script to isolate different components and measure their impact.
"""

import os
import resource
import gc
import tensorflow as tf
import numpy as np
from octo.data.dataset import make_interleaved_dataset
from octo.utils.spec import ModuleSpec

def log_memory(label):
    """Log current memory usage using resource module"""
    gc.collect()
    
    # Get memory usage in MB
    usage = resource.getrusage(resource.RUSAGE_SELF)
    memory_mb = usage.ru_maxrss / 1024  # Convert from KB to MB on Linux
    
    print(f"\n🔍 {label}:")
    print(f"  Peak memory usage: {memory_mb:.1f} MB ({memory_mb/1024:.2f} GB)")
    print(f"  Python objects: {len(gc.get_objects())}")
    
    # Try to get additional system info
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    print(f"  Current RSS: {rss_kb/1024:.1f} MB ({rss_kb/1024/1024:.2f} GB)")
                elif line.startswith('VmSize:'):
                    vms_kb = int(line.split()[1])
                    print(f"  Virtual memory: {vms_kb/1024:.1f} MB ({vms_kb/1024/1024:.2f} GB)")
    except:
        pass

def test_single_dataset_memory():
    """Test memory usage with single dataset"""
    print("=" * 60)
    print("TESTING SINGLE DATASET MEMORY")
    print("=" * 60)
    
    log_memory("Baseline")
    
    # Single dataset config
    UNIFIED_STATS_PATH = "/home/pkarageorgis/geo_octo/libero_datasets/unified_stats/unified_dataset_statistics.json"
    
    single_dataset_kwargs = [{
        "name": "libero_object_vggt",
        "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2",
        "dataset_statistics": UNIFIED_STATS_PATH,
        "standardize_fn": ModuleSpec.create("octo.data.utils.data_utils:standardize_libero_vggt"),
        "image_obs_keys": {"primary": "image_primary"},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        "filter_functions": [],
    }]
    
    log_memory("Before single dataset creation")
    
    single_dataset = make_interleaved_dataset(
        dataset_kwargs_list=single_dataset_kwargs,
        traj_transform_kwargs={},
        frame_transform_kwargs={},
        train=True,
        batch_size=8,
        shuffle_buffer_size=100,
    )
    
    log_memory("After single dataset creation")
    
    # Process and get iterator
    train_data_iter = single_dataset.iterator()
    log_memory("After iterator creation")
    
    # Load a few batches
    for i in range(3):
        batch = next(train_data_iter)
        log_memory(f"After batch {i}")
    
    return single_dataset

def test_multi_dataset_memory():
    """Test memory usage with all 4 datasets"""
    print("=" * 60)
    print("TESTING MULTI-DATASET MEMORY")
    print("=" * 60)
    
    log_memory("Baseline")
    
    # All 4 datasets config
    UNIFIED_STATS_PATH = "/home/pkarageorgis/geo_octo/libero_datasets/unified_stats/unified_dataset_statistics.json"
    
    multi_dataset_kwargs = [
        {
            "name": "libero_object_vggt",
            "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2",
            "dataset_statistics": UNIFIED_STATS_PATH,
            "standardize_fn": ModuleSpec.create("octo.data.utils.data_utils:standardize_libero_vggt"),
            "image_obs_keys": {"primary": "image_primary"},
            "proprio_obs_key": "proprio",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": "normal",
            "filter_functions": [],
        },
        {
            "name": "libero_spatial_vggt",
            "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2",
            "dataset_statistics": UNIFIED_STATS_PATH,
            "standardize_fn": ModuleSpec.create("octo.data.utils.data_utils:standardize_libero_vggt"),
            "image_obs_keys": {"primary": "image_primary"},
            "proprio_obs_key": "proprio",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": "normal",
            "filter_functions": [],
        },
        {
            "name": "libero_goal_vggt",
            "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2",
            "dataset_statistics": UNIFIED_STATS_PATH,
            "standardize_fn": ModuleSpec.create("octo.data.utils.data_utils:standardize_libero_vggt"),
            "image_obs_keys": {"primary": "image_primary"},
            "proprio_obs_key": "proprio",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": "normal",
            "filter_functions": [],
        },
        {
            "name": "liber_o10_vggt",
            "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2",
            "dataset_statistics": UNIFIED_STATS_PATH,
            "standardize_fn": ModuleSpec.create("octo.data.utils.data_utils:standardize_libero_vggt"),
            "image_obs_keys": {"primary": "image_primary"},
            "proprio_obs_key": "proprio",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": "normal",
            "filter_functions": [],
        },
    ]
    
    log_memory("Before multi dataset creation")
    
    multi_dataset = make_interleaved_dataset(
        dataset_kwargs_list=multi_dataset_kwargs,
        traj_transform_kwargs={},
        frame_transform_kwargs={},
        train=True,
        batch_size=8,
        shuffle_buffer_size=100,
    )
    
    log_memory("After multi dataset creation")
    
    # Process and get iterator
    train_data_iter = multi_dataset.iterator()
    log_memory("After iterator creation")
    
    # Load a few batches
    for i in range(3):
        batch = next(train_data_iter)
        log_memory(f"After batch {i}")
    
    return multi_dataset

def test_different_batch_sizes():
    """Test memory usage with different batch sizes"""
    print("=" * 60)
    print("TESTING DIFFERENT BATCH SIZES")
    print("=" * 60)
    
    UNIFIED_STATS_PATH = "/home/pkarageorgis/geo_octo/libero_datasets/unified_stats/unified_dataset_statistics.json"
    
    single_dataset_kwargs = [{
        "name": "libero_object_vggt",
        "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2",
        "dataset_statistics": UNIFIED_STATS_PATH,
        "standardize_fn": ModuleSpec.create("octo.data.utils.data_utils:standardize_libero_vggt"),
        "image_obs_keys": {"primary": "image_primary"},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        "filter_functions": [],
    }]
    
    for batch_size in [4, 8, 16, 32]:
        log_memory(f"Before batch_size={batch_size}")
        
        dataset = make_interleaved_dataset(
            dataset_kwargs_list=single_dataset_kwargs,
            traj_transform_kwargs={},
            frame_transform_kwargs={},
            train=True,
            batch_size=batch_size,
            shuffle_buffer_size=100,
        )
        
        train_data_iter = dataset.iterator()
        batch = next(train_data_iter)
        
        log_memory(f"After batch_size={batch_size} (batch loaded)")
        
        # Clean up
        del dataset, train_data_iter, batch
        gc.collect()

def main():
    """Run all memory tests"""
    tf.config.set_visible_devices([], "GPU")  # Hide GPUs from TensorFlow
    
    print("Starting memory investigation...")
    log_memory("Initial baseline")
    
    # Test 1: Single dataset
    single_dataset = test_single_dataset_memory()
    del single_dataset
    gc.collect()
    
    print("\n" + "="*80 + "\n")
    
    # Test 2: Multi dataset
    multi_dataset = test_multi_dataset_memory()
    del multi_dataset
    gc.collect()
    
    print("\n" + "="*80 + "\n")
    
    # Test 3: Different batch sizes
    test_different_batch_sizes()
    
    print("\n" + "="*80)
    print("MEMORY INVESTIGATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()