#!/usr/bin/env python3
"""
Memory optimization utilities for Octo training.
This script provides functions to optimize memory usage during dataset loading and training.
"""

import os
import gc
import tensorflow as tf
import psutil
from typing import Optional

def optimize_tensorflow_memory():
    """Configure TensorFlow for memory-efficient operation."""
    # Limit TensorFlow GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")
    
    # Configure TensorFlow to use less aggressive prefetching
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

def set_memory_env_variables():
    """Set environment variables for better memory management."""
    # Limit XLA memory allocation
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'  # Reduced from 0.8 to 0.5
    
    # Use more conservative TensorFlow memory allocation
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Reduce TensorFlow's internal buffer sizes
    os.environ['TF_BUFFER_SIZE'] = '128'  # Reduce buffer size
    
def force_cleanup():
    """Force garbage collection and memory cleanup."""
    gc.collect()
    # Try to force TensorFlow cleanup if available
    try:
        tf.keras.backend.clear_session()
    except:
        pass

def get_memory_usage_gb() -> float:
    """Get current memory usage in GB."""
    return psutil.virtual_memory().used / (1024**3)

def check_memory_limit(max_gb: float = 200.0) -> bool:
    """Check if memory usage is within limits."""
    current_usage = get_memory_usage_gb()
    if current_usage > max_gb:
        print(f"WARNING: Memory usage ({current_usage:.1f}GB) exceeds limit ({max_gb}GB)")
        return False
    return True

def optimize_dataset_config(config):
    """Optimize dataset configuration for memory efficiency."""
    # Reduce buffer sizes dramatically
    config.shuffle_buffer_size = min(config.shuffle_buffer_size, 5)
    config.val_kwargs.val_shuffle_buffer_size = min(config.val_kwargs.val_shuffle_buffer_size, 3)
    
    # Reduce batch size if too large
    if config.batch_size > 4:
        print(f"WARNING: Large batch size ({config.batch_size}) may cause memory issues")
        print("Consider reducing batch_size to 4 or lower")
    
    return config

def create_memory_efficient_dataset_kwargs(original_kwargs_list, max_datasets: Optional[int] = 2):
    """Create memory-efficient version of dataset kwargs."""
    # Limit number of concurrent datasets if too many
    if max_datasets and len(original_kwargs_list) > max_datasets:
        print(f"WARNING: Using only first {max_datasets} datasets to reduce memory usage")
        return original_kwargs_list[:max_datasets]
    
    # Optimize individual dataset configs
    optimized_kwargs = []
    for kwargs in original_kwargs_list:
        # Force use of pre-computed statistics to avoid recomputation
        optimized_kwargs.append({
            **kwargs,
            # Ensure we're using cached statistics
            'force_recompute_dataset_statistics': False,
        })
    
    return optimized_kwargs

def monitor_memory_during_training(step: int, threshold_gb: float = 300.0):
    """Monitor memory usage during training and warn if too high."""
    current_usage = get_memory_usage_gb()
    if current_usage > threshold_gb:
        print(f"CRITICAL: Step {step} - Memory usage {current_usage:.1f}GB exceeds threshold {threshold_gb}GB")
        print("Consider stopping training and applying more aggressive memory optimizations")
        return False
    elif current_usage > threshold_gb * 0.8:
        print(f"WARNING: Step {step} - Memory usage {current_usage:.1f}GB approaching limit")
        force_cleanup()
    return True

def get_recommended_prefetch_size(available_memory_gb: float, batch_size: int, num_datasets: int) -> int:
    """
    Recommend prefetch size based on available memory and configuration.
    
    Args:
        available_memory_gb: Available RAM in GB
        batch_size: Training batch size
        num_datasets: Number of datasets being used
    
    Returns:
        Recommended prefetch size
    """
    # Rough estimation: each batch uses ~1-3GB depending on model/data
    estimated_batch_memory_gb = batch_size * 0.2  # Conservative estimate
    total_memory_for_prefetch = estimated_batch_memory_gb * num_datasets
    
    if available_memory_gb > 200:
        # Plenty of memory - can use higher prefetch
        recommended = min(8, max(2, int(available_memory_gb / (total_memory_for_prefetch * 10))))
    elif available_memory_gb > 100:
        # Moderate memory - balanced approach
        recommended = min(4, max(2, int(available_memory_gb / (total_memory_for_prefetch * 20))))
    else:
        # Low memory - conservative
        recommended = 2
    
    print(f"💡 Recommended prefetch size: {recommended}")
    print(f"   (Based on {available_memory_gb:.0f}GB available, batch_size={batch_size}, {num_datasets} datasets)")
    return recommended

def adjust_prefetch_for_performance(current_prefetch: int, gpu_utilization: float) -> int:
    """
    Adjust prefetch based on GPU utilization.
    
    Args:
        current_prefetch: Current prefetch setting
        gpu_utilization: Average GPU utilization (0.0-1.0)
    
    Returns:
        Suggested new prefetch size
    """
    if gpu_utilization < 0.7:  # GPU underutilized
        suggested = min(current_prefetch + 1, 8)
        print(f"🚀 GPU utilization low ({gpu_utilization:.1%}), consider increasing prefetch to {suggested}")
        return suggested
    elif gpu_utilization > 0.95:  # GPU well utilized
        print(f"✅ GPU utilization good ({gpu_utilization:.1%}), prefetch={current_prefetch} is optimal")
        return current_prefetch
    else:
        print(f"👍 GPU utilization acceptable ({gpu_utilization:.1%}), prefetch={current_prefetch} is fine")
        return current_prefetch

if __name__ == "__main__":
    print("Memory optimization utilities loaded")
    print(f"Current memory usage: {get_memory_usage_gb():.1f}GB")
    set_memory_env_variables()
    optimize_tensorflow_memory()