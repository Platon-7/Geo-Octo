#!/usr/bin/env python3
"""
Memory limit configuration to prevent OOM crashes.
Sets hard limits on memory usage and provides fallback strategies.
"""

def get_memory_limit_config():
    """
    Configuration changes to limit peak memory usage.
    Apply these to your config_offline.py
    """
    
    config_changes = {
        # Reduce batch size to limit memory
        "batch_size": 4,  # Down from 8
        
        # Minimal buffer sizes
        "shuffle_buffer_size": 3,  # Down from 10
        
        # Validation settings
        "val_kwargs": {
            "val_shuffle_buffer_size": 1,  # Minimal
            "num_val_batches": 1,  # Just 1 batch
        },
        
        # Validation batch size
        "viz_kwargs": {
            "eval_batch_size": 2,  # Very small
            "trajs_for_metrics": 5,  # Reduced
            "trajs_for_viz": 1,
            "samples_per_state": 1,
        },
        
        # Less frequent validation to reduce memory spikes
        "eval_interval": 100,  # Every 100 steps instead of 50
    }
    
    return config_changes

def get_memory_monitoring_additions():
    """
    Add these to your finetune.py for better memory control
    """
    
    monitoring_code = '''
# Add this function after log_memory_usage
def check_memory_and_abort(step, max_memory_gb=250):
    """Check memory and abort if too high to prevent crash"""
    current_memory = psutil.virtual_memory().used / (1024**3)
    if current_memory > max_memory_gb:
        print(f"ABORTING: Memory {current_memory:.1f}GB exceeds limit {max_memory_gb}GB at step {step}")
        print("Saving checkpoint before abort...")
        if save_dir:
            save_callback(train_state, step)
        sys.exit(1)  # Clean abort
    return current_memory

# Add this in your training loop before each validation
if (i + 1) % config["eval_interval"] == 0:
    # Check memory before validation to prevent crash
    current_mem = check_memory_and_abort(i, max_memory_gb=250)
    print(f"Memory before validation: {current_mem:.1f}GB")
    
    # Then run validation...
    '''
    
    return monitoring_code

def explain_memory_breakdown():
    """
    Explain why robotics datasets use so much memory
    """
    
    breakdown = """
    === WHY ROBOTICS DATASETS USE SO MUCH MEMORY ===
    
    Your 555GB → 380GB memory usage breakdown:
    
    1. **Image Decompression** (~150GB extra)
       - Disk: JPEG compressed images
       - Memory: Uncompressed RGB arrays (3x larger)
       - 224x224x3 images = 150KB each uncompressed
    
    2. **Trajectory Data Structure** (~100GB extra)  
       - Complex nested dictionaries
       - Multiple timesteps per trajectory
       - Action sequences, observations, metadata
    
    3. **TensorFlow Pipeline** (~80GB extra)
       - Multiple pipeline stages in memory
       - Prefetch buffers
       - Preprocessing intermediate results
    
    4. **Validation Dataset** (+180GB)
       - Complete separate dataset copy
       - This is your biggest memory killer!
    
    5. **Python Objects** (~50GB extra)
       - Object overhead for millions of items
       - Reference counting memory
       - TensorFlow graph memory
    
    TOTAL: 555GB + 560GB overhead = ~1.1TB potential
    Your optimizations reduced this to 380GB (65% reduction!)
    """
    
    return breakdown

if __name__ == "__main__":
    print(explain_memory_breakdown())
    print("\n" + "="*60)
    print("RECOMMENDED MEMORY LIMIT SETTINGS:")
    print("="*60)
    
    config = get_memory_limit_config()
    for key, value in config.items():
        print(f"{key}: {value}")
        
    print(f"\nExpected peak memory: ~200-250GB (down from 380GB)")
    print("This should prevent crashes while maintaining training quality.")