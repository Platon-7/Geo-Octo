#!/usr/bin/env python3
"""
Debug version of analysis script to find where it's hanging.
"""

print("🔍 DEBUG: Starting script...")

import os
print("🔍 DEBUG: Imported os")

import numpy as np
print("🔍 DEBUG: Imported numpy")

import sys
print("🔍 DEBUG: Imported sys")
print(f"🔍 DEBUG: Python path: {sys.path[:3]}")

try:
    import tensorflow_datasets as tfds
    print("🔍 DEBUG: Imported tensorflow_datasets successfully")
except Exception as e:
    print(f"❌ DEBUG: tfds import failed: {e}")
    sys.exit(1)

try:
    from vggt_compression_analysis import VGGTCompressor, compare_compression_methods
    print("🔍 DEBUG: Imported vggt_compression_analysis successfully")
except Exception as e:
    print(f"❌ DEBUG: vggt_compression_analysis import failed: {e}")
    print("🔍 DEBUG: Trying without compression analysis...")
    VGGTCompressor = None
    compare_compression_methods = None

print("🔍 DEBUG: All imports complete, starting main function...")

def simple_analysis(dataset_path: str):
    """Simple analysis without heavy compression."""
    
    print(f"🔍 DEBUG: Entering simple_analysis with path: {dataset_path}")
    
    try:
        dataset_names = ['libero_object_vggt', 'libero_spatial_vggt', 'libero_goal_vggt', 'liber_o10_vggt']
        
        for dataset_name in dataset_names:
            print(f"🔍 DEBUG: Trying dataset: {dataset_name}")
            try:
                print(f"🔍 DEBUG: Creating builder for {dataset_name}...")
                builder = tfds.builder(dataset_name, data_dir=dataset_path)
                print(f"🔍 DEBUG: Builder created successfully")
                
                print(f"🔍 DEBUG: Loading dataset...")
                ds = builder.as_dataset(split='train').take(1)
                print(f"🔍 DEBUG: Dataset loaded, iterating...")
                
                for i, episode in enumerate(ds):
                    print(f"🔍 DEBUG: Processing episode {i}")
                    
                    step_count = 0
                    for step in episode['steps']:
                        if step_count >= 3:  # Just check first 3 steps
                            break
                            
                        print(f"🔍 DEBUG: Step {step_count}")
                        vggt_token = step['observation']['vggt_tokens'].numpy()
                        print(f"🔍 DEBUG: VGGT token shape: {vggt_token.shape}")
                        
                        # Calculate size
                        token_size_kb = np.prod(vggt_token.shape) * 2 / 1024
                        print(f"🔍 DEBUG: Memory per timestep: {token_size_kb:.1f} KB")
                        
                        step_count += 1
                    
                    print(f"✅ DEBUG: Successfully analyzed {dataset_name}")
                    return True
                    
            except Exception as e:
                print(f"❌ DEBUG: Failed to load {dataset_name}: {e}")
                continue
    
    except Exception as e:
        print(f"❌ DEBUG: General error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 DEBUG: In main block")
    
    dataset_path = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2"
    print(f"🔍 DEBUG: Dataset path: {dataset_path}")
    
    if os.path.exists(dataset_path):
        print(f"✅ DEBUG: Dataset path exists")
    else:
        print(f"❌ DEBUG: Dataset path does not exist!")
        sys.exit(1)
    
    print("🔍 DEBUG: Starting simple analysis...")
    success = simple_analysis(dataset_path)
    
    if success:
        print("✅ DEBUG: Analysis completed successfully")
        print("💡 DEBUG: Now you can run the full analysis script")
    else:
        print("❌ DEBUG: Analysis failed")