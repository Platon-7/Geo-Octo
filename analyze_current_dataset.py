#!/usr/bin/env python3
"""
Quick analysis of your current VGGT dataset to determine compression strategy.
"""

import os
import numpy as np
import tensorflow_datasets as tfds
from vggt_compression_analysis import VGGTCompressor, compare_compression_methods

def analyze_current_vggt_dataset(dataset_path: str):
    """Analyze your current VGGT dataset to understand the memory impact."""
    
    print("🔍 Analyzing current VGGT dataset...")
    
    try:
        # Try to load one of your datasets
        dataset_names = ['libero_object_vggt', 'libero_spatial_vggt', 'libero_goal_vggt', 'liber_o10_vggt']
        
        for dataset_name in dataset_names:
            try:
                print(f"📦 Trying to load {dataset_name}...")
                builder = tfds.builder(dataset_name, data_dir=dataset_path)
                ds = builder.as_dataset(split='train').take(5)  # Just 5 episodes
                
                print(f"✅ Successfully loaded {dataset_name}")
                
                # Extract VGGT tokens to analyze
                vggt_tokens = []
                total_episodes = 0
                total_timesteps = 0
                
                for episode in ds:
                    total_episodes += 1
                    episode_timesteps = 0
                    
                    for step in episode['steps']:
                        total_timesteps += 1
                        episode_timesteps += 1
                        
                        vggt_token = step['observation']['vggt_tokens'].numpy()
                        vggt_tokens.append(vggt_token)
                        
                        if len(vggt_tokens) >= 100:  # Enough samples for analysis
                            break
                    
                    print(f"  Episode {total_episodes}: {episode_timesteps} timesteps")
                    if len(vggt_tokens) >= 100:
                        break
                
                if vggt_tokens:
                    vggt_array = np.stack(vggt_tokens)
                    print(f"\n📊 DATASET ANALYSIS:")
                    print(f"  Episodes analyzed: {total_episodes}")
                    print(f"  Total timesteps: {total_timesteps}")
                    print(f"  VGGT token shape: {vggt_array[0].shape}")
                    print(f"  Sample array shape: {vggt_array.shape}")
                    
                    # Calculate current memory usage
                    token_size_kb = np.prod(vggt_array[0].shape) * 2 / 1024  # float16 = 2 bytes
                    print(f"  Memory per timestep: {token_size_kb:.1f} KB")
                    
                    # Estimate total dataset memory
                    estimated_timesteps = total_timesteps * (1000 // total_episodes)  # Rough estimate
                    total_memory_gb = (token_size_kb * estimated_timesteps) / (1024 * 1024)
                    print(f"  Estimated total VGGT memory: {total_memory_gb:.1f} GB")
                    
                    # Test compression methods
                    print(f"\n🧮 Testing compression methods...")
                    target_sizes = [(64, 256), (32, 512), (128, 128)]
                    
                    best_compression = None
                    best_ratio = 0
                    
                    for target_size in target_sizes:
                        print(f"\n--- Target size: {target_size} ---")
                        
                        try:
                            results = compare_compression_methods(vggt_array, target_size)
                            
                            for method, result in results.items():
                                compression_ratio = result['compression_ratio']
                                variance = result['variance_preserved']
                                
                                print(f"  {method.upper()}: {compression_ratio:.1f}x compression, {variance:.3f} variance")
                                
                                if variance > 0.8 and compression_ratio > best_ratio:  # Good variance + compression
                                    best_compression = (method, target_size, compression_ratio, variance)
                                    best_ratio = compression_ratio
                        
                        except Exception as e:
                            print(f"  Error testing {target_size}: {e}")
                    
                    # Recommendations
                    print(f"\n🏆 RECOMMENDATIONS:")
                    if best_compression:
                        method, target_size, ratio, variance = best_compression
                        new_memory_gb = total_memory_gb / ratio
                        
                        print(f"  Best method: {method.upper()}")
                        print(f"  Target size: {target_size}")
                        print(f"  Compression: {ratio:.1f}x")
                        print(f"  Variance preserved: {variance:.3f}")
                        print(f"  Memory reduction: {total_memory_gb:.1f}GB → {new_memory_gb:.1f}GB")
                        
                        # Expected training memory reduction
                        current_training_memory = 380  # From your tests
                        vggt_contribution = min(200, total_memory_gb * 4)  # Estimate VGGT's contribution
                        reduced_contribution = vggt_contribution / ratio
                        new_training_memory = current_training_memory - vggt_contribution + reduced_contribution
                        
                        print(f"\n💡 EXPECTED TRAINING MEMORY IMPACT:")
                        print(f"  Current training memory: ~{current_training_memory}GB")
                        print(f"  Estimated VGGT contribution: ~{vggt_contribution:.0f}GB")
                        print(f"  After compression: ~{new_training_memory:.0f}GB")
                        
                        if new_training_memory < 250:
                            print(f"  ✅ Should fit in reasonable memory limits!")
                        else:
                            print(f"  ⚠️  May still need additional optimizations")
                    
                    return vggt_array, best_compression
                
                break  # Found working dataset
                
            except Exception as e:
                print(f"❌ Could not load {dataset_name}: {e}")
                continue
    
    except Exception as e:
        print(f"❌ General error: {e}")
        return None, None

def create_rebuild_script(best_compression, dataset_path: str, output_path: str):
    """Create a script to rebuild the dataset with optimal compression."""
    
    if not best_compression:
        print("❌ No compression recommendation available")
        return
    
    method, target_size, ratio, variance = best_compression
    
    script_content = f'''#!/bin/bash
# Auto-generated script to rebuild VGGT dataset with compression

echo "🚀 Rebuilding VGGT dataset with compression..."
echo "Method: {method.upper()}"
echo "Target size: {target_size}"
echo "Expected compression: {ratio:.1f}x"

python create_vggt_dataset_compressed.py \\
    --input_data_dir="{dataset_path.replace('/libero_vggt_datasets2', '')}" \\
    --output_data_dir="{output_path}" \\
    --compression_method="{method}" \\
    --target_size="{target_size[0]},{target_size[1]}" \\
    --vggt_batch_size=32 \\
    --compression_samples=1000 \\
    --overwrite

echo "✅ Dataset rebuild complete!"
echo "Expected memory reduction: {ratio:.1f}x"
echo "New dataset location: {output_path}"
'''
    
    with open('/workspace/rebuild_dataset.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('/workspace/rebuild_dataset.sh', 0o755)
    print(f"📝 Created rebuild script: /workspace/rebuild_dataset.sh")
    print(f"   Run with: bash /workspace/rebuild_dataset.sh")

if __name__ == "__main__":
    dataset_path = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2"
    output_path = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets_compressed"
    
    print("🔍 Analyzing your current VGGT dataset for optimal compression...")
    
    vggt_data, best_compression = analyze_current_vggt_dataset(dataset_path)
    
    if best_compression:
        create_rebuild_script(best_compression, dataset_path, output_path)
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Run: bash /workspace/rebuild_dataset.sh")
        print(f"2. Update your config to use: {output_path}")
        print(f"3. Expected training memory: ~200-250GB (down from 380GB)")
    else:
        print("❌ Could not analyze dataset. Please check paths and permissions.")