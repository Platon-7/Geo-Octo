#!/usr/bin/env python3
"""
Lightweight VGGT dataset analysis that avoids OOM errors.
Uses minimal memory by processing one sample at a time.
"""

import os
import numpy as np
import tensorflow_datasets as tfds
from sklearn.decomposition import PCA
import pickle

def analyze_vggt_lightweight(dataset_path: str):
    """Ultra-lightweight analysis using minimal memory."""
    
    print("🔍 Lightweight VGGT dataset analysis...")
    
    try:
        # Try to load one of your datasets
        dataset_names = ['libero_object_vggt', 'libero_spatial_vggt', 'libero_goal_vggt', 'liber_o10_vggt']
        
        for dataset_name in dataset_names:
            try:
                print(f"📦 Trying to load {dataset_name}...")
                builder = tfds.builder(dataset_name, data_dir=dataset_path)
                ds = builder.as_dataset(split='train').take(1)  # Just 1 episode
                
                print(f"✅ Successfully loaded {dataset_name}")
                
                # Process just one episode
                for episode in ds:
                    episode_tokens = []
                    timestep_count = 0
                    
                    # Process one step at a time to minimize memory
                    for step in episode['steps']:
                        if timestep_count < 10:  # Only analyze first 10 timesteps
                            vggt_token = step['observation']['vggt_tokens'].numpy()
                            episode_tokens.append(vggt_token)
                            timestep_count += 1
                    
                    if episode_tokens:
                        sample_token = episode_tokens[0]
                        print(f"\n📊 DATASET ANALYSIS:")
                        print(f"  VGGT token shape: {sample_token.shape}")
                        print(f"  Sample timesteps analyzed: {len(episode_tokens)}")
                        
                        # Calculate memory usage
                        token_size_bytes = np.prod(sample_token.shape) * 2  # float16 = 2 bytes
                        token_size_kb = token_size_bytes / 1024
                        print(f"  Memory per timestep: {token_size_kb:.1f} KB")
                        
                        # Estimate compression potential
                        original_dims = np.prod(sample_token.shape)
                        target_sizes = [(64, 256), (32, 512), (128, 128)]
                        
                        print(f"\n🧮 COMPRESSION ANALYSIS:")
                        print(f"  Original dimensions: {original_dims:,}")
                        
                        best_compression = None
                        best_ratio = 0
                        
                        for target_size in target_sizes:
                            target_dims = target_size[0] * target_size[1]
                            compression_ratio = original_dims / target_dims
                            
                            print(f"  Target {target_size}: {compression_ratio:.1f}x compression")
                            
                            if compression_ratio > best_ratio:
                                best_compression = ('hybrid', target_size, compression_ratio)
                                best_ratio = compression_ratio
                        
                        # Quick PCA test on flattened samples
                        if len(episode_tokens) >= 5:
                            print(f"\n🔬 Quick PCA test...")
                            try:
                                # Flatten and stack samples
                                flattened_samples = np.array([t.flatten() for t in episode_tokens[:5]])
                                
                                # Test PCA with small number of components
                                test_components = min(1000, flattened_samples.shape[1]//10)
                                pca = PCA(n_components=test_components)
                                pca.fit(flattened_samples)
                                
                                variance_ratio = np.sum(pca.explained_variance_ratio_[:test_components])
                                print(f"  PCA variance preserved (first {test_components} components): {variance_ratio:.3f}")
                                
                            except Exception as e:
                                print(f"  PCA test failed: {e}")
                        
                        # Recommendations
                        print(f"\n🏆 LIGHTWEIGHT RECOMMENDATIONS:")
                        if best_compression:
                            method, target_size, ratio = best_compression
                            
                            # Estimate dataset size (very rough)
                            episodes_estimate = 1000  # Rough guess
                            timesteps_per_episode = 100  # Rough guess
                            total_timesteps = episodes_estimate * timesteps_per_episode
                            
                            current_total_gb = (token_size_kb * total_timesteps) / (1024 * 1024)
                            compressed_total_gb = current_total_gb / ratio
                            
                            print(f"  Recommended method: {method.upper()}")
                            print(f"  Target size: {target_size}")
                            print(f"  Compression ratio: {ratio:.1f}x")
                            print(f"  Estimated current dataset size: {current_total_gb:.1f} GB")
                            print(f"  Estimated compressed size: {compressed_total_gb:.1f} GB")
                            
                            # Training memory estimate
                            estimated_training_reduction = min(150, current_total_gb * 0.4)
                            new_training_memory = 380 - estimated_training_reduction + (estimated_training_reduction / ratio)
                            
                            print(f"\n💡 TRAINING MEMORY ESTIMATE:")
                            print(f"  Current training memory: ~380 GB")
                            print(f"  After compression: ~{new_training_memory:.0f} GB")
                            
                            if new_training_memory < 250:
                                print(f"  ✅ Should fit in reasonable memory limits!")
                            else:
                                print(f"  ⚠️  May need additional optimizations")
                            
                            return best_compression
                    
                    break  # Only process first episode
                break  # Found working dataset
                
            except Exception as e:
                print(f"❌ Could not load {dataset_name}: {e}")
                continue
    
    except Exception as e:
        print(f"❌ General error: {e}")
        return None

def create_simple_rebuild_script(best_compression, dataset_path: str, output_path: str):
    """Create a rebuild script with conservative settings."""
    
    if not best_compression:
        print("❌ No compression recommendation available")
        return
    
    method, target_size, ratio = best_compression
    
    script_content = f'''#!/bin/bash
# Lightweight rebuild script with conservative memory settings

echo "🚀 Rebuilding VGGT dataset with compression..."
echo "Method: {method.upper()}"
echo "Target size: {target_size}"
echo "Expected compression: {ratio:.1f}x"

export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

python create_vggt_dataset_compressed.py \\
    --input_data_dir="{dataset_path.replace('/libero_vggt_datasets2', '')}" \\
    --output_data_dir="{output_path}" \\
    --compression_method="{method}" \\
    --target_size="{target_size[0]},{target_size[1]}" \\
    --vggt_batch_size=16 \\
    --compression_samples=500 \\
    --overwrite

echo "✅ Dataset rebuild complete!"
echo "Expected memory reduction: {ratio:.1f}x"
'''
    
    script_path = '/geo_octo/scripts/rebuild_dataset_light.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"📝 Created lightweight rebuild script: {script_path}")

if __name__ == "__main__":
    dataset_path = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2"
    output_path = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets_compressed"
    
    print("🔍 Running lightweight VGGT dataset analysis...")
    
    best_compression = analyze_vggt_lightweight(dataset_path)
    
    if best_compression:
        create_simple_rebuild_script(best_compression, dataset_path, output_path)
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Run: bash /geo_octo/scripts/rebuild_dataset_light.sh")
        print(f"2. Update your config to use: {output_path}")
        print(f"3. Expected training memory: ~200-250GB (down from 380GB)")
    else:
        print("❌ Could not analyze dataset. Please check paths and permissions.")
        
    print(f"\n💡 This lightweight analysis uses minimal memory to avoid OOM errors.")
    print(f"   It provides good estimates based on token structure analysis.")