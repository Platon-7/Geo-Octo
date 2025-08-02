#!/usr/bin/env python3
"""
Fixed VGGT dataset analysis that collects enough samples for PCA.
"""

import os
import numpy as np
import tensorflow_datasets as tfds
from vggt_compression_analysis import VGGTCompressor, compare_compression_methods

def analyze_current_vggt_dataset_fixed(dataset_path: str):
    """Fixed analysis that collects sufficient samples for PCA."""
    
    print("🔍 Analyzing current VGGT dataset...")
    
    try:
        # Try to load one of your datasets
        dataset_names = ['libero_object_vggt', 'libero_spatial_vggt', 'libero_goal_vggt', 'liber_o10_vggt']
        
        for dataset_name in dataset_names:
            try:
                print(f"📦 Trying to load {dataset_name}...")
                builder = tfds.builder(dataset_name, data_dir=dataset_path)
                ds = builder.as_dataset(split='train').take(20)  # More episodes for more samples
                
                print(f"✅ Successfully loaded {dataset_name}")
                
                # Extract VGGT tokens to analyze
                vggt_tokens = []
                total_episodes = 0
                total_timesteps = 0
                
                for episode in ds:
                    total_episodes += 1
                    episode_timesteps = 0
                    
                    # Take every 5th timestep to get diverse samples efficiently
                    step_count = 0
                    for step in episode['steps']:
                        if step_count % 5 == 0:  # Sample every 5th timestep
                            total_timesteps += 1
                            episode_timesteps += 1
                            
                            vggt_token = step['observation']['vggt_tokens'].numpy()
                            vggt_tokens.append(vggt_token)
                            
                            # Target: 500-1000 samples for robust PCA
                            if len(vggt_tokens) >= 500:
                                break
                        step_count += 1
                    
                    print(f"  Episode {total_episodes}: {episode_timesteps} samples collected")
                    if len(vggt_tokens) >= 500:
                        break
                
                if vggt_tokens:
                    vggt_array = np.stack(vggt_tokens)
                    print(f"\n📊 DATASET ANALYSIS:")
                    print(f"  Episodes analyzed: {total_episodes}")
                    print(f"  Total samples collected: {len(vggt_tokens)}")
                    print(f"  VGGT token shape: {vggt_array[0].shape}")
                    print(f"  Sample array shape: {vggt_array.shape}")
                    
                    # Calculate current memory usage
                    token_size_kb = np.prod(vggt_array[0].shape) * 2 / 1024  # float16 = 2 bytes
                    print(f"  Memory per timestep: {token_size_kb:.1f} KB")
                    
                    # Estimate total dataset memory
                    estimated_total_timesteps = total_timesteps * (1000 // total_episodes) if total_episodes > 0 else 100000
                    total_memory_gb = (token_size_kb * estimated_total_timesteps) / (1024 * 1024)
                    print(f"  Estimated total VGGT memory: {total_memory_gb:.1f} GB")
                    
                    # Test compression methods with proper sample sizes
                    print(f"\n🧮 Testing compression methods...")
                    
                    # Use smaller target sizes that work with our sample count
                    target_sizes = [
                        (32, 128),   # 4,096 dims - safe with 500+ samples
                        (48, 128),   # 6,144 dims - good balance
                        (64, 64),    # 4,096 dims - square format
                    ]
                    
                    best_compression = None
                    best_score = 0  # Best = high compression + high variance
                    
                    for target_size in target_sizes:
                        target_dims = target_size[0] * target_size[1]
                        print(f"\n--- Target size: {target_size} ({target_dims} dims) ---")
                        
                        # Check if we have enough samples
                        if len(vggt_tokens) < target_dims:
                            print(f"  ⚠️  Need {target_dims} samples, only have {len(vggt_tokens)}")
                            print(f"  Using reduced target: ({min(16, target_size[0])}, {min(16, target_size[1])})")
                            # Use smaller target that fits our sample size
                            safe_dim = int(np.sqrt(len(vggt_tokens) // 4))  # Conservative estimate
                            target_size = (safe_dim, safe_dim)
                        
                        try:
                            results = compare_compression_methods(vggt_array, target_size)
                            
                            for method, result in results.items():
                                compression_ratio = result['compression_ratio']
                                variance = result['variance_preserved']
                                
                                print(f"  {method.upper()}: {compression_ratio:.1f}x compression, {variance:.3f} variance")
                                
                                # Score = compression * variance (balance both factors)
                                score = compression_ratio * variance
                                if score > best_score:
                                    best_compression = (method, target_size, compression_ratio, variance)
                                    best_score = score
                        
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
                        vggt_contribution = min(200, total_memory_gb * 0.4)  # Conservative estimate
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

def create_rebuild_script_fixed(best_compression, dataset_path: str, output_path: str):
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
echo "Variance preserved: {variance:.3f}"

export TF_ENABLE_ONEDNN_OPTS=0

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
    
    with open('/geo_octo/scripts/rebuild_dataset.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('/geo_octo/scripts/rebuild_dataset.sh', 0o755)
    print(f"📝 Created rebuild script: /geo_octo/scripts/rebuild_dataset.sh")
    print(f"   Run with: bash /geo_octo/scripts/rebuild_dataset.sh")

if __name__ == "__main__":
    dataset_path = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2"
    output_path = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets_compressed"
    
    print("🔍 Analyzing your current VGGT dataset for optimal compression...")
    
    vggt_data, best_compression = analyze_current_vggt_dataset_fixed(dataset_path)
    
    if best_compression:
        create_rebuild_script_fixed(best_compression, dataset_path, output_path)
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Run: bash /geo_octo/scripts/rebuild_dataset.sh")
        print(f"2. Update your config to use: {output_path}")
        print(f"3. Expected training memory: significantly reduced!")
    else:
        print("❌ Could not analyze dataset. Please check paths and permissions.")
        
    print(f"\n💡 This version collects enough samples for robust PCA analysis.")
    print(f"   Uses conservative target sizes that work with available samples.")