#!/usr/bin/env python3
"""
Fixed VGGT dataset analysis that collects enough samples for PCA.
"""

import os
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

print("🔍 DEBUG: About to import vggt_compression_analysis...")

# Import the new, focused PCA analysis function
from vggt_compression_analysis import analyze_pca_compression

print("🔍 DEBUG: Imported vggt_compression_analysis")

def analyze_current_vggt_dataset_fixed(dataset_path: str):
    """Fixed analysis that collects sufficient samples for PCA."""
    
    print("🔍 Analyzing current VGGT dataset...")
    
    try:
        # Corrected the typo in 'libero_10_vggt'
        dataset_names = ['liber_o10_vggt', 'libero_object_vggt','libero_spatial_vggt', 'libero_goal_vggt']
        
        for dataset_name in dataset_names:
            try:
                print(f"📦 Trying to load {dataset_name}...")
                
                print(f"  [DEBUG] Calling tfds.builder for {dataset_name} at {dataset_path}")
                builder = tfds.builder(dataset_name, data_dir=dataset_path)
                print(f"  [DEBUG] TFDS builder created. Generating dataset iterator...")
                # Read from more episodes to get enough samples for larger dimensions
                ds = builder.as_dataset(split='train').take(150)
                print(f"  [DEBUG] Dataset iterator is ready. Extracting samples...")

                vggt_tokens = []
                total_episodes = 0
                SAMPLE_GOAL = 2500
                
                for episode in ds:
                    total_episodes += 1
                    episode_timesteps = 0
                    
                    step_count = 0
                    for step in episode['steps']:
                        if step_count % 5 == 0:
                            episode_timesteps += 1
                            vggt_token = step['observation']['vggt_tokens'].numpy()
                            vggt_tokens.append(vggt_token)
                            if len(vggt_tokens) >= SAMPLE_GOAL:
                                break
                        step_count += 1
                    
                    print(f"  Episode {total_episodes}: {episode_timesteps} samples collected ({len(vggt_tokens)} / {SAMPLE_GOAL})")
                    if len(vggt_tokens) >= SAMPLE_GOAL:
                        break
                
                if not vggt_tokens:
                    print("  No samples collected. Skipping to next dataset.")
                    continue

                vggt_array = np.stack(vggt_tokens)
                print(f"\n📊 DATASET ANALYSIS:")
                print(f"  Total samples collected: {len(vggt_tokens)}")
                print(f"  Sample array shape: {vggt_array.shape}")
                
                token_size_kb = np.prod(vggt_array[0].shape) * 2 / 1024
                print(f"  Memory per timestep: {token_size_kb:.1f} KB")
                
                estimated_total_timesteps = 100000 
                total_memory_gb = (token_size_kb * estimated_total_timesteps) / (1024 * 1024)
                print(f"  Estimated total VGGT memory for a full dataset: ~{total_memory_gb:.1f} GB")
                
                print(f"\n🧮 Testing PCA compression with different target dimensions...")
                
                # --- CHANGE: New target sizes to search for the ~98% variance sweet spot ---
                target_sizes = [
                    (32, 48),    # 1536 dims
                ]
                
                best_compression = None
                best_score = 0

                for target_size in target_sizes:
                    target_dims = target_size[0] * target_size[1]
                    
                    if len(vggt_tokens) < target_dims:
                        print(f"\n--- Target size: {target_size} ({target_dims} dims) ---")
                        print(f"  ⚠️  SKIPPING: Need at least {target_dims} samples, but only have {len(vggt_tokens)}.")
                        continue
                    
                    try:
                        # Call the new simplified PCA analysis function
                        result = analyze_pca_compression(vggt_array, target_size)
                        
                        compression_ratio = result['compression_ratio']
                        variance = result['variance_preserved']
                        print(f"  RESULT: {compression_ratio:.1f}x compression, {variance:.4f} variance")
                        
                        # Score balances high compression with high variance preservation
                        score = compression_ratio * variance
                        if score > best_score:
                            best_compression = ('pca', target_size, compression_ratio, variance)
                            best_score = score
                            
                    except Exception as e:
                        print(f"  Error testing {target_size}: {e}")
                
                print(f"\n🏆 RECOMMENDATIONS:")
                if best_compression:
                    method, target_size, ratio, variance = best_compression
                    new_memory_gb = total_memory_gb / ratio
                    print(f"  Best method found: {method.upper()}")
                    print(f"  Optimal target size: {target_size}")
                    print(f"  Compression ratio: {ratio:.1f}x")
                    print(f"  Variance preserved: {variance:.4f}")
                    print(f"  Estimated Memory Reduction: {total_memory_gb:.1f} GB → {new_memory_gb:.1f} GB")
                
                return vggt_array, best_compression
            
            except tfds.core.DatasetNotFoundError:
                print(f"❌ Dataset '{dataset_name}' not found at the specified path. Skipping.")
                continue
            except Exception as e:
                print(f"❌ An unexpected error occurred with {dataset_name}: {e}")
                continue
    
    except Exception as e:
        print(f"❌ A general error occurred during analysis: {e}")
        return None, None

def create_rebuild_script_fixed(best_compression, dataset_path: str, output_path: str):
    """Create a script to rebuild the dataset with optimal compression."""
    
    if not best_compression:
        print("❌ No compression recommendation available to generate script.")
        return
    
    method, target_size, ratio, variance = best_compression
    input_base_dir = os.path.dirname(dataset_path)
    
    script_content = f'''#!/bin/bash
# Auto-generated script to rebuild VGGT dataset with optimal compression

echo "🚀 Rebuilding VGGT dataset with the following settings:"
echo "------------------------------------------------"
echo "Method:             {method.upper()}"
echo "Target size:        {target_size}"
echo "Compression Ratio:  {ratio:.1f}x"
echo "Variance Preserved: {variance:.4f}"
echo "------------------------------------------------"

export TF_ENABLE_ONEDNN_OPTS=0

python create_vggt_dataset_compressed.py \\
    --input_data_dir="{input_base_dir}" \\
    --output_data_dir="{output_path}" \\
    --compression_method="{method}" \\
    --target_size="{target_size[0]},{target_size[1]}" \\
    --vggt_batch_size=32 \\
    --compression_samples=2500 \\
    --overwrite

echo "✅ Dataset rebuild command finished."
echo "New dataset location: {output_path}"
'''
    
    script_path = './rebuild_dataset.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"📝 Created rebuild script in the current directory: {script_path}")
    print(f"   To run it, execute: bash {script_path}")

if __name__ == "__main__":
    dataset_path = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2"
    output_path = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets_compressed"
    
    print("🔍 Analyzing your current VGGT dataset for optimal compression...")
    
    vggt_data, best_compression = analyze_current_vggt_dataset_fixed(dataset_path)
    
    if best_compression:
        create_rebuild_script_fixed(best_compression, dataset_path, output_path)
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Review the recommendation above.")
        print(f"2. Run the generated script: bash ./rebuild_dataset.sh")
        print(f"3. Update your training configuration to use the new dataset path: {output_path}")
    else:
        print("\n❌ Could not determine a best compression method. Please check the logs for errors.")
        print("   Common issues include dataset path errors or not enough samples collected.")