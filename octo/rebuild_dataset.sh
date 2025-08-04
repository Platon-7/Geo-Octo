#!/bin/bash
# Auto-generated script to rebuild VGGT dataset with optimal compression

echo "🚀 Rebuilding VGGT dataset with the following settings:"
echo "------------------------------------------------"
echo "Method:             PCA"
echo "Target size:        (32, 48)"
echo "Compression Ratio:  348.0x"
echo "Variance Preserved: 0.9953"
echo "------------------------------------------------"

export TF_ENABLE_ONEDNN_OPTS=0

python create_vggt_dataset_compressed.py \
    --input_data_dir="/scratch-shared/tmp.cwkV8vOvfY" \
    --output_data_dir="/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets_compressed" \
    --compression_method="pca" \
    --target_size="32,48" \
    --vggt_batch_size=32 \
    --compression_samples=2500 \
    --overwrite

echo "✅ Dataset rebuild command finished."
echo "New dataset location: /scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets_compressed"
