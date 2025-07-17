#!/bin/bash

# Script to run VGGT finetuning
# Make sure to update the paths and parameters according to your setup

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Update this to use the appropriate GPU

# Update these paths according to your setup:
PRETRAINED_PATH="/path/to/pretrained/octo/model"  # Update this path
PRETRAINED_STEP=1000000  # Update this step number
SAVE_DIR="/workspace/vggt_finetuned_models"  # Update this path
WANDB_PROJECT="octo_vggt_finetune"
WANDB_GROUP="libero_experiments"
WANDB_ENTITY="your_wandb_entity"  # Update this

echo "Starting VGGT finetuning..."
echo "Make sure to update the paths in this script before running!"

# Create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Run the finetuning
cd octo/scripts

python finetune.py \
    --name "vggt_libero_finetune" \
    --config.pretrained_path="$PRETRAINED_PATH" \
    --config.pretrained_step=$PRETRAINED_STEP \
    --config.save_dir="$SAVE_DIR" \
    --config.wandb.project="$WANDB_PROJECT" \
    --config.wandb.group="$WANDB_GROUP" \
    --config.wandb.entity="$WANDB_ENTITY" \
    --config.batch_size=4 \
    --config.num_steps=25000 \
    --config.log_interval=100 \
    --config.eval_interval=2500 \
    --config.save_interval=2500

echo "Finetuning complete!"