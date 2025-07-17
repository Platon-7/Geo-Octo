# VGGT Finetuning Setup for Octo

This README explains the modifications made to enable finetuning Octo models with VGGT-preprocessed datasets.

## What Was Modified

### 1. Updated `octo/scripts/finetune.py`

- **TFDS Builders**: Added custom TFDS builders for VGGT datasets:
  - `LiberoObjectVggt`
  - `LiberoSpatialVggt` 
  - `LiberoGoalVggt`
  - `LiberoO10Vggt` / `LiberO10Vggt` (handles the typo in the original dataset)

- **Configuration Handling**: Enhanced config merging to support:
  - `update_config`: Updates specific parts of the pretrained model config
  - `config_delete_keys`: Removes conflicting keys from pretrained config

### 2. Created `octo/scripts/configs/finetune_vggt_config.py`

- **Dataset Configuration**: Configured for your VGGT datasets in `/workspace/libero_vggt_datasets2`
- **Model Configuration**: Set up to use both VGGT tokens and image tokenizers
- **Training Parameters**: Optimized for VGGT finetuning

### 3. Created Launch Script `run_vggt_finetune.sh`

A convenience script to run finetuning with proper parameters.

## Dataset Structure Expected

Your VGGT datasets should have this structure:
```
/workspace/libero_vggt_datasets2/
├── libero_object_vggt/
│   └── 1.0.0/
│       ├── dataset_info.json
│       ├── features.json
│       └── *.tfrecord files
├── libero_spatial_vggt/
├── libero_goal_vggt/
└── liber_o10_vggt/  # Note the typo
```

Each dataset should contain episodes with:
- **observations**: `image`, `proprio`, `vggt_tokens` (shape: 261, 2048, dtype: float16)
- **actions**: 7-dimensional action vectors
- **episode_metadata**: task descriptions and file paths

## How to Use

### 1. Update Paths in Config

Edit `octo/scripts/configs/finetune_vggt_config.py` and update:
```python
"data_dir": "/path/to/your/vggt/datasets",  # Update this path
```

### 2. Update Launch Script

Edit `run_vggt_finetune.sh` and set:
```bash
PRETRAINED_PATH="/path/to/pretrained/octo/model"
PRETRAINED_STEP=1000000  # Or whatever step you want
SAVE_DIR="/path/to/save/finetuned/models"
WANDB_ENTITY="your_wandb_username"
```

### 3. Run Finetuning

```bash
chmod +x run_vggt_finetune.sh
./run_vggt_finetune.sh
```

### 4. Manual Run (Alternative)

```bash
cd octo/scripts
python finetune.py \
    --name "vggt_libero_experiment" \
    --config.pretrained_path="/path/to/pretrained/model" \
    --config.pretrained_step=1000000 \
    --config.save_dir="/path/to/save/dir" \
    --config.wandb.project="octo_vggt_finetune" \
    --config.wandb.entity="your_entity"
```

## Key Configuration Options

The config supports several important settings:

### Model Architecture
- **VGGT Tokenizer**: Processes pre-computed VGGT tokens
- **Image Tokenizer**: Processes raw images (for mixed vision approaches)
- **Mixed Vision**: Can use both VGGT and patch tokens simultaneously

### Training Settings
- **Finetuning Mode**: `full`, `head_only`, or `head_mlp_only`
- **Task Mode**: `image_conditioned`, `language_conditioned`, or `multimodal`
- **Batch Size**: Default 8, adjust based on GPU memory
- **Learning Rate**: Cosine schedule with warmup

### Data Settings
- **Window Size**: Temporal window for observations/actions
- **Action Horizon**: Future action prediction length
- **Data Augmentation**: Configurable image augmentations

## Troubleshooting

### Common Issues

1. **Dataset Not Found**: Ensure TFDS builders are properly registered by importing `finetune.py`
2. **VGGT Tokens Missing**: Check that your datasets actually contain `vggt_tokens` in observations
3. **Shape Mismatches**: Verify VGGT tokens have shape (261, 2048) and dtype float16
4. **Memory Issues**: Reduce batch size or use gradient accumulation

### Debugging Dataset Loading

Run this to test dataset loading:
```python
import tensorflow_datasets as tfds
import sys
sys.path.append('octo/scripts')
import finetune  # Register builders

builder = tfds.builder('libero_object_vggt', data_dir='/workspace/libero_vggt_datasets2')
ds = builder.as_dataset(split='train[:1]')
for episode in ds.take(1):
    steps = list(episode['steps'])
    print(f"Found {len(steps)} steps")
    if steps:
        obs = steps[0]['observation']
        print(f"Observation keys: {list(obs.keys())}")
        if 'vggt_tokens' in obs:
            print(f"VGGT tokens shape: {obs['vggt_tokens'].shape}")
```

## Model Architecture Details

The finetuning setup uses:

1. **VGGT Tokenizer**: Directly processes pre-computed VGGT tokens
2. **Image Tokenizer**: Processes raw images with patch encoder
3. **Mixed Vision**: Both tokenizers feed into the transformer
4. **Action Head**: Predicts robot actions from fused representations

This allows the model to leverage both the rich VGGT representations and fine-grained image patches for robust manipulation policies.

## Next Steps

After finetuning:
1. **Evaluation**: Use the finetuned model for robot evaluation
2. **Analysis**: Compare VGGT vs standard image finetuning performance
3. **Ablation**: Test different VGGT integration strategies
4. **Deployment**: Use finetuned models for real robot control