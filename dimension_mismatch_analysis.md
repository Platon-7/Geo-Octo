# Octo Finetuning 444 vs 430 Dimension Mismatch Analysis

## Problem Summary

You're getting a persistent error when finetuning Octo:
```
ScopeParamShapeError: Initializer expected to generate shape (444, 256) but got shape (430, 256) instead for parameter "kernel" in "/heads_action/diffusion_model/reverse_network/Dense_0"
```

## Key Findings

### 1. **Error is NOT caused by VGGT tokens**
- Error persists even when VGGT tokens are completely removed from data
- Error occurs with standard Octo observation format (only `image_primary` and `proprio`)

### 2. **Error is NOT caused by your LIBERO datasets specifically**
- Same error occurs when trying to use standard Bridge dataset
- This indicates the issue is with model configuration or environment, not your data

### 3. **Your successful online finetuning approach**
- Your `finetune_online_inference.py` worked successfully
- Used on-the-fly VGGT inference rather than model architecture changes
- Used standard Octo model without VGGTTokenizer

### 4. **Current data format (after cleanup)**
```
Batch observation keys: ['image_primary', 'proprio', 'timestep', 'pad_mask_dict', 'timestep_pad_mask', 'task_completed']
  image_primary: shape = (2, 2, 256, 256, 3), dtype = uint8
  proprio: shape = (2, 2, 8), dtype = float32
```

### 5. **GPU/CUDA issues preventing debugging**
- Getting `CUDNN_STATUS_EXECUTION_FAILED` errors
- Unable to examine transformer embeddings due to GPU execution failures

## Potential Root Causes

### 1. **Pretrained Model Version Mismatch**
You're using: `hf://rail-berkeley/octo-small-1.5` at step 300000
- Different model versions might have different embedding dimensions
- The specific checkpoint might have architectural differences

### 2. **JAX/CUDA Environment Issues**
- CUDA execution failures suggest environment problems
- JAX version compatibility with your CUDA/cuDNN setup
- GPU memory or driver issues

### 3. **Image Size Mismatch**
- Your images are 256x256, standard Octo expects 224x224
- This could cause different tokenization output sizes

### 4. **Action Dimension Differences**
- Your action_dim is 7, which matches the error message
- But the embedding dimension mismatch suggests input size issues, not output

## Attempted Solutions

### ✅ **Successfully Tried**
1. Fixed JAX dtype errors (object dtypes, boolean arrays)
2. Removed VGGT tokens from model architecture
3. Cleaned up data to standard Octo format
4. Removed extra observation fields (joint_state, task_completed, etc.)

### ❌ **Unsuccessful**
1. Adding VGGTTokenizer to model architecture
2. Selective parameter merging (excluding action head)
3. CPU-only execution (still getting GPU errors)
4. Standard dataset testing (same error)

## Recommended Next Steps

### 1. **Try Different Pretrained Model**
```bash
# Try the base model instead of the 1.5 version
--config.pretrained_path="hf://rail-berkeley/octo-base"
--config.pretrained_step=200000
```

### 2. **Fix Image Size**
Change your config to use standard Octo image size:
```python
"resize_size": {
    "primary": (224, 224),  # Change from (256, 256)
},
```

### 3. **Environment Debugging**
Check your JAX/CUDA setup:
```bash
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
nvidia-smi
```

### 4. **Use Working Online Approach**
Since your online finetuning worked, consider:
- Adapting the online approach for offline datasets
- Using the exact same model configuration that worked online
- Replicating the successful tokenizer setup from `finetune_online_inference.py`

### 5. **Try Simplified Config**
Start with the most basic Octo finetuning example:
```python
# Minimal config without any customizations
DATASET_KWARGS_LIST = [
    {
        "name": "your_dataset",
        "image_obs_keys": {"primary": "image_primary"},
        "proprio_obs_key": "proprio", 
        "language_key": "language_instruction",
        # Remove ALL other customizations
    }
]
```

### 6. **Debug with Standard Octo Examples**
Try running the official Octo finetuning examples first:
- Use their exact configs and datasets
- Verify your environment works with standard examples
- Then gradually adapt to your setup

## Most Likely Solutions (in order of probability)

1. **Image size mismatch** - Change to 224x224
2. **Pretrained model version** - Try octo-base instead of octo-small-1.5  
3. **Environment issues** - Fix CUDA/JAX setup
4. **Use your working online approach** - Adapt it for offline training

## Code Changes to Try First

### 1. Change image size in config:
```python
"resize_size": {
    "primary": (224, 224),  # Standard Octo size
},
```

### 2. Try different pretrained model:
```bash
--config.pretrained_path="hf://rail-berkeley/octo-base"
```

### 3. Force CPU debugging (stronger approach):
```bash
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"
python -c "import jax; jax.config.update('jax_platform_name', 'cpu')"
```

The 14-dimension difference (444-430=14) is quite specific and suggests a very particular architectural mismatch. This points most strongly to either image size differences or pretrained model version incompatibility.