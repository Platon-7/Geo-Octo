# Memory Optimization Fixes for LIBERO-VGGT Octo Finetuning OOM Error

## Problem Analysis

You successfully passed the JAX dtype error (great!), but now encountered an Out of Memory (OOM) error:
```
slurmstepd: error: Detected 1 oom_kill event in StepId=13215045.0. Some of the step tasks have been OOM Killed.
```

This is expected when finetuning large models with high-dimensional VGGT tokens. The VGGT tokens add significant memory overhead (261 tokens × 2048 dimensions per timestep).

## Memory Optimizations Applied

### 1. Reduced Batch Size and Buffers
```python
# In config_offline.py
batch_size=1,  # Reduced from 2 to 1
shuffle_buffer_size=5,  # Reduced from 10 to 5
```

### 2. Added Gradient Accumulation
```python
# Simulate larger batch size without memory cost
grad_accumulation_steps=4,  # Accumulate over 4 steps = effective batch_size of 4
```

### 3. Reduced Window Size
```python
window_size = FieldReference(default=1)  # Reduced from 2 to 1
```

### 4. Optimized Evaluation Settings
```python
val_kwargs=dict(
    val_shuffle_buffer_size=100,  # Reduced from 1000
    num_val_batches=4,  # Reduced from 16
),
viz_kwargs=dict(
    eval_batch_size=16,  # Reduced from 128
    trajs_for_metrics=20,  # Reduced from 100
    trajs_for_viz=2,  # Reduced from 8
    samples_per_state=2,  # Reduced from 8
),
```

### 5. Aggressive VGGT Token Compression
```python
# In VGGTTokenizer
compression_ratio: float = 0.25  # Compress to 25% of original size
num_output_tokens: int = 64  # Reduce from 261 to 64 tokens
```

### 6. Memory-Efficient Data Types
```python
# Use float16 for VGGT tokens to halve memory usage
tokens = jnp.asarray(vggt_tokens, dtype=jnp.float16)
```

### 7. Memory Cleanup in Training Loop
```python
# Clear batch from memory after training step
del batch

# Periodic garbage collection
if (i + 1) % 10 == 0:
    import gc
    gc.collect()
```

## Expected Memory Reduction

These optimizations should reduce memory usage by approximately:
- **Batch size reduction**: 50% reduction
- **VGGT compression**: 75% reduction in VGGT memory
- **Token reduction**: 75% reduction in token count (261 → 64)
- **Float16**: 50% reduction in VGGT token memory
- **Window size**: 50% reduction in sequence memory

**Total estimated memory reduction: ~80-85%**

## Alternative Solutions if OOM Persists

### Option 1: Further Reduce VGGT Tokens
```python
# Even more aggressive compression
num_output_tokens: int = 32  # Reduce to 32 tokens
compression_ratio: float = 0.125  # Compress to 12.5%
```

### Option 2: Freeze More Layers
```python
# In config_offline.py, use head_only mode
mode = "head_only"  # Only train the head, freeze transformer
```

### Option 3: Use Model Sharding
```python
# Add to finetune.py before model creation
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Create sharding for model parameters
devices = mesh_utils.create_device_mesh((1, len(jax.devices())))
mesh = Mesh(devices, axis_names=('data', 'model'))
```

### Option 4: Disable Validation During Training
```python
# Temporarily disable validation to save memory
eval_interval=999999,  # Effectively disable evaluation
```

## Running the Optimized Code

```bash
# Run with the memory-optimized settings
python octo/scripts/finetune.py \
    --config.pretrained_path="hf://rail-berkeley/octo-small-1.5" \
    --config.pretrained_step=300000 \
    --config.save_dir="./octo/my_octo_vggt_model_offline" \
    --config.wandb.group="octo_vggt_finetune" \
    --config.wandb.entity="your_entity"
```

## Monitoring Memory Usage

To monitor memory usage during training:
```bash
# Check GPU memory (if using GPU)
nvidia-smi -l 1

# Check system memory
watch -n 1 'free -h'

# Check process memory
ps aux | grep python
```

## Expected Behavior

With these optimizations, you should see:
1. Successful initialization without OOM
2. Training loop starting
3. Lower memory usage
4. Slower training (due to smaller batch size) but should complete
5. Gradient accumulation maintaining training quality

## If Still OOM

If you still get OOM errors, try in this order:
1. Reduce `num_output_tokens` to 32 or 16
2. Set `compression_ratio` to 0.125
3. Use `mode="head_only"` in config
4. Disable evaluation entirely during training
5. Consider using a machine with more memory or splitting the datasets

The key is finding the right balance between memory usage and model performance. Start with these conservative settings and gradually increase if you have memory headroom.