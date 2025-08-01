# Octo Memory Optimization Fix

## Problem Summary
The original Octo finetuning was using **~450GB of RAM**, which is excessive and unsustainable. This was caused by:

1. **Aggressive prefetching**: `tf.data.AUTOTUNE` causing TensorFlow to buffer unlimited batches
2. **Large shuffle buffers**: 100 and 50 buffer sizes multiplying memory usage 
3. **Ineffective memory management**: `with_ram_budget(1)` setting too low to be useful
4. **Dataset replication**: Multiple repeated streams loading simultaneously

## Solution Applied

### 1. Configuration Changes (`config_offline.py`)
- **Shuffle buffer sizes**: Reduced from 100→10 (training) and 50→5 (validation)
- These buffers hold batches in memory, so smaller buffers = less memory

### 2. Pipeline Optimizations (`finetune.py`) 
- **Replaced AUTOTUNE**: Fixed `prefetch(2)` and `num_parallel_calls=4` instead of unlimited
- **Added memory monitoring**: Track usage during training with cleanup
- **Force garbage collection**: Regular cleanup to prevent memory leaks

### 3. Dataset Memory Management (`dataset.py`)
- **Increased RAM budget**: Changed from 1GB to 50GB for proper memory management
- **Better memory allocation**: Allows dlimp to manage memory effectively

### 4. System Optimizations (`optimize_memory.py`)
- **TensorFlow memory limits**: Reduced allocation and enabled memory growth
- **Environment variables**: Conservative memory settings
- **Runtime monitoring**: Automatic cleanup when memory gets high

## Expected Results

| Before | After | Improvement |
|--------|-------|-------------|
| ~450GB RAM | ~90-180GB RAM | 60-80% reduction |
| Buffer: 100/50 | Buffer: 10/5 | 10x smaller buffers |
| Unlimited prefetch | Fixed prefetch: 2 | Controlled memory growth |

## How to Use

### Quick Test
```bash
cd /workspace
python test_memory_fix.py
```

### Run Training with Optimizations
The optimizations are automatically applied when you run:
```bash
python octo/scripts/finetune.py --config=octo/scripts/configs/config_offline.py
```

### Manual Control
If you need different settings, you can adjust:
- `shuffle_buffer_size` in config_offline.py (try 5, 10, or 20)
- `prefetch(N)` in finetune.py (try 1, 2, or 4)
- Memory threshold in optimize_memory.py

## Additional Memory Saving Tips

### For Extreme Cases (if still too much memory):
1. **Reduce batch size**: Change `batch_size=8` to `batch_size=4` or `batch_size=2`
2. **Use fewer datasets**: Modify `max_datasets=2` to `max_datasets=1` in the optimization
3. **Lower shuffle buffer**: Set `shuffle_buffer_size=5` or even `shuffle_buffer_size=1`

### Monitor Memory Usage:
```bash
# During training, watch for these messages:
# "WARNING: Memory usage approaching limit" 
# "CRITICAL: Memory usage exceeds threshold"
```

## Technical Details

### Root Cause Analysis
The excessive memory usage was primarily due to TensorFlow's `AUTOTUNE` feature combined with `repeat()` operations on multiple datasets. This created a scenario where:

1. Each dataset was repeated infinitely 
2. AUTOTUNE aggressively prefetched as many batches as possible
3. Shuffle buffers held large numbers of batches in memory
4. Multiple datasets were processed in parallel, multiplying the effect

### Why These Fixes Work
1. **Fixed prefetch limits**: Prevents unlimited buffering
2. **Smaller shuffle buffers**: Reduces memory footprint per dataset  
3. **Proper RAM budget**: Allows dlimp's memory manager to work effectively
4. **Environment controls**: Prevents TensorFlow from over-allocating memory

## Troubleshooting

### If you still see high memory usage:
1. Check that the optimizations are being applied (run `test_memory_fix.py`)
2. Reduce batch size further
3. Use only 1 dataset instead of 4
4. Consider using a machine with more swap space

### If training is slower:
1. Increase prefetch from 2 to 4
2. Increase shuffle buffer sizes slightly (but keep under 20)
3. Increase `num_parallel_calls` from 4 to 8

The key insight is that a small reduction in randomness (smaller shuffle buffers) is worth the massive memory savings.