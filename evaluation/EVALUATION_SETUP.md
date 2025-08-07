# Octo + LIBERO Evaluation Setup Guide

## Overview

I have fixed the `evaluation/scripts/evaluate.py` script to properly evaluate your fine-tuned Octo model on LIBERO tasks. The script now correctly:

1. Loads Octo models using the proper API
2. Formats observations for the model
3. Handles language instructions and task specifications
4. Manages action normalization/denormalization
5. Runs evaluation loops and saves videos

## Key Fixes Made

### 1. Model Loading
**Before:** Used deprecated `OctoModel.load_from_checkpoint()`
**After:** Uses correct `OctoModel.load_pretrained()` method

### 2. Observation Format
**Before:** Incorrect observation format with wrong dimensions
**After:** Proper format with batch and window dimensions:
```python
model_observation = {
    "image_primary": image[None, None, ...],  # (batch, window, H, W, C)
    "timestep_pad_mask": np.array([[True]], dtype=bool)  # No padding
}
```

### 3. Task Specification
**Before:** Manual TensorFlow constants for language
**After:** Uses model's built-in task creation:
```python
task_spec = model.create_tasks(texts=[language_instruction])
```

### 4. Action Sampling
**Before:** Used deprecated `.sample()` method
**After:** Uses correct `.sample_actions()` with proper policy function:
```python
policy_fn = supply_rng(
    partial(
        model.sample_actions,
        unnormalization_statistics=action_stats,
        train=False,
    ),
)
```

### 5. Action Unnormalization
**Before:** Missing action statistics
**After:** Automatically detects and uses available dataset statistics

## Configuration

The script is configured for your specific setup:

- **Model Path:** `/home/pkarageorgis/geo_octo/octo/my_octo_vggt_model_offline/octo_vggt_finetune_staged/experiment_20250805_112710_BEST_RUN/150000/default/checkpoint`
- **Data Path:** `/scratch-shared/tmp.cwkV8vOvfY/libero_evaluation`
- **Task Suite:** `libero_10` (as requested)
- **Episodes:** 200 timesteps max per episode
- **Output:** Videos saved to `evaluation/test_outputs/`

## Usage Instructions

### 1. Activate Environment
First, activate your conda environment:
```bash
conda activate octo-eval  # or whatever you named your environment
```

### 2. Run Evaluation
```bash
cd /workspace
python evaluation/scripts/evaluate.py
```

### 3. Customize Evaluation
You can modify these variables in the script:
- `TASK_SUITE_NAME`: Change to "libero_goal", "libero_spatial", etc.
- `EVAL_TASK_ID`: Set to specific task ID or leave None for random
- `NUM_TIMESTEPS`: Adjust episode length
- `MODEL_PATH` and `DATASET_DIR`: Update paths if needed

## Expected Output

The script will:
1. Load your fine-tuned Octo model
2. Initialize a LIBERO environment for the selected task
3. Run the model for up to 200 steps
4. Save a video of the evaluation to `evaluation/test_outputs/`
5. Print success/failure status and task completion

## Troubleshooting

### Environment Issues
If you get import errors, ensure:
- The conda environment is activated
- All dependencies from `requirements_octo_libero.txt` are installed
- The environment includes both Octo and LIBERO packages

### Path Issues
If you get file not found errors:
- Verify the model checkpoint path exists
- Verify the LIBERO dataset path exists
- Ensure you're running from the `/workspace` directory

### Model Issues
If you get model loading errors:
- Verify the checkpoint is in the correct Octo format
- Check that all required files exist in the checkpoint directory:
  - `config.json`
  - `example_batch.msgpack`
  - `dataset_statistics.json`
  - Model parameter files

## Testing Environment Only

If you want to test the LIBERO environment setup without the full Octo model, use:
```bash
python evaluation/scripts/test_env_only.py
```

This will verify that LIBERO is working correctly and show what's needed for full Octo integration.

## Example Success Output

```
==================================================
OCTO MODEL EVALUATION SCRIPT
==================================================

[INFO] Loading Octo model from: /path/to/checkpoint
[SUCCESS] Octo model loaded.

[INFO] Accessing benchmark suite: libero_10
[INFO] No task ID specified. Randomly selected task #3
[SUCCESS] Retrieved task 'open_the_top_drawer_and_put_the_bowl_inside'
    - Language instruction: 'open the top drawer and put the bowl inside'

[INFO] Created task specification for: 'open the top drawer and put the bowl inside'
[INFO] Setting up policy function...
[INFO] Available dataset statistics: ['bridge_dataset']
[INFO] Using action statistics from: bridge_dataset
[INFO] Starting evaluation loop...
    - Step 1/200: Reward=0.0, Done=False
    - Step 2/200: Reward=0.0, Done=False
    ...
[SUCCESS] Evaluation loop completed.

[INFO] Saving episode video...
[SUCCESS] Video saved to: evaluation/test_outputs/octo_eval_libero_10_3.mp4
==================================================
EVALUATION SCRIPT FINISHED
==================================================
```

## Next Steps

1. **Single Task Evaluation:** Run the script as-is to test one random task
2. **Full Suite Evaluation:** Modify to loop through all tasks in libero_10
3. **Multiple Suites:** Test on libero_goal, libero_spatial, etc.
4. **Metrics Collection:** Add success rate tracking and other metrics
5. **Comparative Analysis:** Compare against baseline models or other checkpoints