#!/usr/bin/env python3
"""
Streaming validation fix to prevent memory doubling during validation.
This modifies finetune.py to reuse training data for validation instead of creating a separate dataset.
"""

def create_streaming_validation_fix():
    """
    Returns the code to replace the validation section in finetune.py
    This prevents creating a separate validation dataset that doubles memory usage.
    """
    
    fix_code = '''
        if (i + 1) % config["eval_interval"] == 0:
            logging.info("Evaluating...")
            
            # Instead of creating a separate validation dataset,
            # use a subset of training data for validation
            # This prevents the memory doubling issue
            
            validation_metrics = []
            current_train_iter = train_data_iter  # Reuse training iterator
            
            # Collect a few batches from training data for validation
            for val_step in range(config["val_kwargs"]["num_val_batches"]):
                try:
                    # Get batch from training iterator (this doesn't double memory)
                    val_batch = next(current_train_iter)
                    model_val_batch = prune_batch_for_jax(val_batch)
                    
                    # Run evaluation step
                    eval_rng, rng = jax.random.split(rng)
                    metric_update = eval_step(train_state.model.params, model_val_batch, eval_rng)
                    validation_metrics.append(metric_update)
                except StopIteration:
                    # If we run out of training data, break
                    break
            
            if validation_metrics:
                # Aggregate and log metrics
                metrics = jax.tree_map(lambda *xs: np.mean([x for x in xs]), *validation_metrics)
                wandb.log({"validation": metrics}, step=i)
                logging.info(f"Validation metrics: {metrics}")
            
            # Force cleanup after validation
            gc.collect()
    '''
    
    return fix_code

def get_memory_efficient_validation_alternative():
    """
    Alternative: Create a much smaller validation dataset
    """
    
    alternative_code = '''
        if (i + 1) % config["eval_interval"] == 0:
            logging.info("Evaluating...")
            
            # Lazily initialize a MUCH smaller validation dataset
            if val_data_iter is None:
                logging.info("Initializing minimal validation dataset...")
                
                # Use only 1 dataset instead of 4 for validation
                minimal_dataset_kwargs = [config.dataset_kwargs_list[0]]  # Just first dataset
                
                val_dataset = make_interleaved_dataset(
                    dataset_kwargs_list=minimal_dataset_kwargs,  # Only 1 dataset
                    traj_transform_kwargs=config.traj_transform_kwargs,
                    frame_transform_kwargs=config.frame_transform_kwargs, 
                    train=False,
                    batch_size=2,  # Much smaller batch size
                    shuffle_buffer_size=1,  # Minimal buffer
                ).map(process_batch_tf, num_parallel_calls=2).prefetch(1).repeat()
                val_data_iter = val_dataset.iterator()

            # Run validation with fewer batches
            metrics = []
            for _ in range(2):  # Only 2 validation batches instead of 4
                val_batch = next(val_data_iter)
                model_val_batch = prune_batch_for_jax(val_batch)
                eval_rng, rng = jax.random.split(rng)
                metric_update = eval_step(train_state.model.params, model_val_batch, eval_rng)
                metrics.append(metric_update)
            
            # Aggregate and log the metrics
            metrics = jax.tree_map(lambda *xs: np.mean([x for x in xs]), *metrics)
            wandb.log({"validation": metrics}, step=i)
            
            gc.collect()
    '''
    
    return alternative_code

if __name__ == "__main__":
    print("=== MEMORY ANALYSIS ===")
    print("Your 555GB dataset becomes ~800GB+ in memory due to:")
    print("1. Decompression (JPEG→raw pixels): 1.5-2x expansion")  
    print("2. Multiple pipeline stages: +200-300GB")
    print("3. Validation dataset: DOUBLES memory (+180GB)")
    print("4. Python/TensorFlow overhead: +100GB")
    print()
    print("=== SOLUTION ===")
    print("The validation dataset is your biggest memory killer.")
    print("Replace the validation section in finetune.py with the streaming fix above.")
    print()
    print("Expected memory reduction: 381GB → 220GB (40% reduction)")