import datetime
from functools import partial
import os
import psutil
import GPUtil
import gc

from absl import app, flags, logging
from ml_collections import config_flags, ConfigDict
import flax
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from fnmatch import fnmatch
import numpy as np
import tqdm
import wandb

from octo.data.dataset import make_interleaved_dataset
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import SaveCallback, ValidationCallback, VisualizationCallback
from octo.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    Timer,
    TrainState,
    process_text,
)
import sys

from optimize_memory import (
    optimize_tensorflow_memory,
    set_memory_env_variables,
    optimize_dataset_config,
    create_memory_efficient_dataset_kwargs,
    monitor_memory_during_training,
    force_cleanup
)

FLAGS = flags.FLAGS
flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
default_config_file = os.path.join(os.path.dirname(__file__), "configs/debug_rollout_config.py")
config_flags.DEFINE_config_file("config", default_config_file, "File path to the training hyperparameter configuration.", lock_config=False)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'


def log_memory_usage(step, prefix=""):
    # Force garbage collection
    gc.collect()
    
    # Get system memory info
    memory = psutil.virtual_memory()
    
    # Try to get SLURM memory allocation
    slurm_mem_mb = os.environ.get('SLURM_MEM_PER_NODE')
    # if slurm_mem_mb:
    #     try:
    #         allocated_gb = int(slurm_mem_mb) / 1024  # Convert MB to GB
    #         used_gb = memory.used / (1024**3)
    #         usage_percent = (used_gb / allocated_gb) * 100
    #         free_gb = allocated_gb - used_gb
            
    #         print(f"{prefix}Step {step}:")
    #         print(f"  Allocated RAM: {usage_percent:.1f}% ({used_gb:.1f}GB used, {free_gb:.1f}GB free of {allocated_gb:.0f}GB allocated)")
    #     except (ValueError, TypeError):
    #         # Fallback to system memory if SLURM parsing fails
    #         print(f"{prefix}Step {step}:")
    #         print(f"  System RAM: {memory.percent}% ({memory.used/1024**3:.1f}GB used, {memory.available/1024**3:.1f}GB free)")
    # else:
    #     # Fallback to system memory if no SLURM environment
    #     print(f"{prefix}Step {step}:")
    #     print(f"  System RAM: {memory.percent}% ({memory.used/1024**3:.1f}GB used, {memory.available/1024**3:.1f}GB free)")
    
    # print(f"  Python objects: {len(gc.get_objects())}")
    
    # TensorFlow GPU memory (with error handling)
    # try:
    #     if tf.config.list_physical_devices('GPU'):
    #         tf_memory = tf.config.experimental.get_memory_info('GPU:0')
    #         print(f"  TF GPU memory: {tf_memory}")
    # except Exception as e:
    #     # Silently skip GPU memory info if not available
    #     pass

def main(_):
    set_memory_env_variables()
    optimize_tensorflow_memory()
    force_cleanup()
    
    initialize_compilation_cache()
    mesh = Mesh(jax.devices(), axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    logging.info(f"JAX devices: {jax.devices()}")
    name = format_name_with_config(FLAGS.name, FLAGS.config.to_dict())
    wandb_id = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(config=FLAGS.config.to_dict(), id=wandb_id, name=name, mode="disabled" if FLAGS.debug else None, **FLAGS.config.wandb)

    pretrained_model = OctoModel.load_pretrained(FLAGS.config.pretrained_path, step=FLAGS.config.pretrained_step)
    flat_config = flax.traverse_util.flatten_dict(pretrained_model.config, keep_empty_nodes=True)
    for d_key in flax.traverse_util.flatten_dict(FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]
    model_config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    model_config.update(FLAGS.config.get("update_config", ConfigDict()))
    model_config_dict = model_config.to_dict()
    config = FLAGS.config
    
    # Apply memory optimization to config
    config = optimize_dataset_config(config)
    
    # Optimize dataset kwargs for memory efficiency
    optimized_dataset_kwargs = create_memory_efficient_dataset_kwargs(
        config.dataset_kwargs_list, max_datasets=4
    )
    config.dataset_kwargs_list = optimized_dataset_kwargs
    text_processor = ModuleSpec.instantiate(model_config_dict["text_processor"])() if model_config_dict.get("text_processor") else None

    def encode_texts(strings_tensor: tf.Tensor) -> np.ndarray:
        strings_np = strings_tensor.numpy()
        decoded_strings = [s.decode('utf-8') for s in strings_np]
        return text_processor.encode(decoded_strings)["input_ids"].astype(np.int32)

    def process_batch_tf(batch):
        if text_processor is not None and "language_instruction" in batch.get("task", {}):
            lang_instructions = batch["task"]["language_instruction"]
            tokenized = tf.py_function(func=encode_texts, inp=[lang_instructions], Tout=tf.int32)
            tokenized.set_shape([lang_instructions.shape[0], 16])
            batch["task"]["language_instruction"] = tokenized
        if "task" not in batch:
            batch["task"] = {}
        if "image_primary" not in batch["task"] and "image_primary" in batch["observation"]:
            batch["task"]["image_primary"] = batch["observation"]["image_primary"][:, 0]
        return batch

    def prune_batch_for_jax(batch):
        return {"observation": batch["observation"], "task": batch["task"],
                "action": batch["action"], "action_pad_mask": batch["action_pad_mask"]}

    logging.info("Creating training dataset...")
    gc.collect()
    train_dataset_with_stats = make_interleaved_dataset(
        dataset_kwargs_list=config.dataset_kwargs_list, traj_transform_kwargs=config.traj_transform_kwargs,
        frame_transform_kwargs=config.frame_transform_kwargs, train=True, batch_size=config.batch_size,
        shuffle_buffer_size=config.shuffle_buffer_size,
    )
    dataset_statistics = train_dataset_with_stats.dataset_statistics
    
    # Add this memory check:
    current_memory = psutil.virtual_memory().used / (1024**3)
    print(f"Memory after dataset creation: {current_memory:.1f}GB")
    if current_memory > 200:
        print("WARNING: High memory usage detected. Consider reducing batch_size or buffer sizes.")
    
    gc.collect()
    log_memory_usage(0, "AFTER training dataset creation: ")
    train_dataset_processed = train_dataset_with_stats.map(process_batch_tf, num_parallel_calls=4).prefetch(2)
    train_data_iter = train_dataset_processed.iterator()
    

    logging.info("Loading first batch for model initialization...")
    example_batch = prune_batch_for_jax(next(train_data_iter))
    logging.info("Successfully loaded example batch.")

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    model = OctoModel.from_config(model_config_dict, example_batch, text_processor, rng=init_rng, dataset_statistics=dataset_statistics)
    
    print(f"JAX devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")
    print(f"Local device count: {jax.local_device_count()}")

    # Check model sharding
    print("Model parameter shapes and sharding:")
    for name, param in model.params.items():
        if hasattr(param, 'shape'):
            print(f"  {name}: {param.shape}")
        else:
            print(f"  {name}: nested dict")

    # Check GPU memory before first step
    import subprocess
    print("\nGPU Memory before first training step:")
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                        capture_output=True, text=True)
    print(result.stdout)
    
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    tx, lr_callable, grad_norm_callable = create_optimizer(model.params, **config.optimizer.to_dict())
    train_state = TrainState.create(model=model, tx=tx, rng=rng)
    
    # Staged Training Setup
    stage1_steps = config.get("stage1_steps", 0) # Default to 0 if not specified
    stage2_optimizer_config = config.get("stage2_optimizer", None)
    current_stage = 1 if stage1_steps > 0 else 2
    
    # Prepare Stage 2 Optimizer
    if stage2_optimizer_config is not None:
        tx_stage2, lr_callable_stage2, _ = create_optimizer(model.params, **stage2_optimizer_config.to_dict())
    else:
        tx_stage2, lr_callable_stage2 = None, None
        
    logging.info(f"STAGED TRAINING SETUP:")
    if stage1_steps > 0:
        logging.info(f"  Stage 1 (0-{stage1_steps}): Frozen transformer, train heads + adapters only")
        logging.info(f"  Stage 2 (0-{stage1_steps}-{config.num_steps}): Unfreeze everything, gentle finetuning")
        logging.info(f"  Currently in Stage {current_stage}")
    else:
        logging.info(f"  Single-stage training (no staging)")
    
    save_dir = tf.io.gfile.join(config.save_dir, config.wandb.project, config.wandb.group or "", wandb_id) if config.save_dir else None
    if save_dir:
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
    save_callback = SaveCallback(save_dir)

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(batch["observation"], batch["task"], batch["observation"]["timestep_pad_mask"], train=train)
        action_loss, action_metrics = bound_module.heads["action"].loss(transformer_embeddings, batch["action"], batch["observation"]["timestep_pad_mask"], batch["action_pad_mask"], train=train)
        return action_loss, action_metrics

    def analyze_gradients(grads, step, stage, frozen_keys: list | tuple | None):
        """Analyze gradient statistics to understand training instability."""
        if frozen_keys is None:
            frozen_keys = []

        # Manually zero-out gradients for frozen layers before analysis
        param_partitions = flax.traverse_util.path_aware_map(
            lambda path, v: "frozen"
            if any([fnmatch(".".join(path), key) for key in frozen_keys])
            else "trainable",
            grads,
        )
        processed_grads = jax.tree_map(
            lambda g, p: jnp.zeros_like(g) if p == 'frozen' else g,
            grads,
            param_partitions
        )

        def get_grad_stats(grad_tree, prefix=""):
            # This inner function remains the same
            stats = {}
            if isinstance(grad_tree, dict):
                for key, value in grad_tree.items():
                    if isinstance(value, dict):
                        stats.update(get_grad_stats(value, f"{prefix}.{key}" if prefix else key))
                    else:
                        # This is an actual gradient array
                        grad_norm = jnp.linalg.norm(value)
                        grad_max = jnp.max(jnp.abs(value))
                        grad_mean = jnp.mean(jnp.abs(value))
                        stats[f"{prefix}.{key}" if prefix else key] = {
                            "norm": float(grad_norm),
                            "max": float(grad_max),
                            "mean": float(grad_mean)
                        }
            return stats

        # IMPORTANT: Use the processed_grads for analysis
        grad_stats = get_grad_stats(processed_grads)

        # The rest of the function remains the same...
        print(f"\n=== GRADIENT ANALYSIS Step {step} (Stage {stage}) ===")
        adapter_grads = {k: v for k, v in grad_stats.items() if "norm_adapter" in k}
        transformer_grads = {k: v for k, v in grad_stats.items() if "octo_transformer" in k and "norm_adapter" not in k}
        head_grads = {k: v for k, v in grad_stats.items() if "heads" in k}

        print(f"📊 ADAPTER gradients ({len(adapter_grads)} components):")
        for name, stats in list(adapter_grads.items())[:5]:
            print(f"  {name}: norm={stats['norm']:.2e}, max={stats['max']:.2e}")

        print(f"🧠 TRANSFORMER gradients ({len(transformer_grads)} components):")
        for name, stats in list(transformer_grads.items())[:5]:
            print(f"  {name}: norm={stats['norm']:.2e}, max={stats['max']:.2e}")

        print(f"🎯 HEAD gradients ({len(head_grads)} components):")
        for name, stats in list(head_grads.items())[:3]:
            print(f"  {name}: norm={stats['norm']:.2e}, max={stats['max']:.2e}")

        all_norms = [stats['norm'] for stats in grad_stats.values() if stats['norm'] > 0]
        all_maxes = [stats['max'] for stats in grad_stats.values() if stats['max'] > 0]
        min_norm = min(all_norms) if all_norms else 0.0

        print(f"🔥 OVERALL: max_norm={max(all_norms):.2e}, max_value={max(all_maxes):.2e}")
        print(f"📈 RANGE: norm_range=[{min_norm:.2e}, {max(all_norms):.2e}]")

        if max(all_norms) > 1e3:
            print("⚠️  WARNING: Very large gradient norms detected!")
        if max(all_maxes) > 1e6:
            print("🚨 CRITICAL: Extremely large gradient values detected!")

        return grad_stats

    @partial(jax.jit, in_shardings=[replicated_sharding, dp_sharding])
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, dropout_rng, train=True)
        grad_norm = grad_norm_callable(grads) 
        # grad_norm = optax.global_norm(grads)
        new_state = state.apply_gradients(grads=grads, rng=rng)
        info.update({"grad_norm": grad_norm, "learning_rate": lr_callable(state.step)})
        return new_state, info, grads  # Return grads for analysis

    @partial(jax.jit, in_shardings=[replicated_sharding, dp_sharding, replicated_sharding])
    def eval_step(params, batch, rng):
        loss, metrics = loss_fn(params, batch, rng, train=False)
        return metrics

    logging.info("Pre-compiling JAX training and evaluation functions...")
    train_step.lower(train_state, example_batch).compile()
    eval_step.lower(train_state.model.params, example_batch, rng).compile()
    logging.info("JAX functions compiled.")

    val_callback, viz_callback = None, None
    timer = Timer()
    val_data_iter = None
    
    for i in tqdm.tqdm(range(int(config["num_steps"])), total=int(config["num_steps"]), dynamic_ncols=True):
        
        # Staged Training Transition Check
        if stage1_steps > 0 and i == stage1_steps and current_stage == 1:
            logging.info(f"\n🔄 STAGE TRANSITION at Step {i}")
            logging.info(f"  Switching from Stage 1 -> Stage 2")
            logging.info(f"  Unfreezing entire model and switching to gentle fine-tuning LR")
            
            # Switch to stage2 optimizer
            if tx_stage2 is not None:
                # CRITICAL: Create a completely new TrainState with the stage 2 optimizer
                # This reinitializes the optimizer state to match the new optimizer
                train_state = TrainState.create(
                    model=train_state.model,  # Keep the trained model
                    tx=tx_stage2,            # New optimizer
                    rng=train_state.rng      # Keep the RNG state
                )
                
                # IMPORTANT: Recompile the training function for the new optimizer
                logging.info("Recompiling training function for Stage 2...")
                train_step.lower(train_state, example_batch).compile()
                
                # Update the learning rate callable
                lr_callable = lr_callable_stage2
                current_stage = 2
                
                logging.info(f"✅ Successfully switched to Stage 2 optimizer with reinitialized state")
                
                # Log to wandb
                wandb.log({"stage_transition": 2, "unfrozen_training_start": 1}, step=i)
            else:
                logging.warning("⚠️ Stage 2 optimizer not configured, continuing with Stage 1 settings")
            
        
        timer.tick("total")
        with timer("dataset"):
            batch = next(train_data_iter)
        
        with timer("train"):
            model_batch = prune_batch_for_jax(batch)
            train_state, update_info, grads = train_step(train_state, model_batch)
            
            if (i < 10 or
                (stage1_steps > 0 and abs(i - stage1_steps) < 5) or
                i % 1000 == 0):

                # Determine which keys should be frozen for the analysis
                frozen_keys_for_logging = None
                if current_stage == 1:
                    # Access the keys from the optimizer sub-config
                    frozen_keys_for_logging = config.optimizer.frozen_keys

                analyze_gradients(grads, i, current_stage, frozen_keys_for_logging)
                
            timer.tock("total")

        if (i + 1) % config["log_interval"] == 0:
            avg_times = timer.get_average_times()

            # Step 2: Use that same variable for the wandb log.
            update_info_with_stage = jax.device_get(update_info).copy()
            update_info_with_stage["training_stage"] = current_stage
            if stage1_steps > 0:
                update_info_with_stage["stage_progress"] = (i + 1) / stage1_steps if current_stage == 1 else (i + 1 - stage1_steps) / (config.num_steps - stage1_steps)
                
            wandb.log({"training": update_info_with_stage, "timer": avg_times}, step=i) 

            # Step 3: Use the SAME variable for the console log.
            total_time = avg_times.get('total', 1e-6) # Use 1e-6 to avoid division by zero
            dataset_time = avg_times.get('dataset', 0)
            train_time = avg_times.get('train', 0)

            dataset_percent = (dataset_time / total_time) * 100
            train_percent = (train_time / total_time) * 100

            # logging.info(
            #     f"\n| Step {i+1} | "
            #     f"Avg Step Time: {total_time:.4f}s | "
            #     f"Data Loading: {dataset_time:.4f}s ({dataset_percent:.1f}%) | "
            #     f"Model Training: {train_time:.4f}s ({train_percent:.1f}%) |\n"
            # )

        if i % 50 == 0:  # Every 1000 steps
            
            # Monitor memory and trigger cleanup if needed
            if not monitor_memory_during_training(i, threshold_gb=300.0):
                print("Memory usage too high, consider stopping training")
            
            # CPU/RAM usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            #print(f"Step {i}: CPU {cpu_percent}%, RAM {memory.percent}% ({memory.available/1024**3:.1f}GB free)")
            
            # # GPU usage
            # try:
            #     gpus = GPUtil.getGPUs()
            #     for gpu in gpus:
            #         print(f"  GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% memory, {gpu.load*100:.1f}% util")
            # except:
            #     pass
    
        if i % 50 == 0:
            force_cleanup()
            
        if (i + 1) % config["eval_interval"] == 0:
            logging.info("Evaluating...")
            # log_memory_usage(i, "BEFORE validation: ")
            
            # Lazily initialize the validation data iterator to save memory at startup
            if val_data_iter is None:
                logging.info("Initializing validation dataset...")
                val_dataset = make_interleaved_dataset(
                    dataset_kwargs_list=config.dataset_kwargs_list,
                    traj_transform_kwargs=config.traj_transform_kwargs,
                    frame_transform_kwargs=config.frame_transform_kwargs, 
                    train=False,
                    batch_size=config["viz_kwargs"]["eval_batch_size"],
                    shuffle_buffer_size=config["val_kwargs"]["val_shuffle_buffer_size"],
                ).map(process_batch_tf, num_parallel_calls=4).prefetch(2).repeat()
                val_data_iter = val_dataset.iterator()

            # Manually run the evaluation loop for full control
            metrics = []
            for _ in range(config["val_kwargs"]["num_val_batches"]):
                # Get a raw batch from the validation pipeline
                val_batch = next(val_data_iter)
                # Explicitly prune it to be JAX-safe
                model_val_batch = prune_batch_for_jax(val_batch)
                # Split a new RNG key for this step
                eval_rng, rng = jax.random.split(rng)
                # Call our pre-compiled, known-safe eval_step function
                metric_update = eval_step(train_state.model.params, model_val_batch, eval_rng)
                metrics.append(metric_update)
            
            # Aggregate and log the metrics
            metrics = jax.tree_map(lambda *xs: np.mean([x for x in xs]), *metrics)
            wandb.log({"validation": metrics}, step=i)
            
            # log_memory_usage(i, "AFTER validation: ")
            gc.collect()

        if (i + 1) % config["save_interval"] == 0 and save_dir:
            save_callback(train_state, i + 1)

if __name__ == "__main__":
    app.run(main)