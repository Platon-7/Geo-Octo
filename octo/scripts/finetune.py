import datetime
from functools import partial
import os

from absl import app, flags, logging
from ml_collections import config_flags, ConfigDict
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tqdm
import wandb
import gc
import psutil

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

FLAGS = flags.FLAGS
flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
default_config_file = os.path.join(os.path.dirname(__file__), "configs/debug_rollout_config.py")
config_flags.DEFINE_config_file("config", default_config_file, "File path to the training hyperparameter configuration.", lock_config=False)


import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

def log_detailed_memory(step, label, prefix=""):
    """Enhanced memory logging with more details"""
    import psutil
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Get process info
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    # System memory
    system_memory = psutil.virtual_memory()
    
    # Try to get SLURM allocation
    slurm_mem_mb = os.environ.get('SLURM_MEM_PER_NODE')
    if slurm_mem_mb:
        try:
            allocated_gb = int(slurm_mem_mb) / 1024
            usage_percent = (memory_info.rss / (1024**3) / allocated_gb) * 100
            free_gb = allocated_gb - (memory_info.rss / (1024**3))
            
            print(f"{prefix}🔍 MEMORY ANALYSIS - {label} (Step {step}):")
            print(f"  Process RSS: {memory_info.rss / (1024**3):.1f}GB ({usage_percent:.1f}% of {allocated_gb:.0f}GB allocated)")
            print(f"  Process VMS: {memory_info.vms / (1024**3):.1f}GB")
            print(f"  System RAM: {system_memory.percent:.1f}% ({system_memory.available / (1024**3):.1f}GB free)")
            print(f"  Python objects: {len(gc.get_objects())}")
            
            # Try to get TensorFlow memory info safely
            try:
                import tensorflow as tf
                if tf.config.list_physical_devices('GPU'):
                    tf_memory = tf.config.experimental.get_memory_info('GPU:0')
                    print(f"  TF GPU memory: {tf_memory}")
            except:
                pass
                
        except (ValueError, TypeError):
            print(f"{prefix}🔍 MEMORY ANALYSIS - {label} (Step {step}):")
            print(f"  Process RSS: {memory_info.rss / (1024**3):.1f}GB")
            print(f"  System RAM: {system_memory.percent:.1f}% ({system_memory.available / (1024**3):.1f}GB free)")

def log_memory_usage(step, prefix=""):
    # Force garbage collection
    gc.collect()
    
    # Get system memory info
    memory = psutil.virtual_memory()
    
    # Try to get SLURM memory allocation
    slurm_mem_mb = os.environ.get('SLURM_MEM_PER_NODE')
    if slurm_mem_mb:
        try:
            allocated_gb = int(slurm_mem_mb) / 1024  # Convert MB to GB
            used_gb = memory.used / (1024**3)
            usage_percent = (used_gb / allocated_gb) * 100
            free_gb = allocated_gb - used_gb
            
            print(f"{prefix}Step {step}:")
            print(f"  Allocated RAM: {usage_percent:.1f}% ({used_gb:.1f}GB used, {free_gb:.1f}GB free of {allocated_gb:.0f}GB allocated)")
        except (ValueError, TypeError):
            # Fallback to system memory if SLURM parsing fails
            print(f"{prefix}Step {step}:")
            print(f"  System RAM: {memory.percent}% ({memory.used/1024**3:.1f}GB used, {memory.available/1024**3:.1f}GB free)")
    else:
        # Fallback to system memory if no SLURM environment
        print(f"{prefix}Step {step}:")
        print(f"  System RAM: {memory.percent}% ({memory.used/1024**3:.1f}GB used, {memory.available/1024**3:.1f}GB free)")
    
    print(f"  Python objects: {len(gc.get_objects())}")
    
    # TensorFlow GPU memory (with error handling)
    try:
        if tf.config.list_physical_devices('GPU'):
            tf_memory = tf.config.experimental.get_memory_info('GPU:0')
            print(f"  TF GPU memory: {tf_memory}")
    except Exception as e:
        # Silently skip GPU memory info if not available
        pass

def main(_):
    # MEMORY CHECKPOINT 0: Script startup baseline
    log_detailed_memory(-1, "Script startup baseline", "")
    
    initialize_compilation_cache()
    mesh = Mesh(jax.devices(), axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    tf.config.set_visible_devices([], "GPU")

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
    text_processor = ModuleSpec.instantiate(model_config_dict["text_processor"])() if model_config_dict.get("text_processor") else None

    # MEMORY CHECKPOINT 1: After model loading
    log_detailed_memory(-1, "After model loading", "")

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

    # MEMORY CHECKPOINT 2: Before training dataset creation
    log_detailed_memory(-1, "Before training dataset creation", "")
    
    logging.info("Creating training dataset...")
    train_dataset_with_stats = make_interleaved_dataset(
        dataset_kwargs_list=config.dataset_kwargs_list, traj_transform_kwargs=config.traj_transform_kwargs,
        frame_transform_kwargs=config.frame_transform_kwargs, train=True, batch_size=config.batch_size,
        shuffle_buffer_size=config.shuffle_buffer_size,
    )
    
    # MEMORY CHECKPOINT 3: After training dataset creation
    log_detailed_memory(-1, "After training dataset creation", "")
    
    dataset_statistics = train_dataset_with_stats.dataset_statistics
    train_dataset_processed = train_dataset_with_stats.map(process_batch_tf, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    train_data_iter = train_dataset_processed.iterator()

    # MEMORY CHECKPOINT 4: After dataset processing and iterator creation
    log_detailed_memory(-1, "After dataset processing and iterator", "")

    logging.info("Loading first batch for model initialization...")
    example_batch = prune_batch_for_jax(next(train_data_iter))
    logging.info("Successfully loaded example batch.")

    # MEMORY CHECKPOINT 5: After loading first batch
    log_detailed_memory(-1, "After loading first batch", "")

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    model = OctoModel.from_config(model_config_dict, example_batch, text_processor, rng=init_rng, dataset_statistics=dataset_statistics)
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    # MEMORY CHECKPOINT 6: After model initialization and parameter merging
    log_detailed_memory(-1, "After model initialization", "")

    tx, lr_callable, _ = create_optimizer(model.params, **config.optimizer.to_dict())
    train_state = TrainState.create(model=model, tx=tx, rng=rng)
    save_dir = tf.io.gfile.join(config.save_dir, config.wandb.project, config.wandb.group or "", wandb_id) if config.save_dir else None
    if save_dir:
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
    save_callback = SaveCallback(save_dir)

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(batch["observation"], batch["task"], batch["observation"]["timestep_pad_mask"], train=train)
        action_loss, action_metrics = bound_module.heads["action"].loss(transformer_embeddings, batch["action"], batch["observation"]["timestep_pad_mask"], batch["action_pad_mask"], train=train)
        return action_loss, action_metrics

    @partial(jax.jit, in_shardings=[replicated_sharding, dp_sharding])
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, dropout_rng, train=True)
        grad_norm = optax.global_norm(grads)
        new_state = state.apply_gradients(grads=grads, rng=rng)
        info.update({"grad_norm": grad_norm, "learning_rate": lr_callable(state.step)})
        return new_state, info

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
    val_dataset = None
    
    for i in tqdm.tqdm(range(int(config["num_steps"])), total=int(config["num_steps"]), dynamic_ncols=True):
        timer.tick("total")
        with timer("dataset"):
            batch = next(train_data_iter)
        
        with timer("train"):
            model_batch = prune_batch_for_jax(batch)
            train_state, update_info = train_step(train_state, model_batch)
        timer.tock("total")

        if (i + 1) % config["log_interval"] == 0:
            wandb.log({"training": jax.device_get(update_info), "timer": timer.get_average_times()}, step=i)

        if (i + 1) % config["eval_interval"] == 0:
            logging.info("Evaluating...")
            log_memory_usage(i, "BEFORE validation: ")
            
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
                ).map(process_batch_tf, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).repeat()
                val_data_iter = val_dataset.iterator()

            # Manually run the evaluation loop for full control
            metrics = []
            for _ in range(config["val_kwargs"]["num_val_batches"]):
                try:
                    # Get a raw batch from the validation pipeline
                    val_batch = next(val_data_iter)
                except StopIteration:
                    # Iterator exhausted - recreate it (shouldn't happen with .repeat())
                    logging.warning("Validation dataset iterator exhausted, recreating...")
                    val_data_iter = val_dataset.iterator()
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
            
            log_memory_usage(i, "AFTER validation: ")
    
            # Force garbage collection but keep validation iterator alive for reuse
            gc.collect()

if __name__ == "__main__":
    app.run(main)