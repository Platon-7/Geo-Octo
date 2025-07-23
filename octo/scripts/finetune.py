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

# TFDS will automatically discover datasets in the data_dir if they have the right structure
# No explicit builders needed since the datasets were created with proper TFDS format

def main(_):
    initialize_compilation_cache()
    mesh = Mesh(jax.devices(), axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    tf.config.set_visible_devices([], "GPU")


    print(f"JAX devices: {jax.devices()}")
    print(f"Number of devices: {len(jax.devices())}")
    print(f"Device type: {jax.devices()[0].device_kind}")

    name = format_name_with_config(FLAGS.name, FLAGS.config.to_dict())
    wandb_id = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(config=FLAGS.config.to_dict(), id=wandb_id, name=name, mode="disabled" if FLAGS.debug else None, **FLAGS.config.wandb)

    #########
    #
    # Load Pretrained model + optionally modify config (following online implementation pattern)
    #
    #########
    
    #########
    #
    # Load Pretrained model + optionally modify config (following online implementation pattern)
    #
    #########
    
    pretrained_model = OctoModel.load_pretrained(FLAGS.config.pretrained_path, step=FLAGS.config.pretrained_step)
    
    # Follow the same pattern as online implementation
    flat_config = flax.traverse_util.flatten_dict(
        pretrained_model.config, keep_empty_nodes=True
    )
    for d_key in flax.traverse_util.flatten_dict(
        FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    ):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]

    model_config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    model_config.update(FLAGS.config.get("update_config", ConfigDict()))
    model_config_dict = model_config.to_dict()
    check_config_diff(model_config_dict, pretrained_model.config)

    # Keep the original FLAGS.config for dataset and training parameters
    config = FLAGS.config
    
    text_processor = ModuleSpec.instantiate(model_config_dict["text_processor"])() if model_config_dict.get("text_processor") else None
    
    logging.info("Creating training dataset...")
    # This call will now succeed because the builder classes are defined in this file.
    train_dataset = make_interleaved_dataset(
        dataset_kwargs_list=config.dataset_kwargs_list,
        traj_transform_kwargs=config.traj_transform_kwargs,
        frame_transform_kwargs=config.frame_transform_kwargs,
        train=True,
        batch_size=config.batch_size,
        shuffle_buffer_size=config.shuffle_buffer_size,
    )
    
    dataset_statistics = train_dataset.dataset_statistics
    train_data_iter = train_dataset.iterator()

    def process_batch(batch):
        # Fixed: handle data that's already in numpy format and ensure JAX compatibility
        def convert_to_numpy_and_fix_dtype(x, path=""):
            if hasattr(x, 'numpy'):
                x = x.numpy()
            
            # Convert to proper numpy array if needed and fix dtypes
            if isinstance(x, np.ndarray):
                # Fix object dtype arrays that can cause JAX errors
                if x.dtype == np.object_:
                    # For object dtypes, try to handle them without removing
                    if x.size > 0:
                        first_elem = x.flat[0]
                        if isinstance(first_elem, (str, bytes)):
                            # Keep strings as they are for language processing
                            if "language_instruction" in path:
                                return x  # Don't convert language instructions
                            # Convert other strings to a consistent format
                            try:
                                str_as_bytes = str(first_elem).encode('utf-8')[:100]
                                padded = str_as_bytes + b'\x00' * (100 - len(str_as_bytes))
                                return np.frombuffer(padded, dtype=np.int8).reshape(-1)
                            except:
                                return np.zeros(100, dtype=np.int8)
                        else:
                            try:
                                return np.array(x, dtype=np.float32)
                            except:
                                return np.zeros(x.shape, dtype=np.float32)
                    else:
                        return x
                elif x.dtype == np.float64:
                    # Convert float64 to float32 for JAX compatibility
                    x = x.astype(np.float32)
                elif x.dtype == np.int64:
                    # Convert int64 to int32 for JAX compatibility  
                    x = x.astype(np.int32)
                elif x.dtype == np.bool_:
                    # Convert boolean arrays to int32 for JAX compatibility
                    x = x.astype(np.int32)
                elif x.dtype.kind == 'U':  # Unicode strings
                    # Keep language instructions as strings
                    if "language_instruction" in path:
                        return x
                    # Convert other unicode strings to bytes and then to int8 array
                    try:
                        if x.size > 0:
                            str_val = str(x.flat[0])
                            str_as_bytes = str_val.encode('utf-8')[:100]
                            padded = str_as_bytes + b'\x00' * (100 - len(str_as_bytes))
                            return np.frombuffer(padded, dtype=np.int8).reshape(-1)
                        else:
                            return np.zeros(100, dtype=np.int8)
                    except:
                        return np.zeros(100, dtype=np.int8)
            
            return x
        
        # Use a recursive function to track the path for language instructions
        def convert_recursive(obj, path=""):
            if isinstance(obj, dict):
                return {k: convert_recursive(v, f"{path}/{k}") for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj))
            else:
                return convert_to_numpy_and_fix_dtype(obj, path)
        
        batch = convert_recursive(batch)
        
        # Process text
        batch = process_text(batch, text_processor)
        
        if "task" not in batch:
            batch["task"] = {}
        
        # Add task images using first timestep of observations
        if "image_primary" not in batch["task"] and "image_primary" in batch["observation"]:
            batch["task"]["image_primary"] = batch["observation"]["image_primary"][:, 0]  # (batch_size, H, W, C)
        
        # if "image_wrist" not in batch["task"] and "image_wrist" in batch["observation"]:
        #     batch["task"]["image_wrist"] = batch["observation"]["image_wrist"][:, 0]  # (batch_size, H, W, C)
        
        return batch

    logging.info("Loading first batch for model initialization...")
    example_batch = process_batch(next(train_data_iter))
    print("DEBUG: Observation keys in batch:", list(example_batch['observation'].keys()))
    logging.info("Successfully loaded example batch.")

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    model = OctoModel.from_config(
        model_config_dict,  # Use the updated config with VGGT tokenizer
        example_batch,
        text_processor,
        rng=init_rng,
        dataset_statistics=dataset_statistics
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    tx, lr_callable, _ = create_optimizer(model.params, **config.optimizer.to_dict())
    train_state = TrainState.create(model=model, tx=tx, rng=rng)

    save_dir = None
    if "save_dir" in config and config.save_dir is not None:
        save_dir = tf.io.gfile.join(config.save_dir, config.wandb.project, config.wandb.group or "", wandb_id)
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
    save_callback = SaveCallback(save_dir)

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"], 
            batch["task"], 
            batch["observation"]["timestep_pad_mask"],  # Fixed: use timestep_pad_mask instead of pad_mask
            train=train
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings, 
            batch["action"], 
            batch["observation"]["timestep_pad_mask"],  # Fixed: use timestep_pad_mask
            batch["action_pad_mask"],  # Fixed: use action_pad_mask instead of pad_mask
            train=train
        )
        return action_loss, action_metrics

    @partial(jax.jit, in_shardings=[replicated_sharding, dp_sharding])
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, dropout_rng, train=True)
        grad_norm = optax.global_norm(grads)
        new_state = state.apply_gradients(grads=grads, rng=rng)
        info.update({"grad_norm": grad_norm, "learning_rate": lr_callable(state.step)})
        return new_state, info

        logging.info("Initializing validation dataset and callbacks...")
        
        val_dataset = make_interleaved_dataset(
            dataset_kwargs_list=config["dataset_kwargs_list"],
            traj_transform_kwargs=config["traj_transform_kwargs"],
            frame_transform_kwargs=config["frame_transform_kwargs"], 
            train=False,
            batch_size=config["viz_kwargs"]["eval_batch_size"],
            shuffle_buffer_size=config["val_kwargs"]["val_shuffle_buffer_size"],
        )
        
        val_iterator = val_dataset.iterator()
        
        val_callback = ValidationCallback(
            loss_fn=loss_fn, 
            process_batch_fn=process_batch, 
            data_iterator=val_iterator, 
            num_batches=config["val_kwargs"]["num_val_batches"]
        )
        
        viz_callback = VisualizationCallback(
            text_processor=text_processor, 
            data_iterator=val_iterator, 
            **config["viz_kwargs"]
        )


    timer = Timer()
    logging.info("Starting training loop...")
    for i in tqdm.tqdm(range(int(config["num_steps"])), total=int(config["num_steps"]), dynamic_ncols=True):
        with timer("dataset"):
            batch = process_batch(next(train_data_iter))
            
        # Debug: Check batch dtypes before training step
        def check_dtypes(x, path=""):
            if isinstance(x, np.ndarray):
                if x.dtype == np.object_:
                    print(f"WARNING: Object dtype found at {path}, shape: {x.shape}")
                    print(f"Sample values: {x.flat[:min(5, x.size)]}")
                elif x.dtype.kind == 'U':  # Unicode strings
                    print(f"WARNING: Unicode string dtype {x.dtype} found at {path}")
                elif not np.issubdtype(x.dtype, np.number) and x.dtype != np.bool_:
                    print(f"WARNING: Non-numeric dtype {x.dtype} found at {path}")
            elif isinstance(x, dict):
                for k, v in x.items():
                    check_dtypes(v, f"{path}/{k}")
            elif isinstance(x, (list, tuple)):
                for i, v in enumerate(x):
                    check_dtypes(v, f"{path}[{i}]")
        

        if i == 0:  # Only on first iteration
            print("=== OFFLINE DEBUGGING DIMENSIONS ===")
            
            # 1. Check what's in the batch
            print("Batch observation keys:", list(batch["observation"].keys()))
            for key, value in batch["observation"].items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
            
            # Check task keys too
            print("Batch task keys:", list(batch.get("task", {}).keys()))
            for key, value in batch.get("task", {}).items():
                if hasattr(value, 'shape'):
                    print(f"  task {key}: shape = {value.shape}, dtype = {value.dtype}")
            
            # 2. Simple transformer call to get embeddings
            print("\n--- Offline Transformer output ---")
            def debug_transformer_simple():
                bound_module = model.module.bind({"params": train_state.model.params})
                transformer_output = bound_module.octo_transformer(
                    batch["observation"], 
                    batch["task"], 
                    batch["observation"]["timestep_pad_mask"],
                    train=False
                )
                
                print("Transformer output keys:", list(transformer_output.keys()))
                for key, token_group in transformer_output.items():
                    print(f"  {key}: tokens shape = {token_group.tokens.shape}")
                
                # Check action readout specifically
                action_readout = transformer_output["readout_action"]
                print(f"\nAction readout details:")
                print(f"  tokens shape: {action_readout.tokens.shape}")
                print(f"  mask shape: {action_readout.mask.shape}")
                
                # This is the key - how many dimensions go to diffusion model
                batch_size, window_size, num_tokens, token_dim = action_readout.tokens.shape
                pooled = action_readout.tokens.mean(axis=2)  # Pool over tokens
                flattened = pooled.reshape(batch_size, -1)
                print(f"  after pooling and flattening: {flattened.shape}")
                print(f"  -> Offline FINAL DIMENSION TO DIFFUSION: {flattened.shape[-1]}")
                
                return transformer_output
            
            try:
                transformer_embeddings = debug_transformer_simple()
            except Exception as e:
                print("Error calling transformer:", str(e))
            
            print("=== END Offline DEBUGGING ===")

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        if (i + 1) % config["log_interval"] == 0:
            wandb.log({"training": jax.device_get(update_info)}, step=i)

        if (i + 1) % config["eval_interval"] == 0:
            logging.info("Evaluating...")
            with timer("val"):
                wandb.log(val_callback(train_state, i + 1), step=i)
            with timer("visualize"):
                wandb.log(viz_callback(train_state, i + 1), step=i)

        if (i + 1) % config["save_interval"] == 0 and save_dir:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)

if __name__ == "__main__":
    app.run(main)