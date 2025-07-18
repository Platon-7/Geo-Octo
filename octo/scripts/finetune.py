import datetime
from functools import partial
import os

from absl import app, flags, logging
from ml_collections import config_flags, ConfigDict
import flax
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
default_config_file = os.path.join(os.path.dirname(__file__), "configs/config_offline.py")
config_flags.DEFINE_config_file("config", default_config_file, "File path to the training hyperparameter configuration.", lock_config=False)


# TFDS will automatically discover datasets in the data_dir if they have the right structure
# No explicit builders needed since the datasets were created with proper TFDS format

def main(_):
    initialize_compilation_cache()
    mesh = Mesh(jax.devices(), axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    tf.config.set_visible_devices([], "GPU")

    name = format_name_with_config(FLAGS.name, FLAGS.config.to_dict())
    wandb_id = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(config=FLAGS.config.to_dict(), id=wandb_id, name=name, mode="disabled" if FLAGS.debug else None, **FLAGS.config.wandb)

    pretrained_model = OctoModel.load_pretrained(FLAGS.config.pretrained_path, step=FLAGS.config.pretrained_step)


    config = FLAGS.config
    model_config = ConfigDict(pretrained_model.config)
    config.model = model_config.model
    config.model.update(FLAGS.config.get("update_config", {}).get("model", {}))

    text_processor = ModuleSpec.instantiate(config.text_processor)() if "text_processor" in config and config.text_processor else None
    
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
        # Fixed: handle data that's already in numpy format
        def convert_to_numpy(x):
            if hasattr(x, 'numpy'):
                return x.numpy()
            else:
                return x  # Already numpy or other format
        
        batch = tf.nest.map_structure(convert_to_numpy, batch)
        return process_text(batch, text_processor)

    logging.info("Loading first batch for model initialization...")
    example_batch = process_batch(next(train_data_iter))
    print("DEBUG: Observation keys in batch:", list(example_batch['observation'].keys()))
    logging.info("Successfully loaded example batch.")

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    model = OctoModel.from_config(
        config,
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
        transformer_embeddings = bound_module.octo_transformer(batch["observation"], batch["task"], batch["observation"]["pad_mask"], train=train)
        action_loss, action_metrics = bound_module.heads["action"].loss(transformer_embeddings, batch["action"], batch["observation"]["pad_mask"], batch["pad_mask"], train=train)
        return action_loss, action_metrics

    @partial(jax.jit, in_shardings=[replicated_sharding, dp_sharding])
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, dropout_rng, train=True)
        grad_norm = optax.global_norm(grads)
        new_state = state.apply_gradients(grads=grads, rng=rng)
        info.update({"grad_norm": grad_norm, "learning_rate": lr_callable(state.step)})
        return new_state, info

    val_callback, viz_callback = None, None

    timer = Timer()
    logging.info("Starting training loop...")
    for i in tqdm.tqdm(range(int(config.num_steps)), total=int(config.num_steps), dynamic_ncols=True):
        with timer("dataset"):
            batch = process_batch(next(train_data_iter))
            
        if "language_instruction" in batch.get("task", {}):
            # Get batch and window size from a tensor we know exists, like the vision tokens.
            b, w = batch["observation"]["vggt_tokens"].shape[:2]
            def reshape_lang(arr):
                # Reshape from (b*w, num_tokens) to (b, w, num_tokens)
                return arr.reshape(b, w, *arr.shape[1:])
            # Apply this to all arrays in the tokenized dictionary (input_ids, attention_mask)
            batch["task"]["language_instruction"] = jax.tree_map(reshape_lang, batch["task"]["language_instruction"])
            
        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        if (i + 1) % config.log_interval == 0:
            wandb.log({"training": jax.device_get(update_info)}, step=i)

        if (i + 1) % config.eval_interval == 0:
            if val_callback is None:
                logging.info("Initializing validation dataset and callbacks...")
                val_dataset = make_interleaved_dataset(
                    dataset_kwargs_list=config.dataset_kwargs_list,
                    traj_transform_kwargs=config.traj_transform_kwargs,
                    frame_transform_kwargs=config.frame_transform_kwargs, 
                    train=False,
                    batch_size=config.viz_kwargs.eval_batch_size,
                    shuffle_buffer_size=config.val_kwargs.val_shuffle_buffer_size,
                )
                val_callback = ValidationCallback(loss_fn=loss_fn, process_batch_fn=process_batch, data_iterator=val_dataset.iterator(), num_batches=config.val_kwargs.num_val_batches)
                viz_callback = VisualizationCallback(text_processor=text_processor, data_iterator=val_dataset.iterator(), **config.viz_kwargs)

            logging.info("Evaluating...")
            wandb.log(val_callback(train_state, i + 1), step=i)
            wandb.log(viz_callback(train_state, i + 1), step=i)

        if (i + 1) % config.save_interval == 0 and save_dir:
            save_callback(train_state, i + 1)

if __name__ == "__main__":
    app.run(main)