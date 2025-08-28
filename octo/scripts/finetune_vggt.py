import datetime
from functools import partial
import os

from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import optax
import tensorflow as tf
import tqdm
import wandb

from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)
from octo.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    process_text,
    Timer,
    TrainState,
)

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
flags.DEFINE_bool("dump_train_images", False, "If True, save a few input images for inspection.")
flags.DEFINE_integer("dump_train_images_max", 20, "Max number of training images to dump.")
flags.DEFINE_string("dump_train_images_dir", "./train_image_dumps", "Directory to save dumped training images.")

default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/finetune_config.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    initialize_compilation_cache()
    devices = jax.devices()
    logging.info(
        f"""
        Octo Finetuning Script
        ======================
        Pretrained model: {FLAGS.config.pretrained_path}
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.name}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}

        # Devices: {jax.device_count()}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
    """
    )

    #########
    #
    # Setup Jax Data Parallelism
    #
    #########

    assert (
        FLAGS.config.batch_size % len(devices) == 0
    ), f"Batch size ({FLAGS.config.batch_size}) must be divisible by the number of devices ({len(devices)})"
    assert (
        FLAGS.config.viz_kwargs.eval_batch_size % len(devices) == 0
    ), f"Eval batch size ({FLAGS.config.viz_kwargs.eval_batch_size}) must be divisible by the number of devices ({len(devices)})"

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    #########
    #
    # Setup WandB
    #
    #########

    name = format_name_with_config(
        FLAGS.name,
        FLAGS.config.to_dict(),
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        config=FLAGS.config.to_dict(),
        id=wandb_id,
        name=name,
        mode="disabled" if FLAGS.debug else None,
        **FLAGS.config.wandb,
    )

    #########
    #
    # Load Pretrained model + optionally modify config
    #
    #########

    pretrained_model = OctoModel.load_pretrained(
        FLAGS.config.pretrained_path,
        step=FLAGS.config.pretrained_step,
    )
    flat_config = flax.traverse_util.flatten_dict(
        pretrained_model.config, keep_empty_nodes=True
    )
    for d_key in flax.traverse_util.flatten_dict(
        FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    ):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]

    config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    config.update(FLAGS.config.get("update_config", ConfigDict()))
    config = config.to_dict()
    check_config_diff(config, pretrained_model.config)

    #########
    #
    # Setup Data Loader
    #
    #########

    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        if "task" not in batch:
            batch["task"] = {}
        if "image_primary" in batch["observation"]:
            # Assumes the goal is the first image in the window_size sequence
            batch["task"]["image_primary"] = batch["observation"]["image_primary"][:, 0]
        return batch

    dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )
    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    #########
    #
    # Load Pretrained Model
    #
    #########

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    #########
    #
    # Setup Optimizer and Train State
    #
    #########

    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    tx, lr_callable, param_norm_callable = create_optimizer(
        params,
        **FLAGS.config.optimizer.to_dict(),
    )
    train_state = TrainState.create(
        model=model,
        tx=tx,
        rng=rng,
    )

    #########
    #
    # Save all metadata
    #
    #########

    if FLAGS.config.save_dir is not None:
        # Allow full resume: if a resume_dir is provided, use it directly
        resume_dir = FLAGS.config.get("resume_dir", None)
        if resume_dir is not None and isinstance(resume_dir, str) and len(resume_dir) > 0:
            save_dir = resume_dir
        else:
            save_dir = tf.io.gfile.join(
                FLAGS.config.save_dir,
                FLAGS.config.wandb.project,
                FLAGS.config.wandb.group or "",
                wandb_id,
            )
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        logging.info("Saving to %s", save_dir)
        save_callback = SaveCallback(save_dir)

        # Add window_size to top of config, to make eval easier
        new_config = ConfigDict(model.config)
        new_config["window_size"] = example_batch["observation"][
            "timestep_pad_mask"
        ].shape[1]
        model = model.replace(config=new_config)

        # Save finetuning config since it's not saved by SaveCallback, i.e. as part of model.save_pretrained()
        with tf.io.gfile.GFile(
            tf.io.gfile.join(save_dir, "finetune_config.json"), "w"
        ) as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        logging.warning("save_dir not passed in, not saving checkpoints")

    example_batch_spec = jax.tree_map(
        lambda arr: (arr.shape, str(arr.dtype)), example_batch
    )
    wandb.config.update(
        dict(example_batch_spec=example_batch_spec), allow_val_change=True
    )

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########


    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # action head knows to pull out the "action" readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    # Data parallelism
    # Model is replicated across devices, data is split across devices
    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
        donate_argnums=(0,),  # donate state buffers to reduce HBM pressure
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)

        # Compute loss and grads
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )

        # Force optimizer.update inside the JIT (prevents the XLA duplicate-constant crash)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        grad_norm = optax.global_norm(grads)

        # Apply updates manually (avoid recomputing inside apply_gradients)
        new_params = optax.apply_updates(state.model.params, updates)
        new_model = state.model.replace(params=new_params)
        new_state = state.replace(model=new_model, opt_state=new_opt_state, rng=rng, step=state.step + 1)

        info.update({
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm_callable(state.model.params),
            "learning_rate": lr_callable(state.step),
        })
        return new_state, info
    

    #########
    #
    # Resume from checkpoint (full TrainState) if available
    #
    #########

    start_step = 0
    if save_dir is not None:
        try:
            latest_step = save_callback.state_checkpointer.latest_step()
        except Exception:
            latest_step = None
        if latest_step is not None:
            train_state = save_callback.state_checkpointer.restore(latest_step, items=train_state)
            start_step = int(train_state.step)
            logging.info("Restored checkpoint from %s at step %d", save_dir, start_step)

    #########
    #
    # Build validation & visualization callbacks
    #
    #########

    if FLAGS.config.modality == "image_conditioned":
        modes_to_evaluate = ["image_conditioned"]
    elif FLAGS.config.modality == "text_conditioned":
        modes_to_evaluate = ["text_conditioned"]
    elif FLAGS.config.modality == "multimodal":
        modes_to_evaluate = ["image_conditioned", "text_conditioned"]
    else:
        modes_to_evaluate = ["base"]

    dataset_kwargs_list = [FLAGS.config.dataset_kwargs]

    val_callback = ValidationCallback(
        loss_fn=loss_fn,
        process_batch_fn=process_batch,
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        modes_to_evaluate=modes_to_evaluate,
        **FLAGS.config.val_kwargs,
    )

    viz_callback = VisualizationCallback(
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        modes_to_evaluate=modes_to_evaluate,
        **FLAGS.config.viz_kwargs,
    )

    #########
    #
    # Optionally build visualizers for sim env evals
    #
    #########

    if "rollout_kwargs" in FLAGS.config:
        rollout_callback = RolloutVisualizationCallback(
            text_processor=text_processor,
            unnormalization_statistics=dataset.dataset_statistics["action"],
            **FLAGS.config.rollout_kwargs.to_dict(),
        )
    else:
        rollout_callback = None

    #########
    #
    # Train loop
    #
    #########

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    dumped = 0
    dump_enabled = FLAGS.dump_train_images
    if dump_enabled:
        try:
            import imageio
            os.makedirs(FLAGS.dump_train_images_dir, exist_ok=True)
        except Exception:
            dump_enabled = False
            logging.warning("Disabling dump_train_images: imageio not available or cannot create output dir.")
    
    _batch_check_printed = False 
    
    for i in tqdm.tqdm(
        range(start_step, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps) - start_step,
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)
            
        # --- THIS IS OUR NEW DEBUGGING PRINT STATEMENT ---
        if not _batch_check_printed:
            print("\n" + "="*50)
            print("DEBUG: Final batch check (what the model receives)!")
            print("Batch observation keys:", list(batch['observation'].keys()))
            if 'vggt_tokens' in batch['observation']:
                vggt_shape = batch['observation']['vggt_tokens'].shape
                vggt_dtype = batch['observation']['vggt_tokens'].dtype
                print(f"  -> SUCCESS: 'vggt_tokens' are in the final batch!")
                print(f"     Shape: {vggt_shape} (Batch, Window, H, W)")
                print(f"     DType: {vggt_dtype}")
            else:
                print("  -> CRITICAL WARNING: 'vggt_tokens' were dropped somewhere in the data pipeline!")

            # Additional check: ensure no image observations remain
            image_obs_keys = [k for k in batch['observation'].keys() if 'image' in k]
            if image_obs_keys:
                print(f"  -> WARNING: image observation keys still present: {image_obs_keys}")
            else:
                print("  -> SUCCESS: no image observations present; using only VGGT tokens.")
            print("="*50 + "\n")
            _batch_check_printed = True
        # --- END OF DEBUGGING PRINT STATEMENT ---

        # Optional: dump a few images to verify orientation
        if dump_enabled and dumped < FLAGS.dump_train_images_max:
            obs = batch.get("observation", {})
            img = obs.get("image_primary")
            if img is not None:
                # Expect shape (B, T, H, W, C); save first sample and first timestep
                try:
                    import numpy as _np
                    arr = _np.asarray(img)
                    if arr.ndim >= 5:
                        frame = arr[0, 0]
                    elif arr.ndim == 4:
                        frame = arr[0]
                    else:
                        frame = arr
                    out_path = os.path.join(FLAGS.dump_train_images_dir, f"step_{i:06d}_idx_{dumped:03d}.png")
                    imageio.imwrite(out_path, frame)
                    dumped += 1
                except Exception as _e:
                    pass

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")

            with timer("val"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i)

            with timer("visualize"):
                viz_metrics = viz_callback(train_state, i + 1)
                wandb_log(viz_metrics, step=i)

            if rollout_callback is not None:
                with timer("rollout"):
                    rollout_metrics = rollout_callback(train_state, i + 1)
                    wandb_log(rollout_metrics, step=i)

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    app.run(main)