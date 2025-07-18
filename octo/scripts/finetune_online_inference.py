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

import torch
import numpy as np
from vggt.models.vggt import VGGT

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

default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/debug_rollout_config.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


### START MODIFICATION ###
# We use a global dictionary to hold the model state AND the captured hook output.
VGGT_MODEL_STATE = {
    "model": None,
    "device": None,
    "dtype": None,
    "captured_output": None, # This will store the tensor from our hook
}

def run_vggt_inference(image_horizon_tensor: tf.Tensor) -> np.ndarray:
    """
    Runs VGGT inference and captures the true visual tokens using a forward hook.
    """
    # Step 1: Lazy-load the model and register the hook ONCE.
    if VGGT_MODEL_STATE["model"] is None:
        logging.info("First call to run_vggt_inference. Initializing VGGT model and hook...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        VGGT_MODEL_STATE["device"] = device
        VGGT_MODEL_STATE["dtype"] = dtype
        
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        model.eval()
        VGGT_MODEL_STATE["model"] = model

        def hook_fn(module, input, output):
            visual_tokens = output[0][0]
            VGGT_MODEL_STATE["captured_output"] = visual_tokens
        
        model.aggregator.register_forward_hook(hook_fn)
        logging.info("VGGT model and forward hook loaded successfully.")

    model = VGGT_MODEL_STATE["model"]
    device = VGGT_MODEL_STATE["device"]
    dtype = VGGT_MODEL_STATE["dtype"]

    # Step 2: Prepare the input tensor
    images_np = image_horizon_tensor.numpy()
    b, h, H, W, C = images_np.shape
    images_reshaped = images_np.reshape(b * h, H, W, C).transpose(0, 3, 1, 2)
    images_torch = torch.from_numpy(images_reshaped).to(device)

    # Step 3: Run inference to trigger the hook
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            model(images_torch)

    # Step 4: Retrieve the captured tensor
    captured_tensor = VGGT_MODEL_STATE["captured_output"]
    assert captured_tensor is not None, "Hook failed to capture output!"
    
    # The hook gives us a tensor, e.g. of shape (1, 8, 261, 2048).
    # We will reshape it into the final (b, h, num_tokens, embedding_dim) shape.
    # The number of elements must match.
    
    # We know the TRUE number of tokens and embedding dim from the previous log.
    true_num_tokens = 261
    true_embedding_dim = 2048
    
    vggt_tokens = captured_tensor.reshape(b, h, true_num_tokens, true_embedding_dim)

    # Step 5: Convert to NumPy for TensorFlow
    return vggt_tokens.to(torch.float32).cpu().numpy()

def process_batch(batch, text_processor):
    # Note: text_processor is now passed as an argument.
    
    batch = process_text(batch, text_processor)
    image_horizon = batch["observation"]["image_primary"]
    
    vggt_tokens = tf.py_function(
        run_vggt_inference,
        [image_horizon],
        tf.float32,
    )
    
    # Set the final shape for the TF graph using the TRUE dimensions.
    b, h, _, _, _ = image_horizon.shape
    num_vggt_tokens = 261
    vggt_embedding_dim = 2048
    vggt_tokens.set_shape([b, h, num_vggt_tokens, vggt_embedding_dim])

    batch["observation"]["vggt_tokens"] = vggt_tokens.numpy()
    del batch["dataset_name"]
    return batch

### END MODIFICATION ###

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

    ### START MODIFICATION ###

    bound_process_batch = partial(process_batch, text_processor=text_processor) # change process_batch with this below as well

    def enrich_observation(observation: dict) -> dict:
        """
        Takes an observation dict from any callback, passes its image to VGGT,
        and returns the observation enriched with the new tokens.
        """
        # The image is already in a 5D format that run_vggt_inference can handle.
        image_5d = observation["image_primary"]
        
        # Run our trusted inference function.
        vggt_tokens_5d = run_vggt_inference(tf.convert_to_tensor(image_5d))
        
        # Add the tokens to the observation and return it.
        observation["vggt_tokens"] = vggt_tokens_5d
        return observation

    ### END MODIFICATION ###


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
    train_data_iter = map(bound_process_batch, train_data_iter)
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
            tf.io.gfile.join(save_dir, "debug_rollout_config.json"), "w"
        ) as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        logging.warning("save_dir not passed in, not saving checkpoints")

    example_batch_spec = jax.tree_map(
        lambda arr: (list(arr.shape), str(arr.dtype)), example_batch
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
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.model.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

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
        process_batch_fn=bound_process_batch,
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
        observation_enricher_fn=enrich_observation,
        **FLAGS.config.viz_kwargs,
    )

    #########
    #
    # Optionally build visualizers for sim env evals
    #
    #########

    if "rollout_kwargs" in FLAGS.config:
        try:
            # This import will only succeed if libero is properly installed
            import libero.envs
            
            rollout_callback = RolloutVisualizationCallback(
                text_processor=text_processor,
                action_proprio_metadata=dataset.dataset_statistics["action"],
                observation_enricher_fn=enrich_single_observation,
                **FLAGS.config.rollout_kwargs.to_dict(),
            )
            logging.info("Successfully created RolloutVisualizationCallback.")
        except (ImportError, NameError) as e:
            rollout_callback = None
            logging.warning(f"Could not create RolloutVisualizationCallback due to missing dependency: {e}")
            logging.warning("Continuing without live rollouts. Offline visualization will still work.")
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
    for i in tqdm.tqdm(
        range(0, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

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

            # Uncomment the following lines to visualize the model's performance
            

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