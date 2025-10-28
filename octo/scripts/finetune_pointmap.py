import datetime
from functools import partial
import os
from typing import Optional, List, Tuple

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
import sys
import jaxlib
from jaxlib import version as jv
from flax.core import unfreeze
import jax.numpy as jnp

# --- Online VGGT pointmap computation ---
import numpy as np
import torch
from contextlib import nullcontext
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

print("==== TRAIN START ENV ====")
print("PYTHON exe:", sys.executable)
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("jax:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("cuda baked:", getattr(jv, "__cuda_version__", None))
print("cudnn baked:", getattr(jv, "__cudnn_version__", None))
print("jaxlib path:", jaxlib.__file__)
print("devices:", jax.devices())
print("=========================")

# Keep existing flags
flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
flags.DEFINE_bool("dump_train_images", False, "If True, save a few input images for inspection.")
flags.DEFINE_integer("dump_train_images_max", 20, "Max number of training images to dump.")
flags.DEFINE_string("dump_train_images_dir", "./train_image_dumps", "Directory to save dumped training images.")
flags.DEFINE_bool("use_vision_encoder", True, "If True then use Octo's vision encoder, else discard it.")

# Config file for pointmap finetuning
default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/config_pointmap.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

# --- New CLI flags for online VGGT pointmaps ---
flags.DEFINE_bool("vggt_use_cuda", True, "Use CUDA for VGGT if available.")
flags.DEFINE_integer("vggt_device_id", 0, "CUDA device index to run VGGT on.")
flags.DEFINE_integer("vggt_input_res", 224, "VGGT input resolution (square).")
flags.DEFINE_integer("vggt_eval_batch_size", 16, "Batch size for VGGT forward inside process_batch.")
flags.DEFINE_bool("jax_use_first_device_only", True, "Restrict JAX to the first visible GPU so VGGT can use another.")

# Profiling toggles
flags.DEFINE_bool("profile_vggt", False, "If True, print VGGT timing breakdowns per batch.")
flags.DEFINE_integer("profile_vggt_every", 50, "Print VGGT profile every N batches (1=every batch).")
flags.DEFINE_bool("compile_vggt", False, "If True, compile VGGT aggregator for potential speedups.")
flags.DEFINE_bool("overlap_vggt_with_train", True, "If True, compute pointmaps for the next batch in parallel with JAX train on current batch (2-GPU overlap).")
flags.DEFINE_integer("prefetch_batches", 2, "Number of future batches to precompute when overlapping.")

# Pointmap controls
flags.DEFINE_string("pointmap_key", "pointmap", "Observation key to store pointmap under (B,T,H,W,4).")
flags.DEFINE_bool("normalize_pointmap", True, "Per-image mean/std normalize XYZ (keep conf channel unchanged).")


# =========================
# Online VGGT pointmap helpers
# =========================

def _preprocess_images_for_vggt(images_np: np.ndarray, target_size: int) -> np.ndarray:
    """Aspect-preserving resize to nearest multiple of 14 and white-pad to square; returns CHW floats in [0,1]."""
    from PIL import Image

    processed_images = []
    for img_array in images_np:
        pil_image = Image.fromarray(img_array)

        if pil_image.mode == 'RGBA':
            background = Image.new('RGBA', pil_image.size, (255, 255, 255, 255))
            pil_image = Image.alpha_composite(background, pil_image)
        pil_image = pil_image.convert('RGB')

        width, height = pil_image.size
        if width >= height:
            new_width = target_size
            new_height = int(round(height * (new_width / width) / 14) * 14)
        else:
            new_height = target_size
            new_width = int(round(width * (new_height / height) / 14) * 14)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))

        h_padding = target_size - arr.shape[1]
        w_padding = target_size - arr.shape[2]
        pad_top = h_padding // 2
        pad_bottom = h_padding - pad_top
        pad_left = w_padding // 2
        pad_right = w_padding - pad_left
        arr = np.pad(arr, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=1.0)
        processed_images.append(arr)

    return np.stack(processed_images, axis=0)


class OnlineVGGTPointmap:
    def __init__(self):
        device = (
            torch.device(f'cuda:{int(FLAGS.vggt_device_id)}')
            if (FLAGS.vggt_use_cuda and torch.cuda.is_available())
            else torch.device('cpu')
        )
        self.device = device
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()
        self.input_res = int(FLAGS.vggt_input_res)
        # Optional compile: compile aggregator submodule
        if bool(getattr(FLAGS, 'compile_vggt', False)):
            try:
                self.model.aggregator = torch.compile(self.model.aggregator, mode="reduce-overhead", fullgraph=False)
            except Exception:
                pass
        logging.info("[PointMap Online] VGGT device=%s input_res=%d", self.device, self.input_res)

    @torch.no_grad()
    def compute_pointmap(self, images_bt3hw: np.ndarray) -> np.ndarray:
        """images_bt3hw: (B,T,H,W,3) -> returns (B,T,H',W',4) pointmaps (xyz+conf or depth+conf)."""
        do_profile = bool(FLAGS.profile_vggt)
        should_print = False
        if do_profile:
            OnlineVGGTPointmap._calls = getattr(OnlineVGGTPointmap, '_calls', 0) + 1
            should_print = (OnlineVGGTPointmap._calls % max(1, int(FLAGS.profile_vggt_every)) == 0)
        t0 = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        t1 = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        if t0: t0.record()

        b, t, h, w, c = images_bt3hw.shape
        flat = images_bt3hw.reshape(b * t, h, w, c)
        chw = _preprocess_images_for_vggt(flat, self.input_res)  # (N,3,H,W)

        out_list = []
        N = chw.shape[0]
        bs = max(1, int(FLAGS.vggt_eval_batch_size))
        amp_ctx = (
            torch.cuda.amp.autocast(enabled=True)
            if self.device.type == 'cuda' else nullcontext()
        )
        for i in range(0, N, bs):
            x = torch.from_numpy(chw[i:i+bs]).to(self.device)  # (k,3,H,W)
            x = x.unsqueeze(1)  # (k,1,3,H,W)
            with amp_ctx:
                preds = self.model(x)
            if isinstance(preds, dict) and 'world_points' in preds:
                pts = preds['world_points'][:, 0]  # (k,H,W,3)
                conf = preds.get('world_points_conf', None)
                conf = conf[:, 0][..., None] if conf is not None else torch.ones((*pts.shape[:3], 1), device=pts.device)
                out = torch.cat([pts, conf], dim=-1)  # (k,H,W,4)
            elif 'depth' in preds:
                depth = preds['depth'][:, 0, ..., 0][..., None]
                conf = preds.get('depth_conf', None)
                conf = conf[:, 0][..., None] if conf is not None else torch.ones_like(depth)
                zeros = torch.zeros_like(depth)
                out = torch.cat([zeros, zeros, depth, conf], dim=-1)
            else:
                raise RuntimeError("VGGT did not return point/depth predictions")
            out_list.append(out.detach().cpu().numpy())

        stacked = np.concatenate(out_list, axis=0)  # (N,H,W,4)
        if t1:
            t1.record(); torch.cuda.synchronize();
            if should_print:
                logging.info("[POINTMAP PROFILE] N=%d res=%d bs=%d | total=%.1fms", N, int(self.input_res), int(bs), t0.elapsed_time(t1))
        return stacked.reshape(b, t, stacked.shape[1], stacked.shape[2], 4)


def _normalize_pointmap(pm: np.ndarray, keep_conf: bool = True) -> np.ndarray:
    """Normalize XYZ per image; keep confidence unchanged."""
    if pm.ndim != 5 or pm.shape[-1] < 1:
        return pm
    x = pm.astype(np.float32)
    if keep_conf and x.shape[-1] >= 4:
        xyz = x[..., :3]
        conf = x[..., 3:4]
        mean = np.nanmean(xyz, axis=(-3, -2), keepdims=True)
        std = np.nanstd(xyz, axis=(-3, -2), keepdims=True) + 1e-6
        xyz = (xyz - mean) / std
        return np.concatenate([xyz, conf], axis=-1)
    else:
        mean = np.nanmean(x, axis=(-3, -2), keepdims=True)
        std = np.nanstd(x, axis=(-3, -2), keepdims=True) + 1e-6
        return (x - mean) / std


# =========================
# Main finetuning script (mirrors finetune_vggt_online; injects pointmaps)
# =========================

def main(_):
    # initialize_compilation_cache()
    raw_devices = jax.devices()
    devices = raw_devices[:1] if (FLAGS.jax_use_first_device_only and len(raw_devices) > 1) else raw_devices
    logging.info(
        f"""
        Octo Finetuning (PointMap Injection)
        ====================================
        Pretrained model: {FLAGS.config.pretrained_path}
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.name}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}

        # Devices: {len(devices)}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
    """
    )

    #########
    # Setup Jax Data Parallelism
    #########

    assert (
        FLAGS.config.batch_size % len(devices) == 0
    ), f"Batch size ({FLAGS.config.batch_size}) must be divisible by the number of devices ({len(devices)})"
    assert (
        FLAGS.config.viz_kwargs.eval_batch_size % len(devices) == 0
    ), f"Eval batch size ({FLAGS.config.viz_kwargs.eval_batch_size}) must be divisible by the number of devices ({len(devices)})"

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(devices, axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    #########
    # Setup WandB
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
    # Load Pretrained model + modify config minimally
    #########

    # Handle HF path: do not pass a step for HuggingFace checkpoints
    _pretrained_path = str(FLAGS.config.pretrained_path)
    _step = None if _pretrained_path.startswith("hf://") else getattr(FLAGS.config, "pretrained_step", None)
    pretrained_model = OctoModel.load_pretrained(
        _pretrained_path,
        step=_step,
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
    # Ensure pointmap encoder is present in model config
    model_cfg = config.get("model", {})
    if "pointmap_encoder" not in model_cfg:
        model_cfg["pointmap_encoder"] = ModuleSpec.create(
            "octo.model.components.vit_encoders:PointMapEncoder",
            in_channels=4,
            base_width=64,
            embed_dim=model_cfg.get("token_embedding_size", 512),
            pre_downsample=2,
            use_bfloat16=True,
        )
    # Align Octo module's expected pointmap key with CLI flag
    model_cfg["pointmap_input_key"] = str(getattr(FLAGS, "pointmap_key", "pointmap"))
    config["model"] = model_cfg

    # Apply any extra updates from the pointmap config file
    config.update(FLAGS.config.get("update_config", ConfigDict()))
    # Freeze Octo base explicitly; unfreeze pointmap encoder and its gates/projections
    # Optimizer handles this via frozen_keys; ensure they are present
    if "optimizer" not in config:
        config["optimizer"] = {}
    if "frozen_keys" not in config["optimizer"]:
        # Use the defaults from config_pointmap.py; leave as is otherwise
        pass
    config = config.to_dict()

    #########
    # Setup Data Loader
    #########

    # create text processor
    if config.get("text_processor") is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    pm_runner = OnlineVGGTPointmap()

    def process_batch(batch):
        # Keep existing text processing
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        if "task" not in batch:
            batch["task"] = {}

        # Compute online pointmap from already-augmented images if present
        obs = batch.get("observation", {})
        image_primary = obs.get("image_primary")
        try:
            if image_primary is not None:
                pointmap = pm_runner.compute_pointmap(np.asarray(image_primary))
                if bool(FLAGS.normalize_pointmap):
                    pointmap = _normalize_pointmap(pointmap)
                # Ensure shape matches (B,T,H,W,4) float32
                pmf32 = pointmap.astype(np.float32)
                if pmf32.ndim != 5 or pmf32.shape[-1] != 4:
                    raise ValueError(f"Pointmap wrong shape {pmf32.shape}; expected (B,T,H,W,4)")
                obs[str(FLAGS.pointmap_key)] = pmf32
                batch["observation"] = obs
                #logging.info("[PointMap] injected %s %s", FLAGS.pointmap_key, pointmap.shape)
        except Exception as e:
            logging.warning("Online pointmap computation failed for this batch: %s", e)

        # If using pointmap-only mode, optionally drop images
        if not FLAGS.use_vision_encoder:
            for k in list(obs.keys()):
                if "image" in k:
                    obs.pop(k, None)
            # No extra pad mask required for pointmap injection
        else:
            if "image_primary" in batch["observation"]:
                batch["task"]["image_primary"] = batch["observation"]["image_primary"][:, 0]

        return batch

    dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=True,
    )
    raw_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )

    # Build example_batch synchronously
    example_batch = process_batch(next(raw_iter))

    # Optional overlap: precompute next batches on a background thread
    from concurrent.futures import ThreadPoolExecutor, Future
    from collections import deque

    overlap_enabled = bool(FLAGS.overlap_vggt_with_train) and (len(jax.devices()) >= 1)
    prefetch_depth = max(0, int(FLAGS.prefetch_batches))
    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1) if overlap_enabled and prefetch_depth > 0 else None
    prefetch_q: deque[Future] = deque()

    def _submit_next():
        try:
            nxt = next(raw_iter)
        except StopIteration:
            return False
        fut = executor.submit(process_batch, nxt) if executor is not None else None
        if fut is not None:
            prefetch_q.append(fut)
        return True

    if executor is not None:
        # Pre-fill the queue
        for _ in range(prefetch_depth):
            if not _submit_next():
                break

    obs = example_batch.get("observation", {})
    img = obs.get("image_primary")
    if img is not None:
        print("[finetune-pointmap] image_primary shape:", getattr(img, "shape", None), "dtype:", getattr(img, "dtype", None))
    print("example_batch observation keys:", list(example_batch["observation"].keys()))

    #########
    # Load Model (init fresh + merge pretrained weights)
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
    
    # ===== Unfreeze last N transformer blocks + obs_* projection/adapter =====
    unfreeze_last_n = 4  # change to how many tail blocks you want trainable

    # Get total number of blocks from the loaded config
    num_layers = int(model.config["model"]["transformer_kwargs"]["num_layers"])

    # Start from the config's frozen_keys
    frozen = list(FLAGS.config.optimizer.frozen_keys)

    # 1) Remove the "freeze all blocks" wildcard
    frozen = [p for p in frozen if p != "octo_transformer.BlockTransformer_*"]

    # 2) Unfreeze obs_* projections + norm adapters (remove their freeze patterns)
    frozen = [
        p for p in frozen
        if not (p.startswith("octo_transformer.obs_") and
                ("_projection" in p or "_norm_adapter" in p))
    ]

    # 3) Freeze only encoder blocks [0 .. num_layers - unfreeze_last_n - 1]
    early_block_patterns = [
        f"octo_transformer.BlockTransformer_0.Transformer_0.encoderblock_{i}.*"
        for i in range(max(0, num_layers - unfreeze_last_n))
    ]

    # Commit back to flags (used by create_optimizer)
    FLAGS.config.optimizer.frozen_keys = tuple(frozen + early_block_patterns)
    print("[FINETUNE] Freezing early blocks:", early_block_patterns)
    print("[FINETUNE] Final frozen_keys size:", len(FLAGS.config.optimizer.frozen_keys))
    
    print("Total Layers")
    print(model.config["model"]["transformer_kwargs"]["num_layers"])

    # Merge pretrained params into freshly initialized model (copy matching shapes)
    try:
        from octo.utils.train_utils import merge_params
        model = model.replace(params=merge_params(model.params, pretrained_model.params))
        logging.info("Merged pretrained weights into model parameters (shape-matched).")
    except Exception as e:
        logging.warning("Could not merge pretrained weights: %s", e)

    # Optimizer and Train State
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
        rng=jax.random.PRNGKey(FLAGS.config.seed),
    )
    print("[FINETUNE] Frozen patterns:", FLAGS.config.optimizer.frozen_keys)

    #########
    # Save all metadata
    #########

    if FLAGS.config.save_dir is not None:
        # Allow full resume
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

        # Add window_size to config to ease eval
        new_config = ConfigDict(model.config)
        new_config["window_size"] = example_batch["observation"][
            "timestep_pad_mask"
        ].shape[1]
        model = model.replace(config=new_config)

        # Save finetuning config
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
    # Define loss, train_step, and eval_step
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

    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
        donate_argnums=(0,),
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        grad_norm = optax.global_norm(grads)
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
    # Resume from checkpoint
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
    # Callbacks
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

    if "rollout_kwargs" in FLAGS.config:
        try:
            import libero.envs  # type: ignore
            rollout_callback = RolloutVisualizationCallback(
                text_processor=text_processor,
                unnormalization_statistics=dataset.dataset_statistics["action"],
                **FLAGS.config.rollout_kwargs.to_dict(),
            )
        except Exception as e:
            rollout_callback = None
            logging.warning(f"Could not create RolloutVisualizationCallback: {e}")
    else:
        rollout_callback = None

    #########
    # Train loop
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
    jax_timer = Timer()

    for i in tqdm.tqdm(
        range(start_step, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps) - start_step,
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            if executor is None:
                # No overlap: process synchronously
                batch = process_batch(next(raw_iter))
            else:
                # Overlap: get the oldest completed future and top-up queue
                try:
                    batch = example_batch if i == start_step else prefetch_q.popleft().result()
                except IndexError:
                    # In case queue drained unexpectedly, fall back to sync
                    batch = process_batch(next(raw_iter))
                # Top up the queue
                while executor is not None and len(prefetch_q) < prefetch_depth:
                    if not _submit_next():
                        break

        if not _batch_check_printed:
            print("\n" + "="*50)
            print("DEBUG: Final batch check (what the model receives)!")
            print("Batch observation keys:", list(batch['observation'].keys()))
            key = FLAGS.pointmap_key
            if key in batch['observation']:
                pm_shape = batch['observation'][key].shape
                pm_dtype = batch['observation'][key].dtype
                print(f"  -> SUCCESS: '{key}' is in the final batch! shape={pm_shape} dtype={pm_dtype}")
            else:
                print(f"  -> CRITICAL WARNING: '{key}' missing in the final batch!")
            print("="*50 + "\n")
            _batch_check_printed = True

        if dump_enabled and dumped < FLAGS.dump_train_images_max:
            obs_dump = batch.get("observation", {})
            img_dump = obs_dump.get("image_primary")
            if img_dump is not None:
                try:
                    import numpy as _np
                    arr = _np.asarray(img_dump)
                    if arr.ndim >= 5:
                        frame = arr[0, 0]
                    elif arr.ndim == 4:
                        frame = arr[0]
                    else:
                        frame = arr
                    out_path = os.path.join(FLAGS.dump_train_images_dir, f"step_{i:06d}_idx_{dumped:03d}.png")
                    import imageio
                    imageio.imwrite(out_path, frame)
                    dumped += 1
                except Exception as _e:
                    pass

        jax_timer.tick("jax_train")
        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        jax_timer.tock("jax_train")
        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times(), "jax_step_timer": jax_timer.get_average_times()}, step=i
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

        # Early stop aligned with original script constraint (optional)
        if (i + 1) >= 100000:
            logging.info(
                "Early stopping at step %d (target 150000). Cosine scheduler remains configured for %d steps.",
                i + 1,
                int(FLAGS.config.num_steps),
            )
            if (i + 1) % FLAGS.config.save_interval != 0 and save_dir is not None:
                logging.info("Saving final checkpoint before early stop...")
                save_callback(train_state, i + 1)
            break

    # Cleanup executor
    try:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


if __name__ == "__main__":
    app.run(main)