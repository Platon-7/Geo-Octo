import os
import datetime
from absl import app, flags, logging
from ml_collections import config_flags
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import tensorflow as tf
import wandb
from copy import deepcopy

from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import TrainState, create_optimizer, merge_params, Timer, format_name_with_config

FLAGS = flags.FLAGS

# Basic flags
flags.DEFINE_string("name", "pointmap_finetune", "Experiment name.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the pointmap finetuning config.",
    lock_config=False,
)

# Pointmap input key and normalization option
flags.DEFINE_string("pointmap_key", "pointmap", "Observations key for VGGT pointmap (B,T,H,W,C=4).")
flags.DEFINE_bool("normalize_pointmap", True, "Per-image mean/std normalize XYZ (conf unchanged).")


def _normalize_pointmap(pm: np.ndarray, keep_conf: bool = True) -> np.ndarray:
    # pm: (B,T,H,W,C)
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


def add_pointmap_to_batch(batch: dict, key: str, normalize: bool) -> dict:
    obs = batch.setdefault("observation", {})
    pm = obs.get(key)
    if pm is None:
        logging.warning("[PointMap] '%s' missing in observations; skipping injection for this batch.", key)
        return batch
    arr = np.asarray(pm)
    logging.info("[PointMap] raw %s shape=%s dtype=%s", key, arr.shape, arr.dtype)
    if normalize:
        arr = _normalize_pointmap(arr)
        logging.info("[PointMap] normalized %s -> dtype=%s", key, arr.dtype)
    # Store under standardized key expected by OctoTransformer
    obs["pointmap"] = arr
    return batch


def main(_):
    # Load config
    cfg = FLAGS.config.to_dict()

    # Setup WandB
    name = format_name_with_config(FLAGS.name, cfg)
    wandb_id = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(config=cfg, id=wandb_id, name=name, mode=None)

    # Build dataset
    dataset = make_single_dataset(
        cfg["dataset_kwargs"],
        traj_transform_kwargs=cfg["traj_transform_kwargs"],
        frame_transform_kwargs=cfg["frame_transform_kwargs"],
        train=True,
    )
    data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(cfg["shuffle_buffer_size"])  # requires enough RAM
        .batch(cfg["batch_size"])             # (B,T,*)
        .iterator()
    )

    # Prime example batch and add pointmap key
    example_batch = next(data_iter)
    example_batch = add_pointmap_to_batch(example_batch, FLAGS.pointmap_key, FLAGS.normalize_pointmap)

    # Load pretrained model for shapes, then create from updated config
    logging.info("Loading pretrained model from %s", cfg["pretrained_path"])
    pretrained = OctoModel.load_pretrained(cfg["pretrained_path"], step=cfg["pretrained_step"])
    # Start from pretrained config and apply updates from CLI config
    new_conf = deepcopy(pretrained.config)
    # Merge top-level overrides (non-structural)
    for k, v in cfg.items():
        if k in ("update_config", "config_delete_keys"):  # handled separately
            continue
        new_conf[k] = v
    # Apply config_delete_keys if provided
    def _delete_keys(tree, delete_spec):
        for k, v in delete_spec.items():
            if isinstance(v, dict) and k in tree:
                _delete_keys(tree[k], v)
                if not tree[k]:
                    tree.pop(k, None)
            elif v is True and k in tree:
                tree.pop(k, None)
        return tree
    if "config_delete_keys" in cfg:
        _delete_keys(new_conf, cfg["config_delete_keys"])  # in-place
    # Apply update_config (deep update)
    def _deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = _deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    # Ensure pointmap encoder is present
    upd = cfg.get("update_config", {})
    upd_model = upd.setdefault("model", {})
    upd_model.setdefault(
        "pointmap_encoder",
        ModuleSpec.create(
            "octo.model.components.vit_encoders:PointMapEncoder",
            in_channels=4,
            base_width=64,
            embed_dim=new_conf.get("model", {}).get("token_embedding_size", 512),
        ),
    )
    upd_model.setdefault("pointmap_input_key", "pointmap")
    _deep_update(new_conf, upd)

    model = OctoModel.from_config(
        new_conf,
        example_batch,
        text_processor=None,
        dataset_statistics=dataset.dataset_statistics,
    )
    # Merge pretrained params into new structure
    merged = merge_params(model.params, pretrained.params)
    model = model.replace(params=merged)

    # Freeze everything except pointmap encoder and readout gates
    frozen_keys = ("octo_transformer.*",)
    opt_conf = dict(cfg["optimizer"])  # copy
    opt_conf["frozen_keys"] = frozen_keys
    tx, lr_callable, param_norm_callable = create_optimizer(model.params, **opt_conf)
    rng = jax.random.PRNGKey(cfg.get("seed", 42))
    train_state = TrainState.create(rng=rng, model=model, tx=tx)

    # Loss fn using action head
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, True
        )
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.model.params)
        new_params = optax.apply_updates(state.model.params, updates)
        new_model = state.model.replace(params=new_params)
        new_state = state.replace(model=new_model, opt_state=new_opt_state, rng=rng, step=state.step + 1)
        info.update({
            "loss": loss,
        })
        return new_state, info

    # Training loop (minimal)
    timer = Timer()
    for step in tqdm.tqdm(range(int(cfg["num_steps"]))):
        batch = next(data_iter)
        batch = add_pointmap_to_batch(batch, FLAGS.pointmap_key, FLAGS.normalize_pointmap)
        train_state, info = train_step(train_state, batch)
        if (step + 1) % int(cfg["log_interval"]) == 0:
            info = jax.device_get(info)
            wandb.log({"training/loss": float(info["loss"])}, step=step)

    # Save checkpoint dir like finetune_vggt
    save_dir = cfg.get("save_dir")
    if save_dir:
        ckpt_dir = os.path.join(save_dir, FLAGS.name)
        tf.io.gfile.makedirs(ckpt_dir)
        model.save_pretrained(step=train_state.step, checkpoint_path=ckpt_dir)
        logging.info("Saved checkpoint to %s", ckpt_dir)


if __name__ == "__main__":
    app.run(main)
