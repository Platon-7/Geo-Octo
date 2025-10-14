import os
from absl import app, flags, logging
from ml_collections import config_flags
import jax
import jax.numpy as jnp
import numpy as np

from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import TrainState, create_optimizer
from octo.utils.train_utils import format_name_with_config

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

    # Build model config: freeze base, add pointmap encoder at readout injection
    model_conf = cfg["model"]
    model_conf = dict(model_conf)
    # Add pointmap encoder module spec (CNN -> 512)
    pointmap_encoder_spec = ModuleSpec.create(
        "octo.model.components.vit_encoders:PointMapEncoder",
        in_channels=4,
        base_width=64,
        embed_dim=model_conf["token_embedding_size"],
    )
    model_conf["pointmap_encoder"] = pointmap_encoder_spec
    model_conf["pointmap_input_key"] = "pointmap"

    # Load pretrained model for shapes, then create from updated config
    logging.info("Loading pretrained model from %s", cfg["pretrained_path"])
    pretrained = OctoModel.load_pretrained(cfg["pretrained_path"], step=cfg["pretrained_step"])
    model = OctoModel.from_config(
        dict(cfg, model=model_conf),
        example_batch,
        text_processor=None,
        dataset_statistics=dataset.dataset_statistics,
    )

    # Freeze everything except pointmap encoder and readout gates
    frozen_keys = ("octo_transformer.*",)
    opt_conf = dict(cfg["optimizer"])  # copy
    opt_conf["frozen_keys"] = frozen_keys
    tx, lr_callable, param_norm_callable = create_optimizer(model.params, **opt_conf)
    rng = jax.random.PRNGKey(cfg.get("seed", 42))
    train_state = TrainState.create(rng=rng, model=model, tx=tx)

    logging.info("Model ready. To train: use existing train loop.")
    logging.info("Shapes: token_dim=%d; expecting pointmap embedding injected into readout.", model.config["model"]["token_embedding_size"])
    logging.info("IMPORTANT: remove legacy VGGT+vision concatenation prints; this script does additive readout injection only.")


if __name__ == "__main__":
    app.run(main)
