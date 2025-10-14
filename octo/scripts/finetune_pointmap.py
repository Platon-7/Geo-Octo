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
import torch

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

# Online VGGT settings (mirrors prior online flags)
flags.DEFINE_integer("vggt_input_res", 224, "VGGT input resolution (square).")
flags.DEFINE_bool("vggt_use_cuda", True, "Use CUDA for VGGT online.")
flags.DEFINE_integer("vggt_device_id", 0, "CUDA device id for VGGT online.")
flags.DEFINE_integer("vggt_eval_batch_size", 16, "Batch size for VGGT online forward.")


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


def _preprocess_images_for_vggt(images_np: np.ndarray, target_size: int) -> np.ndarray:
    # images_np: (N,H,W,3) uint8
    from PIL import Image
    proc = []
    for img in images_np:
        im = Image.fromarray(img)
        if im.mode == 'RGBA':
            bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert('RGB')
        w, h = im.size
        if w >= h:
            new_w = target_size
            new_h = int(round(h * (new_w / w) / 14) * 14)
        else:
            new_h = target_size
            new_w = int(round(w * (new_h / h) / 14) * 14)
        im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        hp = target_size - arr.shape[1]
        wp = target_size - arr.shape[2]
        pt, pb = hp // 2, hp - hp // 2
        pl, pr = wp // 2, wp - wp // 2
        arr = np.pad(arr, ((0, 0), (pt, pb), (pl, pr)), mode='constant', constant_values=1.0)
        proc.append(arr)
    return np.stack(proc, axis=0)


class OnlineVGGTPointmap:
    def __init__(self, input_res: int, use_cuda: bool, device_id: int):
        device = f"cuda:{device_id}" if (use_cuda and torch.cuda.is_available()) else "cpu"
        self.device = torch.device(device)
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()
        self.input_res = int(input_res)
        logging.info("[PointMap Online] VGGT device=%s input_res=%d", self.device, self.input_res)

    @torch.no_grad()
    def compute(self, images_bt3hw: np.ndarray, batch_size: int) -> np.ndarray:
        # images_bt3hw: (B,T,H,W,3) uint8
        b, t, h, w, c = images_bt3hw.shape
        flat = images_bt3hw.reshape(b * t, h, w, c)
        chw = _preprocess_images_for_vggt(flat, self.input_res)  # (N,3,H,W)
        out_list = []
        N = chw.shape[0]
        for i in range(0, N, batch_size):
            x = torch.from_numpy(chw[i:i+batch_size]).to(self.device)  # (k,3,H,W)
            x = x.unsqueeze(1)  # (k,1,3,H,W)
            preds = self.model(x)
            if isinstance(preds, dict) and 'world_points' in preds:
                pts = preds['world_points'][:, 0]  # (k,H,W,3)
                conf = preds.get('world_points_conf', None)
                if conf is not None:
                    conf = conf[:, 0][..., None]  # (k,H,W,1)
                else:
                    conf = torch.ones((*pts.shape[:-1], 1), device=pts.device)
                out = torch.cat([pts, conf], dim=-1)  # (k,H,W,4)
            elif 'depth' in preds:
                depth = preds['depth'][:, 0, ..., 0][..., None]
                conf = preds.get('depth_conf', None)
                conf = conf[:, 0][..., None] if conf is not None else torch.ones_like(depth)
                # tile to xyz-like format (optional): here keep (H,W,2) -> expand to 4 by padding zeros
                zeros = torch.zeros_like(depth)
                out = torch.cat([zeros, zeros, depth, conf], dim=-1)
            else:
                raise RuntimeError("VGGT did not return point/depth predictions")
            out_list.append(out.detach().cpu().numpy())
        stacked = np.concatenate(out_list, axis=0)  # (N,H,W,4)
        logging.info("[PointMap Online] produced (N,H,W,C)=%s", stacked.shape)
        return stacked.reshape(b, t, stacked.shape[1], stacked.shape[2], 4)


def add_pointmap_to_batch(batch: dict, key: str, normalize: bool, runner: OnlineVGGTPointmap | None = None) -> dict:
    obs = batch.setdefault("observation", {})
    pm = obs.get(key)
    if pm is None:
        # Try to compute online from image_primary
        img = obs.get("image_primary")
        if runner is not None and img is not None:
            arr = np.asarray(img)
            logging.info("[PointMap Online] computing pointmap from image_primary %s", arr.shape)
            pm = runner.compute(arr, batch_size=FLAGS.vggt_eval_batch_size)
        else:
            logging.warning("[PointMap] '%s' missing and no online runner/images; skipping.", key)
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

    # Prime example batch and add pointmap key (compute online if missing)
    example_batch = next(data_iter)
    vggt_runner = OnlineVGGTPointmap(FLAGS.vggt_input_res, FLAGS.vggt_use_cuda, FLAGS.vggt_device_id)
    example_batch = add_pointmap_to_batch(example_batch, FLAGS.pointmap_key, FLAGS.normalize_pointmap, runner=vggt_runner)

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
        batch = add_pointmap_to_batch(batch, FLAGS.pointmap_key, FLAGS.normalize_pointmap, runner=vggt_runner)
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
