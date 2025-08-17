import os
import sys
import warnings
from typing import List

# JAX/Transformers compatibility shim (pre-import)
try:
	import jax.numpy as jnp
	if not hasattr(jnp, "DeviceArray"):
		jnp.DeviceArray = jnp.ndarray
		print("[FIX] Added DeviceArray compatibility shim")
except Exception:
	pass

# Suppress noisy warnings and tokenizer parallelism
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf

from octo.model.octo_model import OctoModel
from octo.data.dataset import make_interleaved_dataset


def np_mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def flatten_last_two_dims(x: np.ndarray) -> np.ndarray:
    # Expect (..., action_horizon, action_dim) -> (..., action_horizon * action_dim)
    return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline regression check for finetuned Octo model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to finetuned checkpoint directory (same one used by evaluate.py)")
    parser.add_argument("--batches", type=int, default=16, help="Number of validation batches to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--config_string", type=str, default="full,multimodal",
                        help="Offline finetuning config string used during training")
    parser.add_argument("--data_root", type=str, default=os.environ.get("LIBERO_DATA_ROOT", ""),
                        help="Override dataset root directory if config_offline points to a different machine path")
    args = parser.parse_args()

    print("=== OFFLINE REGRESSION CHECK ===")
    print(f"[INFO] Loading model from: {args.model_path}")
    model = OctoModel.load_pretrained(args.model_path)
    print("[OK] Model loaded")

    # Build validation dataset using the same config as training (no augmentation)
    cfg = None
    try:
        # Lazy import to avoid packaging path issues
        from scripts.configs.config_offline import get_config as _get_config
        cfg = _get_config(args.config_string)
        for ds in cfg.dataset_kwargs_list:
            ds["data_dir"] = args.data_root or "/gpfs/home4/pkarageorgis/geo_octo/libero_datasets"
            ds.pop("dataset_statistics", None)
    except Exception as e:
        print(f"[WARN] Could not import training config (octo.scripts.configs.config_offline): {e}")
        print("[INFO] Falling back to constructing dataset kwargs from checkpoint statistics")
        # Minimal shim: emulate the parts of config we need
        class CfgShim:
            pass
        cfg = CfgShim()
        cfg.window_size = 2
        cfg.val_kwargs = {"val_shuffle_buffer_size": 50}
        cfg.traj_transform_kwargs = {"action_horizon": 4}
        cfg.frame_transform_kwargs = {"resize_size": {"primary": (224, 224)}}
        # Build dataset kwargs list from dataset_statistics keys
        dataset_names = list(model.dataset_statistics.keys())
        # Heuristic: ignore non-dataset meta keys if present
        dataset_names = [n for n in dataset_names if isinstance(model.dataset_statistics.get(n), dict) and
                         "action" in model.dataset_statistics[n]]
        if not args.data_root:
            raise RuntimeError("--data_root is required when training config cannot be imported.")
        cfg.dataset_kwargs_list = []
        for name in dataset_names:
            cfg.dataset_kwargs_list.append(
                dict(
                    name=name,
                    data_dir=args.data_root,
                    dataset_statistics=None,  # let loader compute or find per-dataset cache
                    standardize_fn={
                        "module": "octo.octo.data.utils.data_utils",
                        "name": "standardize_libero_vggt",
                        "args": (),
                        "kwargs": {},
                    },
                    image_obs_keys={"primary": "image_primary"},
                    proprio_obs_key="proprio",
                    language_key="language_instruction",
                    action_proprio_normalization_type="normal",
                    filter_functions=[],
                )
            )

    # Patch dataset paths and missing statistics for this machine
    for ds_kwargs in cfg.dataset_kwargs_list:
        if args.data_root:
            ds_kwargs["data_dir"] = args.data_root
        stats_path = ds_kwargs.get("dataset_statistics")
        if isinstance(stats_path, str):
            try:
                if not tf.io.gfile.exists(stats_path):
                    # Drop stats so they are computed / loaded per-dataset
                    ds_kwargs.pop("dataset_statistics", None)
            except Exception:
                ds_kwargs.pop("dataset_statistics", None)

    # Reduce threads for stability in ad-hoc runs
    val_ds = make_interleaved_dataset(
        dataset_kwargs_list=cfg.dataset_kwargs_list,
        train=False,
        shuffle_buffer_size=cfg.val_kwargs.get("val_shuffle_buffer_size", 50),
        traj_transform_kwargs=dict(
            window_size=cfg.window_size,
            action_horizon=cfg.traj_transform_kwargs.get("action_horizon", 4),
            task_augment_strategy=None,
            task_augment_kwargs={},
        ),
        frame_transform_kwargs=dict(
            resize_size=cfg.frame_transform_kwargs.get("resize_size", {"primary": (224, 224)}),
            image_augment_kwargs={},
        ),
        batch_size=args.batch_size,
    )

    iterator = val_ds.iterator()

    # Collect metrics
    mse_list: List[float] = []
    mae_list: List[float] = []
    corr_list: List[float] = []

    for i in range(args.batches):
        batch = next(iterator)
        obs = batch["observation"]
        task = batch.get("task", {})

        # Match training preprocessing for tasks:
        # - language_instruction: convert to token ids (int32), not HF dict
        # - image_primary: copy goal image from first history frame, like finetune.py
        if "language_instruction" in task and model.text_processor is not None:
            strings = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                       for s in task["language_instruction"]]
            tokenized = model.text_processor.encode(strings)
            if isinstance(tokenized, dict) and "input_ids" in tokenized:
                task["language_instruction"] = np.asarray(tokenized["input_ids"], dtype=np.int32)
            else:
                # If encode returns embeddings, stack as-is
                task["language_instruction"] = np.asarray(tokenized)
        if "image_primary" not in task and "image_primary" in obs:
            # Copy the first image from the history window as "goal" image
            task["image_primary"] = np.asarray(obs["image_primary"])[:, 0]

        # Build a minimal pad_mask_dict for tasks
        task_pad = {
            k: np.ones(task[k].shape[0], dtype=bool)
            for k in task.keys() if k != "pad_mask_dict"
        }
        task["pad_mask_dict"] = task_pad

        # Sanity-check shapes vs example_batch
        try:
            _ = model.run_transformer(obs, task, obs["timestep_pad_mask"], train=False)
        except Exception as e:
            print(f"[FATAL] Shape mismatch feeding model: {e}")
            raise

        # Predict normalized actions (no unnormalization stats)
        actions_pred = model.sample_actions(
            obs,
            task,
            unnormalization_statistics=None,
            rng=jax.random.PRNGKey(i),
            argmax=False,
            temperature=1.0,
        )
        # Remove any sample dimension: (*, batch, horizon, dim) -> (batch, horizon, dim)
        if actions_pred.ndim == 4:
            actions_pred = actions_pred[0]

        actions_true = np.asarray(batch["action"])  # already normalized by data pipeline
        # Align shapes: both should be (batch, action_horizon, action_dim)
        if actions_true.ndim == 4:  # (batch, window, horizon, dim)
            actions_true = actions_true[:, 0]  # take current window head to match transformer readout
        if actions_pred.shape != actions_true.shape:
            print(f"[WARN] Pred shape {actions_pred.shape} != True shape {actions_true.shape}; attempting to align")
            min_h = min(actions_pred.shape[-2], actions_true.shape[-2])
            actions_pred = actions_pred[..., :min_h, :]
            actions_true = actions_true[..., :min_h, :]

        # Metrics
        mse = np_mean_squared_error(actions_pred, actions_true)
        mae = float(np.mean(np.abs(actions_pred - actions_true)))
        # Flatten for correlation
        a_flat = flatten_last_two_dims(actions_pred).reshape(actions_pred.shape[0], -1)
        t_flat = flatten_last_two_dims(actions_true).reshape(actions_true.shape[0], -1)
        # Average per-example Pearson correlation
        corr_vals = []
        for bi in range(a_flat.shape[0]):
            if np.std(a_flat[bi]) < 1e-6 or np.std(t_flat[bi]) < 1e-6:
                continue
            corr = np.corrcoef(a_flat[bi], t_flat[bi])[0, 1]
            if np.isfinite(corr):
                corr_vals.append(corr)
        corr = float(np.mean(corr_vals)) if corr_vals else float("nan")

        mse_list.append(mse)
        mae_list.append(mae)
        corr_list.append(corr)

        if (i + 1) % 4 == 0:
            print(f"[Batch {i+1}/{args.batches}] MSE={mse:.4f} MAE={mae:.4f} Corr={corr:.3f}")

    print("\n=== SUMMARY ===")
    print(f"Batches: {len(mse_list)}  BatchSize: {args.batch_size}")
    print(f"MSE (mean/median): {np.mean(mse_list):.4f} / {np.median(mse_list):.4f}")
    print(f"MAE (mean/median): {np.mean(mae_list):.4f} / {np.median(mae_list):.4f}")
    valid_corr = [c for c in corr_list if np.isfinite(c)]
    if valid_corr:
        print(f"Corr (mean/median): {np.mean(valid_corr):.3f} / {np.median(valid_corr):.3f}")
    else:
        print("Corr: n/a")

    # Heuristic: very high MSE (>>1.0 in normalized units) or near-zero corr suggests finetuning mismatch
    if np.mean(mse_list) > 1.0 or (valid_corr and np.mean(valid_corr) < 0.1):
        print("\n[DIAGNOSIS] Offline regression looks poor. This points to finetuning/training issues.")
    else:
        print("\n[DIAGNOSIS] Offline regression is reasonable. Mismatch likely in evaluation mapping/env semantics.")


if __name__ == "__main__":
    # Eager mode is fine for this diagnostic
    tf.config.run_functions_eagerly(True)
    main()