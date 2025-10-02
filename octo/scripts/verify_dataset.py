#!/usr/bin/env python3
import os
import random
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics.pairwise import cosine_similarity

# IMPORTANT: These imports must not bring in absl flags parsing
# Ensure PYTHONPATH includes the repo root so this resolves
from create_vggt_dataset_compressed_torch_ae import (
    preprocess_images_in_memory,
    TorchVGGTExtractor,
    AECompressor,
    resize_and_stack_per_layer,
)
import torch.nn.functional as F  # ensure present

def parse_args():
    p = ArgumentParser()
    p.add_argument("--original_data_dir", required=True)
    p.add_argument("--vggt_data_dir", required=True)
    p.add_argument("--dataset_name", default="libero_spatial_no_noops")
    p.add_argument("--ae_path", required=True)
    p.add_argument("--episode_idx_to_check", type=int, default=0)
    p.add_argument("--step_idx_to_check", type=int, default=5)
    p.add_argument("--vggt_input_res", type=int, default=518)
    p.add_argument("--ae_hidden", type=int, default=2048)
    p.add_argument("--vggt_agg_layers", type=int, default=24)
    p.add_argument("--vggt_layer_indices", default="3,10,16,22")
    p.add_argument("--checks", type=int, default=8, help="How many random (episode,step) pairs to verify in addition to the fixed one.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output_dir", default="verify_outputs")
    p.add_argument("--vggt_suffix", default="_vggt_compressed_torch", help="Suffix of the new TFDS dataset.")
    p.add_argument("--vggt_dataset_name", default="", help="Explicit TFDS name for the new dataset (overrides suffix).")

    # Verification thresholds
    p.add_argument("--atol", type=float, default=1e-3, help="Allclose absolute tolerance for float16 tokens.")
    p.add_argument("--rtol", type=float, default=1e-2, help="Allclose relative tolerance for float16 tokens.")
    p.add_argument("--cosine_threshold", type=float, default=0.999, help="Accept if mean cosine >= this value.")
    return p.parse_args()

def set_env_sane(args):
    # Keep TF on CPU to avoid CUDA noise
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    os.makedirs(args.output_dir, exist_ok=True)
    np.set_printoptions(suppress=True, precision=3)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def load_builders(args):
    import tensorflow_datasets as tfds

    # Original
    builder_original = tfds.builder(args.dataset_name, data_dir=args.original_data_dir)
    ds_original = builder_original.as_dataset(split="train")

    # New (prefer explicit name; else fallback to suffix; else directory discovery)
    vggt_name = args.vggt_dataset_name or f"{args.dataset_name}{args.vggt_suffix}"

    def _discover_builder_dir(base_dir, needle):
        # Try exact folder first
        exact = os.path.join(base_dir, needle)
        if tf.io.gfile.exists(exact):
            for child in tf.io.gfile.listdir(exact):
                d = os.path.join(exact, child)
                if tf.io.gfile.isdir(d) and tf.io.gfile.exists(os.path.join(d, "dataset_info.json")):
                    return d
        # Fuzzy search
        for entry in tf.io.gfile.listdir(base_dir):
            full = os.path.join(base_dir, entry)
            if tf.io.gfile.isdir(full) and needle.replace("_no_noops", "") in entry:
                for child in tf.io.gfile.listdir(full):
                    d = os.path.join(full, child)
                    if tf.io.gfile.isdir(d) and tf.io.gfile.exists(os.path.join(d, "dataset_info.json")):
                        return d
        return None

    try:
        builder_vggt = tfds.builder(vggt_name, data_dir=args.vggt_data_dir)
        print(f"[INFO] Loaded builder by name: {vggt_name}")
    except tfds.core.registered.DatasetNotFoundError:
        builder_dir = _discover_builder_dir(args.vggt_data_dir, vggt_name)
        if builder_dir is None:
            avail = []
            try:
                avail = [b for b in tfds.list_builders(data_dir=args.vggt_data_dir)]
            except Exception:
                pass
            raise RuntimeError(
                f"Dataset not found: '{vggt_name}'. "
                f"Try --vggt_dataset_name=libero_spatial_vggt_compressed_torch.\n"
                f"Available in data_dir: {avail}"
            )
        print(f"[INFO] Loaded builder from directory: {builder_dir}")
        builder_vggt = tfds.builder_from_directory(builder_dir=builder_dir)

    ds_vggt = builder_vggt.as_dataset(split="train")
    return ds_original, ds_vggt

def get_episode(ds, idx):
    # returns a dict with Tensors and a nested Dataset at ['steps']
    return next(iter(ds.skip(idx).take(1)))

def get_step(episode, step_idx):
    # returns a dict of Tensors for the step
    return next(iter(episode["steps"].skip(step_idx).take(1)))

def init_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_indices = [int(x) for x in args.vggt_layer_indices.split(",")] if args.vggt_agg_layers < 24 else None

    extractor = TorchVGGTExtractor(
        device=device,
        input_res=args.vggt_input_res,
        agg_layers=args.vggt_agg_layers,
        layer_indices=layer_indices,
    )

    # Probe L, D with dummy to construct AE
    dummy = np.zeros((1, 3, args.vggt_input_res, args.vggt_input_res), dtype=np.float32)
    klnd, _ = extractor.extract_layers(dummy)
    L, D = klnd.shape[1], klnd.shape[3]

    compressor = AECompressor(num_layers=L, input_dim=D, bottleneck_dim=512, hidden_dim=args.ae_hidden)
    sd = torch.load(args.ae_path, map_location="cpu")
    compressor.load_state_dict(sd)
    compressor = compressor.to(device).eval()

    return extractor, compressor, device

def _resize_layers_to_grid(klnd: np.ndarray, sqrt_n: int, target_grid: int) -> np.ndarray:
    # klnd: [K, L, N, D], with N = sqrt_n*sqrt_n
    K, L, N, D = klnd.shape
    assert N == sqrt_n * sqrt_n, f"Expected N={sqrt_n*sqrt_n}, got {N}"
    x = torch.from_numpy(klnd).float().view(K * L, sqrt_n, sqrt_n, D).permute(0, 3, 1, 2)  # (K*L, D, s, s)
    x_small = F.interpolate(x, size=(target_grid, target_grid), mode="bilinear", align_corners=False)
    x_small = x_small.permute(0, 2, 3, 1).contiguous().view(K, L, target_grid * target_grid, D)
    return x_small.numpy()  # [K, L, target_grid^2, D]

def recompute_tokens(image_hwc_uint8, extractor, compressor, input_res, target_grid: int):
    chw = preprocess_images_in_memory(np.expand_dims(image_hwc_uint8, axis=0), input_res)   # [1,3,H,W]
    klnd, sqrt_n = extractor.extract_layers(chw)                                            # [1,L,N,D]
    k_l_t_d = _resize_layers_to_grid(klnd, sqrt_n, target_grid)                             # [1,L,T,D], T=grid^2
    K, L, T, D = k_l_t_d.shape
    tokens_to_compress = torch.from_numpy(k_l_t_d).float().view(K * T, L, D)                # [T,L,D]
    with torch.no_grad():
        z = compressor.compress_tokens(tokens_to_compress)                                   # [T,512]
    return z.view(K, T, -1).cpu().numpy().astype(np.float16)[0]                              # [T,512]

def _select_image_np(orig_step, vggt_step):
    # Prefer the image stored in the VGGT dataset to avoid any tiny preprocessing drift
    obs_vggt = vggt_step.get("observation", {})
    img_vggt = None
    if "image" in obs_vggt:
        img_vggt = obs_vggt["image"]
    elif "image_primary" in obs_vggt:
        img_vggt = obs_vggt["image_primary"]

    if img_vggt is not None:
        return img_vggt.numpy() if hasattr(img_vggt, "numpy") else np.asarray(img_vggt)

    img_t = orig_step["observation"]["image"]
    return img_t.numpy() if hasattr(img_t, "numpy") else np.asarray(img_t)

def _ensure_absl_target_size(target_grid: int):
    g = f"{target_grid},{target_grid}"
    try:
        from absl import flags as absl_flags
        FLAGS = absl_flags.FLAGS
        if not FLAGS.is_parsed():
            FLAGS(['verify_dataset', f'--target_size={g}'])
        else:
            # If the flag exists, parse the new string value; otherwise attach it
            if 'target_size' in FLAGS:
                FLAGS['target_size'].parse(g)
            else:
                setattr(FLAGS, 'target_size', g)
    except Exception:
        # Fallback: monkey-patch module FLAGS used by the function
        try:
            import create_vggt_dataset_compressed_torch_ae as mod
            if not hasattr(mod, 'FLAGS'):
                class _Dummy: pass
                mod.FLAGS = _Dummy()
            setattr(mod.FLAGS, 'target_size', g)
        except Exception:
            pass

def _cosine_stats(a_fp16, b_fp16):
    # a_fp16, b_fp16: [T, D]
    a = a_fp16.astype(np.float32)
    b = b_fp16.astype(np.float32)
    # Per-token cosine similarity
    # Add small epsilon to norms to avoid NaNs
    eps = 1e-8
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    cos = np.sum(a_norm * b_norm, axis=1)  # [T]
    return float(np.mean(cos)), float(np.min(cos)), float(np.max(cos))

def verify_pair(ep_idx, step_idx, ds_original, ds_vggt, extractor, compressor, args):
    orig_ep = get_episode(ds_original, ep_idx)
    vggt_ep = get_episode(ds_vggt, ep_idx)
    orig_step = get_step(orig_ep, step_idx)
    vggt_step = get_step(vggt_ep, step_idx)

    img = _select_image_np(orig_step, vggt_step)

    stored = vggt_step["observation"]["vggt_tokens"].numpy()  # [T,512]
    target_grid = int(round(np.sqrt(stored.shape[0])))
    recomputed = recompute_tokens(img, extractor, compressor, args.vggt_input_res, target_grid)

    if stored.shape != recomputed.shape:
        return dict(ok=False, reason="shape_mismatch", ep=ep_idx, step=step_idx,
                    stored_shape=stored.shape, recomputed_shape=recomputed.shape)

    allclose_ok = np.allclose(recomputed, stored, atol=args.atol, rtol=args.rtol)
    mean_cos, min_cos, max_cos = _cosine_stats(recomputed, stored)
    cosine_ok = mean_cos >= args.cosine_threshold

    if not (allclose_ok or cosine_ok):
        diff = np.abs(recomputed - stored)
        return dict(
            ok=False,
            reason="content_mismatch",
            ep=ep_idx,
            step=step_idx,
            max_diff=float(diff.max()),
            mean_diff=float(diff.mean()),
            mean_cos=mean_cos,
            min_cos=min_cos,
            max_cos=max_cos,
        )

    return dict(
        ok=True,
        ep=ep_idx,
        step=step_idx,
        allclose_ok=bool(allclose_ok),
        mean_cos=mean_cos,
        min_cos=min_cos,
        max_cos=max_cos,
    )

def visual_check(orig_img, klnd, out_path):
    # Visualize uncompressed features’ similarity map
    uncompressed_features = klnd.mean(axis=1).squeeze(0)  # [N, D]
    sim = cosine_similarity(uncompressed_features)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(orig_img)
    axs[0].set_title("Original image")
    axs[0].axis("off")
    im = axs[1].imshow(sim)
    axs[1].set_title("Token similarity (uncompressed)")
    fig.colorbar(im, ax=axs[1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main(args):
    set_env_sane(args)
    ds_original, ds_vggt = load_builders(args)
    extractor, compressor, device = init_models(args)

    # Fixed pair
    results = []
    res = verify_pair(args.episode_idx_to_check, args.step_idx_to_check,
                      ds_original, ds_vggt, extractor, compressor, args)
    results.append(res)
    print(f"[Fixed] {res}")

    # Random pairs
    # Estimate dataset size (best-effort); fallback to 1000
    try:
        info = tfds.builder(args.dataset_name, data_dir=args.original_data_dir).info
        num_eps = info.splits["train"].num_examples
    except Exception:
        num_eps = 1000

    for _ in range(args.checks):
        e = random.randint(0, max(0, num_eps - 1))
        s = random.randint(0, max(0, 30))  # typical horizon; adjust if needed
        res = verify_pair(e, s, ds_original, ds_vggt, extractor, compressor, args)
        results.append(res)
        print(f"[Random] {res}")

    # Summary
    oks = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    print("\n=== Verification Summary ===")
    print(f"Total checks: {len(results)}  OK: {len(oks)}  ERR: {len(errs)}")
    if oks:
        allclose_count = sum(1 for r in oks if r.get("allclose_ok", False))
        mean_cos_ok = np.mean([r["mean_cos"] for r in oks]) if oks else float("nan")
        print(f"OK details: allclose_ok in {allclose_count}/{len(oks)}; mean of mean_cos over OKs = {mean_cos_ok:.6f}")
    if errs:
        err_counts = {}
        for r in errs:
            err_counts[r["reason"]] = err_counts.get(r["reason"], 0) + 1
        print("Errors by type:", json.dumps(err_counts, indent=2))

    # Optional visual sanity on the fixed example
    try:
        orig_ep = get_episode(ds_original, args.episode_idx_to_check)
        orig_step = get_step(orig_ep, args.step_idx_to_check)
        orig_img = orig_step["observation"]["image"].numpy()
        chw = preprocess_images_in_memory(np.expand_dims(orig_img, axis=0), args.vggt_input_res)
        klnd, sqrt_n = extractor.extract_layers(chw)
        out_path = os.path.join(args.output_dir, "dataset_verification.png")
        visual_check(orig_img, klnd, out_path)
        print(f"Saved visual sanity to {out_path}")
    except Exception as e:
        print(f"Visual sanity failed: {e}")

    # Exit nonzero if any failure
    if errs:
        raise SystemExit(2)

if __name__ == "__main__":
    args = parse_args()
    main(args)