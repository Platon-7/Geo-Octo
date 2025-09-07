import os
from typing import Tuple, Optional, List

import numpy as np
from absl import app, flags, logging
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm

try:
    import onnxruntime as ort
except Exception as _e:  # pragma: no cover
    ort = None

from vggt_compression_analysis import VGGTCompressor

# -------------------------
# Flags (No changes needed here)
# -------------------------
FLAGS = flags.FLAGS
# ... (all your flags remain the same) ...
flags.DEFINE_string("input_data_dir", None, "Path to the root directory containing ORIGINAL sub-datasets.", required=True)
flags.DEFINE_string("output_data_dir", None, "Path where the NEW compressed TFDS datasets will be written.", required=True)
flags.DEFINE_string("vggt_onnx_path", None, "Path to VGGT ONNX model.", required=True)
flags.DEFINE_integer("vggt_input_res", 224, "Input resolution for ONNX VGGT model (square).")
flags.DEFINE_bool("vggt_use_cuda", True, "Use CUDAExecutionProvider if available.")
flags.DEFINE_string("compressor_path", None, "Path to a saved compressor .pkl file. If None, a new one is created.")
flags.DEFINE_string("compression_method", "pca", "Compression method (currently only 'pca').")
flags.DEFINE_string("target_size", "32,48", "Target compressed size as 'height,width'.")
flags.DEFINE_integer("compression_samples", 2500, "Number of samples for fitting a new compressor.")
flags.DEFINE_integer("batch_size_eval", 32, "ONNX inference batch size (images per call). Default 32 for performance.")
flags.DEFINE_bool("overwrite", False, "Overwrite existing datasets under output_data_dir.")


# -------------------------
# CORRECT In-Memory Preprocessing (Mirrors Evaluation)
# (RGBA-on-white, aspect-preserving resize, 14-multiple rounding, bilinear resampling, CHW, white padding
# to 224). It’s per-image, so heterogeneous frame sizes are handled identically to evaluation.
# -------------------------
def preprocess_images_in_memory(images_np: np.ndarray, target_size: int) -> np.ndarray:
    """
    Exactly mirrors evaluation preprocessing (RGBA-on-white, aspect-preserving resize,
    14-multiple rounding, and padding) without using temporary files.
    """
    processed_images = []
    for img_array in images_np:
        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(img_array)

        # 1. Handle RGBA by alpha-compositing on a white background
        if pil_image.mode == 'RGBA':
            background = Image.new('RGBA', pil_image.size, (255, 255, 255, 255))
            pil_image = Image.alpha_composite(background, pil_image)
        pil_image = pil_image.convert('RGB')

        # 2. Calculate new dimensions, preserving aspect ratio and rounding to nearest 14
        width, height = pil_image.size
        if width >= height:
            new_width = target_size
            new_height = int(round(height * (new_width / width) / 14) * 14)
        else:
            new_height = target_size
            new_width = int(round(width * (new_height / height) / 14) * 14)

        # 3. Resize using BILINEAR resampling
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # 4. Convert to NumPy array and normalize to [0, 1]
        processed_arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        
        # 5. Transpose to (C, H, W) for the model
        processed_arr = np.transpose(processed_arr, (2, 0, 1))

        # 6. Pad with white (1.0) to make the image a square
        h_padding = target_size - processed_arr.shape[1]
        w_padding = target_size - processed_arr.shape[2]
        pad_top = h_padding // 2
        pad_bottom = h_padding - pad_top
        pad_left = w_padding // 2
        pad_right = w_padding - pad_left

        processed_arr = np.pad(
            processed_arr,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=1.0
        )
        processed_images.append(processed_arr)

    return np.stack(processed_images, axis=0)


# -------------------------
# TFDS Builder and Compressor Utils (No changes needed to the logic inside)
# -------------------------
def transpose_list_of_dicts(list_of_dicts):
    from collections import defaultdict
    transposed = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            transposed[k].append(v)
    final = {}
    for k, val_list in transposed.items():
        if isinstance(val_list[0], dict):
            final[k] = transpose_list_of_dicts(val_list)
        else:
            try: final[k] = np.stack(val_list)
            except Exception: final[k] = val_list
    return final


class CompressedVggtDatasetOnnx(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    def __init__(self, original_builder, session, input_name, input_res, compressor, **kwargs):
        self._original_builder = original_builder
        self._session = session
        self._input_name = input_name
        self._input_res = input_res
        self._compressor = compressor
        self.name = f"{self._original_builder.name}_vggt_compressed_onnx"
        super().__init__(**kwargs)

    def _info(self):
        original_info = self._original_builder.info
        step_features = dict(original_info.features['steps'].feature)
        observation_features = dict(step_features['observation'])
        # Store (64, 2048) features per image as requested
        observation_features['vggt_tokens'] = tfds.features.Tensor(
            shape=(64, 2048), dtype=np.float16,
            doc='VGGT per-image tokens after 24-layer aggregation and bilinear downsample to 64x2048.')
        step_features['observation'] = tfds.features.FeaturesDict(observation_features)
        final_features = tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset(tfds.features.FeaturesDict(step_features)),
            'episode_metadata': original_info.features['episode_metadata']})
        return tfds.core.DatasetInfo(builder=self,
            description="Libero dataset with VGGT tokens (24-layer aggregated, resized to 64x2048).",
            features=final_features)

    def _split_generators(self, dl_manager):
        return {'train': self._generate_examples(split='train')}

    def _generate_examples(self, split: str):
        ds = self._original_builder.as_dataset(split=split)
        num_episodes = self._original_builder.info.splits[split].num_examples
        batch_size = FLAGS.batch_size_eval
        i = 0
        for episode in tqdm(ds, total=num_episodes, desc=f"Processing {self._original_builder.name}"):
            steps_list_of_dicts = list(tfds.as_numpy(episode['steps']))
            if not steps_list_of_dicts:
                i += 1; continue
            steps = transpose_list_of_dicts(steps_list_of_dicts)
            if 'image' not in steps['observation'] or len(steps['observation']['image']) == 0:
                i += 1; continue
            images_np = steps['observation']['image']
            # Confirm original image resolution (expect 128x128x3)
            try:
                first_shape = images_np[0].shape if hasattr(images_np, '__len__') and len(images_np) > 0 else None
                logging.info("First episode image shape: %s", first_shape)
            except Exception:
                pass
            chw_images = preprocess_images_in_memory(images_np, self._input_res)
            # Collect per-image features of shape (64, 2048)
            per_image_features_64x2048 = []
            num_images = chw_images.shape[0]
            for j in range(0, num_images, batch_size):
                image_batch = chw_images[j:j+batch_size]  # [K, C, H, W]
                # ONNX expects [B, S, 3, H, W]; set B=1, S=K
                image_batch_5d = np.expand_dims(image_batch, axis=0)
                outputs = self._session.run(None, {self._input_name: image_batch_5d})
                # Build name->output map for robustness
                output_names = [o.name for o in self._session.get_outputs()]
                outputs_by_name = {name: arr for name, arr in zip(output_names, outputs)}
                if 'layer_patch_tokens' not in outputs_by_name:
                    logging.error("layer_patch_tokens not found in ONNX outputs: %s", list(outputs_by_name.keys()))
                    continue
                feats = np.asarray(outputs_by_name['layer_patch_tokens'])  # [1, K, 24, N, 2048]
                if feats.ndim != 5:
                    logging.error("Unexpected layer_patch_tokens rank: %s with shape %s", feats.ndim, feats.shape)
                    continue
                _, K, L, N, D = feats.shape
                if L != 24 or D != 2048:
                    logging.warning("Unexpected (L,D)=(%d,%d); expected (24,2048)", L, D)
                # Derive spatial size from N; expect nearly square (e.g., 37x37)
                sqrt_n = int(round(np.sqrt(N)))
                if sqrt_n * sqrt_n != N:
                    logging.warning("Token count %d is not a perfect square; cropping to %d", N, sqrt_n * sqrt_n)
                N_sq = sqrt_n * sqrt_n
                # For each image in this batch, produce (64, 2048)
                for k in range(K):
                    per_img = feats[0, k]              # [24, N, 2048]
                    per_img = per_img[:, :N_sq, :]     # crop if needed
                    per_img = per_img.reshape((L, sqrt_n, sqrt_n, D))  # [24, H, W, 2048]
                    # Bilinear resize to 8x8 per layer using TF
                    per_img_tf = tf.convert_to_tensor(per_img, dtype=tf.float32)  # treat 24 as batch
                    per_img_small = tf.image.resize(per_img_tf, size=(8, 8), method='bilinear', antialias=True)
                    per_img_small = per_img_small.numpy().reshape((L, 64, D))  # [24,64,2048]
                    # Fuse layers by mean -> [64,2048]
                    fused = per_img_small.mean(axis=0).astype(np.float16)
                    per_image_features_64x2048.append(fused)
            if not per_image_features_64x2048:
                i += 1; continue
            features_array = np.stack(per_image_features_64x2048, axis=0)  # [T, 64, 2048]
            if len(features_array) != len(steps_list_of_dicts):
                logging.warning(f"Token/step mismatch in episode {i}. Skipping.")
                i += 1; continue
            for t in range(len(features_array)):
                steps_list_of_dicts[t]['observation']['vggt_tokens'] = features_array[t]
            yield i, {'steps': steps_list_of_dicts, 'episode_metadata': tfds.as_numpy(episode['episode_metadata'])}
            i += 1


def load_or_create_compressor(builders, session, input_name, input_res, compressor_path, num_samples, target_size):
    if compressor_path and os.path.exists(compressor_path):
        return VGGTCompressor.load_compressor(compressor_path)

    logging.info("Creating new PCA compressor with target size %s", target_size)
    batch_size = FLAGS.batch_size_eval
    if batch_size < 16:
        logging.warning("Using small batch_size_eval=%d. This will be slow.", batch_size)

    samples = []
    first = builders[0]
    ds = first.as_dataset(split='train').take(200) # Take more episodes to ensure enough samples
    count = 0
    for episode in tqdm(ds, desc="Collecting Compressor Samples"):
        steps = list(tfds.as_numpy(episode['steps']))
        if not steps: continue
        trans = transpose_list_of_dicts(steps)
        if 'image' not in trans['observation']: continue
        episode_images = trans['observation']['image']
        if episode_images.size == 0: continue
        chw_images = preprocess_images_in_memory(episode_images, input_res)
        all_episode_tokens = []
        num_images = chw_images.shape[0]
        for i in range(0, num_images, batch_size):
            image_batch = chw_images[i:i + batch_size]
            outputs = session.run(None, {input_name: image_batch})
            tokens_batch = np.asarray(outputs[0])
            while tokens_batch.ndim > 2 and tokens_batch.shape[0] == 1:
                tokens_batch = np.squeeze(tokens_batch, axis=0)
            if tokens_batch.ndim == 2:
                tokens_batch = np.expand_dims(tokens_batch, axis=0)
            all_episode_tokens.append(tokens_batch)
        episode_tokens_array = np.concatenate(all_episode_tokens, axis=0)
        samples.extend(list(episode_tokens_array))
        count = len(samples)
        if count >= num_samples:
            break
    if not samples:
        raise ValueError("Could not extract any VGGT tokens to fit compressor.")
    all_samples = np.stack(samples[:num_samples], axis=0)
    compressor = VGGTCompressor(target_size=target_size)
    compressor.fit_compressor(all_samples)
    save_path = f"vggt_compressor_pca_{target_size[0]}x{target_size[1]}.pkl"
    compressor.save_compressor(save_path)
    logging.info("Saved new compressor to %s", save_path)
    return compressor


# -------------------------
# Main
# -------------------------
def main(_):
    if ort is None:
        raise RuntimeError("onnxruntime is not available. Please install it first.")

    logging.set_verbosity(logging.INFO)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Successfully enabled memory growth on {len(gpus)} GPU(s).")
        except RuntimeError as e:
            logging.error(f"Could not set memory growth on GPU: {e}")
            
    input_root = FLAGS.input_data_dir
    output_root = FLAGS.output_data_dir
    onnx_path = FLAGS.vggt_onnx_path
    input_res = FLAGS.vggt_input_res
    target_size = tuple(map(int, FLAGS.target_size.split(',')))

    # Create session options and enable all graph optimizations
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['CUDAExecutionProvider'] if FLAGS.vggt_use_cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    input_name = session.get_inputs()[0].name
    logging.info("Loaded ONNX model %s with input '%s' and ALL graph optimizations enabled.", onnx_path, input_name)

    dataset_names = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    logging.info("Found dataset directories under input_data_dir: %s", dataset_names)
    if not dataset_names:
        raise ValueError("No valid datasets found under input_data_dir")

    original_builders = []
    for name in dataset_names:
        logging.info("Initializing TFDS builder: %s", name)
        b = tfds.builder(name, data_dir=input_root)
        logging.info("Builder %s splits: %s", name, list(b.info.splits.keys()))
        original_builders.append(b)

    logging.info("Starting compressor load/fit stage...")
    compressor = load_or_create_compressor(
        original_builders, session, input_name, input_res,
        FLAGS.compressor_path, FLAGS.compression_samples, target_size)
    logging.info("Compressor ready. Target size=%s", target_size)

    for builder in original_builders:
        logging.info("###### PROCESSING DATASET: %s ######", builder.name)
        try:
            new_builder = CompressedVggtDatasetOnnx(
                original_builder=builder, session=session, input_name=input_name,
                input_res=input_res, compressor=compressor, data_dir=output_root)
            
            # Correct overwrite logic
            dataset_output_dir = os.path.join(output_root, new_builder.name)
            if FLAGS.overwrite and tf.io.gfile.exists(dataset_output_dir):
                logging.warning("Overwriting existing dataset at %s", dataset_output_dir)
                tf.io.gfile.rmtree(dataset_output_dir)

            new_builder.download_and_prepare()
            logging.info("Successfully created TFDS dataset '%s' at '%s'.", new_builder.name, output_root)
        except Exception as e:
            logging.error("Failed to process dataset %s. Error: %s", builder.name, e, exc_info=True)


if __name__ == "__main__":
    app.run(main)