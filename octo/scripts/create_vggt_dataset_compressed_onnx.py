#!/usr/bin/env python3
"""
Create a VGGT-compressed dataset using ONNX Runtime for token extraction.

This aligns training-time token generation with evaluation-time ONNX inference
to avoid distribution shift between PyTorch and ONNX paths.
"""
import os
from typing import Tuple, Optional, List

import numpy as np
from absl import app, flags, logging
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

try:
    import onnxruntime as ort
except Exception as _e:  # pragma: no cover
    ort = None

from vggt_compression_analysis import VGGTCompressor
from evaluation.supporting_files.load_fn import load_and_preprocess_images


# -------------------------
# Flags
# -------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string("input_data_dir", None, "Path to the root directory containing ORIGINAL sub-datasets.", required=True)
flags.DEFINE_string("output_data_dir", None, "Path where the NEW compressed TFDS datasets will be written.", required=True)

flags.DEFINE_string("vggt_onnx_path", None, "Path to VGGT ONNX model.", required=True)
flags.DEFINE_integer("vggt_input_res", 224, "Input resolution for ONNX VGGT model (square).")
flags.DEFINE_bool("vggt_use_cuda", True, "Use CUDAExecutionProvider if available.")

flags.DEFINE_string("compressor_path", None, "Path to a saved compressor .pkl file. If None, a new one is created.")
flags.DEFINE_string("compression_method", "pca", "Compression method (currently only 'pca').")
flags.DEFINE_string("target_size", "32,48", "Target compressed size as 'height,width'.")
flags.DEFINE_integer("compression_samples", 2500, "Number of samples for fitting a new compressor.")

flags.DEFINE_integer("batch_size_eval", 1, "ONNX inference batch size (images per call).")
flags.DEFINE_bool("overwrite", False, "Overwrite existing datasets under output_data_dir.")


"""
Note: We intentionally reuse evaluation helpers to ensure token generation is
IDENTICAL to evaluation (preprocessing + output selection).
We will save each frame to a temp PNG and call load_and_preprocess_images,
then feed the resulting NCHW into the ONNX session and extract tokens via
_extract_tokens_from_outputs.
"""


# -------------------------
# TFDS Builder using ONNX
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
            try:
                final[k] = np.stack(val_list)
            except Exception:
                final[k] = val_list
    return final


class CompressedVggtDatasetOnnx(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    def __init__(
        self,
        original_builder,
        session: "ort.InferenceSession",
        input_name: str,
        input_res: int,
        compressor: VGGTCompressor,
        **kwargs,
    ):
        self._original_builder = original_builder
        self._session = session
        self._input_name = input_name
        self._input_res = input_res
        self._compressor = compressor
        self.name = f"{self._original_builder.name}_vggt_compressed_onnx"
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        original_info = self._original_builder.info
        step_features = dict(original_info.features['steps'].feature)
        observation_features = dict(step_features['observation'])

        target_h, target_w = self._compressor.target_size
        observation_features['vggt_tokens'] = tfds.features.Tensor(
            shape=(target_h, target_w),
            dtype=np.float16,
            doc='Compressed VGGT tokens using ONNX-based extraction.',
        )

        step_features['observation'] = tfds.features.FeaturesDict(observation_features)
        final_features = tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset(tfds.features.FeaturesDict(step_features)),
            'episode_metadata': original_info.features['episode_metadata']
        })

        return tfds.core.DatasetInfo(
            builder=self,
            description=f"Libero dataset with compressed VGGT tokens (ONNX, {target_h}x{target_w}).",
            features=final_features,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {'train': self._generate_examples(split='train')}

    def _generate_examples(self, split: str):
        ds = self._original_builder.as_dataset(split=split)
        num_episodes = self._original_builder.info.splits[split].num_examples

        i = 0
        for episode in ds:
            steps_list_of_dicts = list(tfds.as_numpy(episode['steps']))
            if not steps_list_of_dicts:
                i += 1
                continue

            steps = transpose_list_of_dicts(steps_list_of_dicts)

            primary_key = 'image'  # consistent with existing builders
            if primary_key not in steps['observation'] or len(steps['observation'][primary_key]) == 0:
                continue

            images_np = steps['observation'][primary_key]  # (T, H, W, C)
            tokens_compressed = []

            # Process each frame
            for img in images_np:
                # Save to temp file and preprocess exactly like evaluation
                temp_img_path = "/tmp/vggt_dataset_frame.png"
                Image.fromarray(img).save(temp_img_path)
                batched = load_and_preprocess_images([temp_img_path])  # (1, 3, H, W) with current eval settings
                outputs = self._session.run(None, {self._input_name: batched})
                # Local copy of evaluation's token extraction logic
                tokens_2d = np.asarray(outputs[0])
                while tokens_2d.ndim > 2 and tokens_2d.shape[0] == 1:
                    tokens_2d = np.squeeze(tokens_2d, axis=0)
                if tokens_2d.ndim == 3:
                    tokens_2d = tokens_2d[0]

                # Compress to target grid
                compressed = self._compressor.compress(tokens_2d[None, ...])  # (1, Hc*Wc) or (1, Hc, Wc)
                if compressed.ndim == 3:
                    compressed = compressed[0]
                tokens_compressed.append(compressed.astype(np.float16))

            # Write back into steps
            for t in range(len(tokens_compressed)):
                steps_list_of_dicts[t]['observation']['vggt_tokens'] = tokens_compressed[t]

            yield i, {'steps': steps_list_of_dicts, 'episode_metadata': tfds.as_numpy(episode['episode_metadata'])}
            i += 1


# -------------------------
# Compressor utils (load or fit using ONNX tokens)
# -------------------------
def load_or_create_compressor(
    builders: list,
    session: "ort.InferenceSession",
    input_name: str,
    input_res: int,
    compressor_path: Optional[str],
    num_samples: int,
    target_size: Tuple[int, int],
) -> VGGTCompressor:
    if compressor_path and os.path.exists(compressor_path):
        return VGGTCompressor.load_compressor(compressor_path)

    logging.info("Creating new PCA compressor with target size %s", target_size)

    # Collect sample tokens from the first dataset
    samples: List[np.ndarray] = []
    first = builders[0]
    ds = first.as_dataset(split='train').take(100)
    count = 0
    for episode in ds:
        steps = list(tfds.as_numpy(episode['steps']))
        if not steps:
            continue
        trans = transpose_list_of_dicts(steps)
        if 'image' not in trans['observation']:
            continue
        for img in trans['observation']['image']:
            temp_img_path = "/tmp/vggt_dataset_frame.png"
            Image.fromarray(img).save(temp_img_path)
            batched = load_and_preprocess_images([temp_img_path])
            outputs = session.run(None, {input_name: batched})
            tokens_2d = np.asarray(outputs[0])
            while tokens_2d.ndim > 2 and tokens_2d.shape[0] == 1:
                tokens_2d = np.squeeze(tokens_2d, axis=0)
            if tokens_2d.ndim == 3:
                tokens_2d = tokens_2d[0]
            samples.append(tokens_2d)
            count += 1
            if count >= num_samples:
                break
        if count >= num_samples:
            break

    if not samples:
        raise ValueError("Could not extract any VGGT tokens to fit compressor.")

    all_samples = np.stack(samples, axis=0)  # (N, L, D)
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
    input_root = FLAGS.input_data_dir
    output_root = FLAGS.output_data_dir
    onnx_path = FLAGS.vggt_onnx_path
    input_res = FLAGS.vggt_input_res
    target_size = tuple(map(int, FLAGS.target_size.split(',')))

    providers = ['CUDAExecutionProvider'] if FLAGS.vggt_use_cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    logging.info("Loaded ONNX model %s with input '%s'", onnx_path, input_name)

    dataset_names = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    if not dataset_names:
        raise ValueError("No valid datasets found under input_data_dir")

    original_builders = [tfds.builder(name, data_dir=input_root) for name in dataset_names]

    compressor = load_or_create_compressor(
        original_builders,
        session,
        input_name,
        input_res,
        FLAGS.compressor_path,
        FLAGS.compression_samples,
        target_size,
    )

    # Iterate datasets and write compressed versions
    for builder in original_builders:
        logging.info("###### PROCESSING DATASET: %s ######", builder.name)
        try:
            new_builder = CompressedVggtDatasetOnnx(
                original_builder=builder,
                session=session,
                input_name=input_name,
                input_res=input_res,
                compressor=compressor,
                data_dir=output_root,
            )

            # Overwrite handling
            if FLAGS.overwrite and tf.io.gfile.exists(new_builder.data_dir):
                logging.warning("Overwriting existing dataset at %s", new_builder.data_dir)
                tf.io.gfile.rmtree(new_builder.data_dir)

            new_builder.download_and_prepare()
            logging.info("Successfully created TFDS dataset '%s' at '%s'.", new_builder.name, output_root)
        except Exception as e:
            logging.error("Failed to process dataset %s. Error: %s", builder.name, e, exc_info=True)


if __name__ == "__main__":
    app.run(main)

