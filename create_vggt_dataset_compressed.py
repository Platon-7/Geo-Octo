#!/usr/bin/env python3
"""
VGGT Dataset Creation with Intelligent Compression
Creates VGGT-enhanced datasets with smart compression to reduce memory usage.
"""

import os
import tensorflow as tf
import torch
import tqdm
from absl import app, flags, logging
import numpy as np
import tensorflow_datasets as tfds
from collections import defaultdict
from typing import Dict, Any
import pickle

from vggt.models.vggt import VGGT
from vggt_compression_analysis import VGGTCompressor

# Configure TF to not grab all GPU memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def get_compressed_vggt_tokens_for_episode(
    model, device, dtype, all_images_np: np.ndarray, 
    batch_size: int, compressor: VGGTCompressor
) -> np.ndarray:
    """
    Get VGGT tokens for an episode and compress them intelligently.
    
    Returns compressed tokens with much smaller memory footprint.
    """
    if all_images_np.size == 0:
        return np.array([])
    
    def _preprocess_image(image):
        resized = tf.image.resize(image, (224, 224), method=tf.image.ResizeMethod.BICUBIC)
        normalized = tf.cast(resized, tf.float32) / 255.0
        return normalized
    
    dataset = tf.data.Dataset.from_tensor_slices(all_images_np)
    dataset = dataset.map(_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    all_raw_tokens_list = []
    
    # Extract raw VGGT tokens
    for image_batch_tensor in dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE):
        captured_output_list = []
        
        def hook_fn(module, input, output):
            captured_output_list.append(output[0][0])
        
        hook_handle = model.aggregator.register_forward_hook(hook_fn)
        
        images_np_4d = image_batch_tensor.numpy()
        images_np_chw = images_np_4d.transpose(0, 3, 1, 2)
        images_np_5d = np.expand_dims(images_np_chw, axis=1)
        images_torch = torch.from_numpy(images_np_5d).to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            model(images_torch)
        
        hook_handle.remove()
        
        if not captured_output_list:
            raise RuntimeError("VGGT hook failed to capture output!")
        
        captured_tensor = captured_output_list[0]
        tokens_batch = captured_tensor.squeeze(1)  # Shape: (batch_size, 261, 2048)
        
        # Store raw tokens for compression
        all_raw_tokens_list.append(tokens_batch.cpu().numpy())
    
    if not all_raw_tokens_list:
        return np.array([])
    
    # Concatenate all raw tokens
    all_raw_tokens = np.concatenate(all_raw_tokens_list, axis=0)  # (episode_length, 261, 2048)
    
    # Apply intelligent compression
    compressed_tokens = compressor.compress(all_raw_tokens)  # (episode_length, 64, 256)
    
    return compressed_tokens.astype(np.float16)

def transpose_list_of_dicts(list_of_dicts):
    """Helper function to transpose list of dicts."""
    transposed = defaultdict(list)
    for d in list_of_dicts:
        for key, val in d.items():
            transposed[key].append(val)
    final_dict = {}
    for key, val_list in transposed.items():
        if isinstance(val_list[0], dict):
            final_dict[key] = transpose_list_of_dicts(val_list)
        else:
            try: 
                final_dict[key] = np.stack(val_list)
            except: 
                final_dict[key] = val_list
    return final_dict

class CompressedVggtDataset(tfds.core.GeneratorBasedBuilder):
    """A TFDS builder for adding compressed VGGT tokens to an existing robot dataset."""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release with compressed VGGT tokens.'}

    def __init__(self, original_builder, vggt_model, vggt_device, vggt_dtype, 
                 vggt_batch_size, compressor, **kwargs):
        self._original_builder = original_builder
        self._vggt_model = vggt_model
        self._vggt_device = vggt_device
        self._vggt_dtype = vggt_dtype
        self._vggt_batch_size = vggt_batch_size
        self._compressor = compressor
        self.name = f"{self._original_builder.name}_vggt_compressed"
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Defines the new dataset structure with compressed VGGT tokens."""
        original_info = self._original_builder.info

        step_features = dict(original_info.features['steps'].feature)
        observation_features = dict(step_features['observation'])
        
        # Use compressed dimensions instead of original (261, 2048)
        target_h, target_w = self._compressor.target_size
        observation_features['vggt_tokens'] = tfds.features.Tensor(
            shape=(target_h, target_w),  # e.g., (64, 256) instead of (261, 2048)
            dtype=np.float16,
            doc=f'Compressed VGGT tokens from the primary image using {self._compressor.method}.',
        )
        
        step_features['observation'] = tfds.features.FeaturesDict(observation_features)
        
        final_features = tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset(tfds.features.FeaturesDict(step_features)),
            'episode_metadata': original_info.features['episode_metadata']
        })
        
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"Libero dataset with compressed VGGT tokens ({target_h}x{target_w} vs 261x2048).",
            features=final_features,
            homepage="https://github.com/geopavlo/geo-octo",
            citation=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Specifies the splits to generate."""
        return {'train': self._generate_examples(split='train')}

    def _generate_examples(self, split: str):
        """Reads the original dataset, computes compressed tokens, and yields new examples."""
        ds = self._original_builder.as_dataset(split=split)
        num_episodes = self._original_builder.info.splits[split].num_examples

        for i, episode in enumerate(tqdm.tqdm(ds, total=num_episodes, desc=f"Generating {self.name}")):
            # Process episode in streaming fashion to save memory
            steps_list_of_dicts = []
            
            # Convert episode steps to list (this loads the episode)
            for step in tfds.as_numpy(episode['steps']):
                steps_list_of_dicts.append(step)
            
            if not steps_list_of_dicts:
                continue
            
            steps = transpose_list_of_dicts(steps_list_of_dicts)
            
            primary_image_key = 'image' 
            if primary_image_key not in steps['observation'] or len(steps['observation'][primary_image_key]) == 0:
                continue

            images_np = steps['observation'][primary_image_key]
            
            # Get compressed VGGT tokens
            compressed_vggt_tokens = get_compressed_vggt_tokens_for_episode(
                self._vggt_model, self._vggt_device, self._vggt_dtype, 
                images_np, self._vggt_batch_size, self._compressor
            )

            if compressed_vggt_tokens.size == 0 or len(compressed_vggt_tokens) != len(steps['action']):
                logging.warning(f"Token generation failed or length mismatch for episode {i}. Skipping.")
                continue

            # Create new steps with compressed tokens
            new_steps = []
            for t in range(len(steps['action'])):
                new_step = steps_list_of_dicts[t].copy()  # Make a copy to avoid modifying original
                new_step['observation']['vggt_tokens'] = compressed_vggt_tokens[t].astype(np.float16)
                new_steps.append(new_step)
            
            yield i, {
                'steps': new_steps, 
                'episode_metadata': tfds.as_numpy(episode['episode_metadata'])
            }

# Flags and Main Script
FLAGS = flags.FLAGS
flags.DEFINE_string("input_data_dir", None, "Path to the root directory containing the ORIGINAL sub-datasets.", required=True)
flags.DEFINE_string("output_data_dir", None, "Path where the NEW compressed TFDS datasets will be written.", required=True)
flags.DEFINE_string("compressor_path", None, "Path to saved compressor pickle file. If None, will fit new compressor.", required=False)
flags.DEFINE_string("compression_method", "hybrid", "Compression method: 'pca', 'svd', or 'hybrid'")
flags.DEFINE_string("target_size", "64,256", "Target size as 'height,width' (e.g., '64,256')")
flags.DEFINE_integer("vggt_batch_size", 32, "Batch size for running VGGT inference.")
flags.DEFINE_integer("compression_samples", 1000, "Number of samples to use for fitting compressor.")
flags.DEFINE_bool("overwrite", False, "Whether to overwrite existing datasets.")

def load_or_create_compressor(
    original_builders: list, 
    compressor_path: str, 
    method: str, 
    target_size: tuple,
    num_samples: int,
    vggt_model, vggt_device, vggt_dtype, vggt_batch_size
) -> VGGTCompressor:
    """Load existing compressor or create and fit a new one."""
    
    if compressor_path and os.path.exists(compressor_path):
        print(f"📂 Loading existing compressor from {compressor_path}")
        return VGGTCompressor.load_compressor(compressor_path)
    
    print(f"🧮 Creating new {method} compressor with target size {target_size}")
    
    # Sample VGGT tokens from first dataset to fit compressor
    print(f"📦 Extracting {num_samples} samples for compressor fitting...")
    sample_tokens = []
    
    first_builder = original_builders[0]
    ds = first_builder.as_dataset(split='train').take(50)  # Take first 50 episodes
    
    samples_collected = 0
    for episode in tqdm.tqdm(ds, desc="Collecting samples for compression fitting"):
        steps_list = list(tfds.as_numpy(episode['steps']))
        if not steps_list:
            continue
            
        steps = transpose_list_of_dicts(steps_list)
        
        if 'image' not in steps['observation']:
            continue
            
        images_np = steps['observation']['image']
        
        # Get raw VGGT tokens (uncompressed) for fitting
        raw_tokens = get_raw_vggt_tokens_for_episode(
            vggt_model, vggt_device, vggt_dtype, images_np, vggt_batch_size
        )
        
        if raw_tokens.size > 0:
            sample_tokens.append(raw_tokens)
            samples_collected += len(raw_tokens)
            
            if samples_collected >= num_samples:
                break
    
    if not sample_tokens:
        raise ValueError("Could not extract any VGGT tokens for compressor fitting!")
    
    # Concatenate samples and fit compressor
    all_samples = np.concatenate(sample_tokens, axis=0)[:num_samples]
    print(f"✅ Collected {len(all_samples)} samples with shape {all_samples.shape}")
    
    # Fit compressor
    compressor = VGGTCompressor(method=method, target_size=target_size)
    compressor.fit_compressor(all_samples)
    
    # Save compressor for future use
    save_path = f"vggt_compressor_{method}_{target_size[0]}x{target_size[1]}.pkl"
    compressor.save_compressor(save_path)
    
    return compressor

def get_raw_vggt_tokens_for_episode(model, device, dtype, all_images_np: np.ndarray, batch_size: int) -> np.ndarray:
    """Get raw (uncompressed) VGGT tokens for compressor fitting."""
    if all_images_np.size == 0:
        return np.array([])
    
    def _preprocess_image(image):
        resized = tf.image.resize(image, (224, 224), method=tf.image.ResizeMethod.BICUBIC)
        normalized = tf.cast(resized, tf.float32) / 255.0
        return normalized
    
    dataset = tf.data.Dataset.from_tensor_slices(all_images_np)
    dataset = dataset.map(_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    all_tokens_list = []
    
    for image_batch_tensor in dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE):
        captured_output_list = []
        def hook_fn(module, input, output):
            captured_output_list.append(output[0][0])
        hook_handle = model.aggregator.register_forward_hook(hook_fn)
        images_np_4d = image_batch_tensor.numpy()
        images_np_chw = images_np_4d.transpose(0, 3, 1, 2)
        images_np_5d = np.expand_dims(images_np_chw, axis=1)
        images_torch = torch.from_numpy(images_np_5d).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            model(images_torch)
        hook_handle.remove()
        if not captured_output_list:
            raise RuntimeError("VGGT hook failed to capture output!")
        captured_tensor = captured_output_list[0]
        tokens_batch = captured_tensor.squeeze(1)
        all_tokens_list.append(tokens_batch.cpu().numpy())
    
    if not all_tokens_list:
        return np.array([])
    return np.concatenate(all_tokens_list, axis=0)

def main(_):
    logging.set_verbosity(logging.INFO)
    
    # Parse target size
    target_size = tuple(map(int, FLAGS.target_size.split(',')))
    
    logging.info("Initializing VGGT model...")
    vggt_device = "cuda" if torch.cuda.is_available() else "cpu"
    vggt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(vggt_device).eval()
    logging.info(f"VGGT model loaded to {vggt_device}. Using {vggt_dtype} for inference.")

    input_root = FLAGS.input_data_dir
    output_root = FLAGS.output_data_dir
    dataset_names = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    logging.info(f"Found {len(dataset_names)} original sub-datasets to process: {dataset_names}")

    # Load original builders for compressor fitting
    original_builders = []
    for dataset_name in dataset_names:
        try:
            builder = tfds.builder(dataset_name, data_dir=input_root)
            original_builders.append(builder)
        except Exception as e:
            logging.warning(f"Could not load builder for {dataset_name}: {e}")

    if not original_builders:
        raise ValueError("No valid dataset builders found!")

    # Load or create compressor
    compressor = load_or_create_compressor(
        original_builders=original_builders,
        compressor_path=FLAGS.compressor_path,
        method=FLAGS.compression_method,
        target_size=target_size,
        num_samples=FLAGS.compression_samples,
        vggt_model=vggt_model,
        vggt_device=vggt_device,
        vggt_dtype=vggt_dtype,
        vggt_batch_size=FLAGS.vggt_batch_size
    )

    # Process each dataset with compression
    for dataset_name in dataset_names:
        logging.info(f"###### PROCESSING DATASET: {dataset_name} ######")
        try:
            original_builder = tfds.builder(dataset_name, data_dir=input_root)
            
            compressed_builder = CompressedVggtDataset(
                original_builder=original_builder,
                vggt_model=vggt_model,
                vggt_device=vggt_device,
                vggt_dtype=vggt_dtype,
                vggt_batch_size=FLAGS.vggt_batch_size,
                compressor=compressor,
                data_dir=output_root,
            )
            
            # Handle overwrite logic
            if FLAGS.overwrite and tf.io.gfile.exists(compressed_builder.data_dir):
                logging.warning(f"Overwriting existing dataset at {compressed_builder.data_dir}")
                tf.io.gfile.rmtree(compressed_builder.data_dir)
            
            compressed_builder.download_and_prepare()

            logging.info(f"Successfully created compressed TFDS dataset '{compressed_builder.name}' at '{output_root}'.")
            
            # Print compression stats
            target_h, target_w = target_size
            original_size = 261 * 2048 * 2  # bytes (float16)
            compressed_size = target_h * target_w * 2  # bytes (float16)
            compression_ratio = original_size / compressed_size
            
            logging.info(f"Compression stats:")
            logging.info(f"  Original size per timestep: {original_size/1024:.1f} KB")
            logging.info(f"  Compressed size per timestep: {compressed_size/1024:.1f} KB")
            logging.info(f"  Compression ratio: {compression_ratio:.1f}x")
            logging.info(f"  Variance preserved: {compressor.compression_stats.get('variance_preserved', 'N/A')}")

        except Exception as e:
            logging.error(f"Failed to process dataset {dataset_name}. Error: {e}", exc_info=True)

    logging.info(f"--- ALL DATASETS PROCESSED ---")
    logging.info(f"New compressed TFDS datasets written to: {FLAGS.output_data_dir}")
    logging.info(f"Memory reduction: ~{compression_ratio:.1f}x smaller than original VGGT tokens")

if __name__ == "__main__":
    app.run(main)