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

# --- REFINED: A single, unified function for getting tokens ---
def get_vggt_tokens_for_episode(
    model, device, dtype, all_images_np: np.ndarray, 
    batch_size: int, compressor: VGGTCompressor = None
) -> np.ndarray:
    """
    Gets VGGT tokens for an episode. If a compressor is provided, it returns
    compressed tokens. Otherwise, it returns the raw, uncompressed tokens.
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
        
        tokens_batch = captured_output_list[0].squeeze(1)
        all_raw_tokens_list.append(tokens_batch.cpu().numpy())
    
    if not all_raw_tokens_list:
        return np.array([])
    
    all_raw_tokens = np.concatenate(all_raw_tokens_list, axis=0)
    
    # If a compressor is provided, use it. Otherwise, return raw tokens.
    if compressor:
        compressed_tokens = compressor.compress(all_raw_tokens)
        return compressed_tokens.astype(np.float16)
    else:
        return all_raw_tokens

def transpose_list_of_dicts(list_of_dicts):
    transposed = defaultdict(list)
    for d in list_of_dicts:
        for key, val in d.items():
            transposed[key].append(val)
    final_dict = {}
    for key, val_list in transposed.items():
        if isinstance(val_list[0], dict):
            final_dict[key] = transpose_list_of_dicts(val_list)
        else:
            try: final_dict[key] = np.stack(val_list)
            except: final_dict[key] = val_list
    return final_dict

class CompressedVggtDataset(tfds.core.GeneratorBasedBuilder):
    """A TFDS builder for adding compressed VGGT tokens to a robot dataset."""
    VERSION = tfds.core.Version('1.0.0')

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
        original_info = self._original_builder.info
        step_features = dict(original_info.features['steps'].feature)
        observation_features = dict(step_features['observation'])
        
        target_h, target_w = self._compressor.target_size
        observation_features['vggt_tokens'] = tfds.features.Tensor(
            shape=(target_h, target_w),
            dtype=np.float16,
            doc=f'Compressed VGGT tokens using {self._compressor.method}.',
        )
        
        step_features['observation'] = tfds.features.FeaturesDict(observation_features)
        final_features = tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset(tfds.features.FeaturesDict(step_features)),
            'episode_metadata': original_info.features['episode_metadata']
        })
        
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"Libero dataset with compressed VGGT tokens ({target_h}x{target_w}).",
            features=final_features,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {'train': self._generate_examples(split='train')}

    def _generate_examples(self, split: str):
        ds = self._original_builder.as_dataset(split=split)
        num_episodes = self._original_builder.info.splits[split].num_examples

        for i, episode in enumerate(tqdm.tqdm(ds, total=num_episodes, desc=f"Generating {self.name}")):
            steps_list_of_dicts = list(tfds.as_numpy(episode['steps']))
            if not steps_list_of_dicts: continue
            
            steps = transpose_list_of_dicts(steps_list_of_dicts)
            
            primary_image_key = 'image' 
            if primary_image_key not in steps['observation'] or len(steps['observation'][primary_image_key]) == 0: continue

            images_np = steps['observation'][primary_image_key]
            
            compressed_tokens = get_vggt_tokens_for_episode(
                self._vggt_model, self._vggt_device, self._vggt_dtype, 
                images_np, self._vggt_batch_size, compressor=self._compressor
            )

            if compressed_tokens.size == 0 or len(compressed_tokens) != len(steps['action']):
                logging.warning(f"Token generation failed or mismatch for episode {i}. Skipping.")
                continue

            for t in range(len(steps['action'])):
                steps_list_of_dicts[t]['observation']['vggt_tokens'] = compressed_tokens[t]
            
            yield i, {'steps': steps_list_of_dicts, 'episode_metadata': tfds.as_numpy(episode['episode_metadata'])}

# --- MOVED and CORRECTED: This function must be defined before it is called ---
def load_or_create_compressor(
    original_builders: list, compressor_path: str, method: str, target_size: tuple,
    num_samples: int, vggt_model, vggt_device, vggt_dtype, vggt_batch_size
) -> VGGTCompressor:
    """Load existing compressor or create and fit a new one."""
    
    if compressor_path and os.path.exists(compressor_path):
        return VGGTCompressor.load_compressor(compressor_path)
    
    logging.info(f"Creating new {method} compressor with target size {target_size}")
    logging.info(f"Note: Compressor will be trained on samples from the first dataset: {original_builders[0].name}")
    
    sample_tokens = []
    first_builder = original_builders[0]
    ds = first_builder.as_dataset(split='train').take(100)
    
    for episode in tqdm.tqdm(ds, desc=f"Collecting {num_samples} samples for compression fitting"):
        steps_list = list(tfds.as_numpy(episode['steps']))
        if not steps_list: continue
        steps = transpose_list_of_dicts(steps_list)
        if 'image' not in steps['observation']: continue
        
        raw_tokens = get_vggt_tokens_for_episode( # Use the unified function
            vggt_model, vggt_device, vggt_dtype, steps['observation']['image'], vggt_batch_size, compressor=None
        )
        if raw_tokens.size > 0:
            sample_tokens.append(raw_tokens)
        if sum(len(t) for t in sample_tokens) >= num_samples:
            break
    
    if not sample_tokens:
        raise ValueError("Could not extract any VGGT tokens for compressor fitting!")
    
    all_samples = np.concatenate(sample_tokens, axis=0)[:num_samples]
    logging.info(f"✅ Collected {len(all_samples)} samples with shape {all_samples.shape}")
    
    compressor = VGGTCompressor(target_size=target_size)
    compressor.fit_compressor(all_samples)
    
    save_path = f"vggt_compressor_{method}_{target_size[0]}x{target_size[1]}.pkl"
    compressor.save_compressor(save_path)
    
    return compressor

# --- Flags and Main Script ---
FLAGS = flags.FLAGS
flags.DEFINE_string("input_data_dir", None, "Path to the root directory containing the ORIGINAL sub-datasets.", required=True)
flags.DEFINE_string("output_data_dir", None, "Path where the NEW compressed TFDS datasets will be written.", required=True)
flags.DEFINE_string("compressor_path", None, "Path to a saved compressor .pkl file. If None, a new one is created.")
flags.DEFINE_string("compression_method", "pca", "Compression method to use (currently only 'pca').")
flags.DEFINE_string("target_size", "32,48", "Target compressed size as 'height,width'.")
flags.DEFINE_integer("vggt_batch_size", 32, "Batch size for running VGGT inference.")
flags.DEFINE_integer("compression_samples", 2500, "Number of samples to use for fitting a new compressor.")
flags.DEFINE_bool("overwrite", False, "Whether to overwrite existing datasets.")

def main(_):
    logging.set_verbosity(logging.INFO)
    target_size = tuple(map(int, FLAGS.target_size.split(',')))
    
    logging.info("Initializing VGGT model...")
    vggt_device = "cuda" if torch.cuda.is_available() else "cpu"
    vggt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(vggt_device).eval()
    logging.info(f"VGGT model loaded to {vggt_device}.")

    input_root = FLAGS.input_data_dir
    output_root = FLAGS.output_data_dir
    dataset_names = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    original_builders = [tfds.builder(name, data_dir=input_root) for name in dataset_names]
    if not original_builders:
        raise ValueError("No valid dataset builders found in the input directory!")

    compressor = load_or_create_compressor(
        original_builders, FLAGS.compressor_path, FLAGS.compression_method, target_size,
        FLAGS.compression_samples, vggt_model, vggt_device, vggt_dtype, FLAGS.vggt_batch_size
    )

    for builder in original_builders:
        logging.info(f"###### PROCESSING DATASET: {builder.name} ######")
        try:
            compressed_builder = CompressedVggtDataset(
                original_builder=builder, vggt_model=vggt_model, vggt_device=vggt_device,
                vggt_dtype=vggt_dtype, vggt_batch_size=FLAGS.vggt_batch_size,
                compressor=compressor, data_dir=output_root
            )
            
            if FLAGS.overwrite and tf.io.gfile.exists(compressed_builder.data_dir):
                logging.warning(f"Overwriting existing dataset at {compressed_builder.data_dir}")
                tf.io.gfile.rmtree(compressed_builder.data_dir)
            
            compressed_builder.download_and_prepare()
            logging.info(f"Successfully created compressed TFDS dataset '{compressed_builder.name}'.")

        except Exception as e:
            logging.error(f"Failed to process dataset {builder.name}. Error: {e}", exc_info=True)

    logging.info(f"--- ALL DATASETS PROCESSED ---")
    ratio = compressor.compression_stats.get('compression_ratio', 0)
    logging.info(f"New compressed datasets written to: {FLAGS.output_data_dir}")
    if ratio > 0:
        logging.info(f"Memory reduction: ~{ratio:.1f}x smaller than original VGGT tokens")

if __name__ == "__main__":
    app.run(main)