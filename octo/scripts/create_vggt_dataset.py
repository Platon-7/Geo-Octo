# scripts/create_vggt_tfrecords.py (v18 - Final Corrected Builder)
import os
import tensorflow as tf
import torch
import tqdm
from absl import app, flags, logging
import numpy as np
import tensorflow_datasets as tfds
from collections import defaultdict
from typing import Dict, Any

from vggt.models.vggt import VGGT

# Configure TF to not grab all GPU memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Self-contained episode processing function (Unchanged) ---
def get_vggt_tokens_for_episode(model, device, dtype, all_images_np: np.ndarray, batch_size: int) -> np.ndarray:
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
        all_tokens_list.append(tokens_batch.to(torch.float16).cpu().numpy())
    if not all_tokens_list:
        return np.array([])
    return np.concatenate(all_tokens_list, axis=0)

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


# --- The New TFDS Builder Class ---

class VggtDataset(tfds.core.GeneratorBasedBuilder):
    """A TFDS builder for adding VGGT tokens to an existing robot dataset."""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release with VGGT tokens.'}

    def __init__(self, original_builder, vggt_model, vggt_device, vggt_dtype, vggt_batch_size, **kwargs):
        self._original_builder = original_builder
        self._vggt_model = vggt_model
        self._vggt_device = vggt_device
        self._vggt_dtype = vggt_dtype
        self._vggt_batch_size = vggt_batch_size
        self.name = f"{self._original_builder.name}_vggt"
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Defines the new dataset structure, adding the `vggt_tokens` field."""
        original_info = self._original_builder.info

        step_features = dict(original_info.features['steps'].feature)
        observation_features = dict(step_features['observation'])
        
        observation_features['vggt_tokens'] = tfds.features.Tensor(
            shape=(261, 2048),
            dtype=np.float16,
            doc='Pre-computed VGGT tokens from the primary image.',
        )
        
        step_features['observation'] = tfds.features.FeaturesDict(observation_features)
        
        final_features = tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset(tfds.features.FeaturesDict(step_features)),
            'episode_metadata': original_info.features['episode_metadata']
        })
        
        return tfds.core.DatasetInfo(
            builder=self,
            description="A version of the Libero dataset with pre-computed VGGT tokens.",
            features=final_features,
            homepage="https://github.com/geopavlo/geo-octo",
            citation=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Specifies the splits to generate."""
        return {'train': self._generate_examples(split='train')}

    def _generate_examples(self, split: str):
        """Reads the original dataset, computes tokens, and yields new examples."""
        ds = self._original_builder.as_dataset(split=split)
        num_episodes = self._original_builder.info.splits[split].num_examples

        for i, episode in enumerate(tqdm.tqdm(ds, total=num_episodes, desc=f"Generating {self.name}")):
            steps_list_of_dicts = list(tfds.as_numpy(episode['steps']))
            if not steps_list_of_dicts:
                continue
            
            steps = transpose_list_of_dicts(steps_list_of_dicts)
            
            primary_image_key = 'image' 
            if primary_image_key not in steps['observation'] or len(steps['observation'][primary_image_key]) == 0:
                continue

            images_np = steps['observation'][primary_image_key]
            
            all_vggt_tokens = get_vggt_tokens_for_episode(
                self._vggt_model, self._vggt_device, self._vggt_dtype, images_np, self._vggt_batch_size
            )

            if all_vggt_tokens.size == 0 or len(all_vggt_tokens) != len(steps['action']):
                logging.warning(f"Token generation failed or length mismatch for episode {i}. Skipping.")
                continue

            new_steps = []
            for t in range(len(steps['action'])):
                new_step = steps_list_of_dicts[t]
                new_step['observation']['vggt_tokens'] = all_vggt_tokens[t].astype(np.float16)
                new_steps.append(new_step)
            
            yield i, {'steps': new_steps, 'episode_metadata': tfds.as_numpy(episode['episode_metadata'])}


# --- Flags and Main Script ---
FLAGS = flags.FLAGS
flags.DEFINE_string("input_data_dir", None, "Path to the root directory containing the ORIGINAL sub-datasets.", required=True)
flags.DEFINE_string("output_data_dir", None, "Path where the NEW TFDS datasets will be written.", required=True)
flags.DEFINE_integer("vggt_batch_size", 32, "Batch size for running VGGT inference.")
flags.DEFINE_bool("overwrite", False, "Whether to overwrite existing datasets.")

def main(_):
    logging.set_verbosity(logging.INFO)
    
    logging.info("Initializing VGGT model...")
    vggt_device = "cuda" if torch.cuda.is_available() else "cpu"
    vggt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(vggt_device).eval()
    logging.info(f"VGGT model loaded to {vggt_device}. Using {vggt_dtype} for inference.")

    input_root = FLAGS.input_data_dir
    output_root = FLAGS.output_data_dir
    dataset_names = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    logging.info(f"Found {len(dataset_names)} original sub-datasets to process: {dataset_names}")

    for dataset_name in dataset_names:
        logging.info(f"###### PROCESSING DATASET: {dataset_name} ######")
        try:
            original_builder = tfds.builder(dataset_name, data_dir=input_root)
            
            vggt_builder = VggtDataset(
                original_builder=original_builder,
                vggt_model=vggt_model,
                vggt_device=vggt_device,
                vggt_dtype=vggt_dtype,
                vggt_batch_size=FLAGS.vggt_batch_size,
                data_dir=output_root,
            )
            
            # --- THIS IS THE CORRECTED LOGIC ---
            # Manually handle the overwrite logic before calling download_and_prepare
            if FLAGS.overwrite and tf.io.gfile.exists(vggt_builder.data_dir):
                logging.warning(f"Overwriting existing dataset at {vggt_builder.data_dir}")
                tf.io.gfile.rmtree(vggt_builder.data_dir)
            
            # Since all data is local, no special DownloadConfig is needed.
            vggt_builder.download_and_prepare()
            # --- END OF CORRECTION ---

            logging.info(f"Successfully created TFDS dataset '{vggt_builder.name}' at '{output_root}'.")

        except Exception as e:
            logging.error(f"Failed to process dataset {dataset_name}. Error: {e}", exc_info=True)

    logging.info(f"--- ALL DATASETS PROCESSED ---")
    logging.info(f"New TFDS-compatible datasets with VGGT tokens written to: {FLAGS.output_data_dir}")

if __name__ == "__main__":
    app.run(main)