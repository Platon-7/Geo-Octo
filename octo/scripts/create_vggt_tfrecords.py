# scripts/create_vggt_tfrecords.py (v14 - The definitive version)
import os
import tensorflow as tf
import torch
import tqdm
from absl import app, flags, logging
import numpy as np
import tensorflow_datasets as tfds
from collections import defaultdict

from vggt.models.vggt import VGGT

# Configure TF to not grab all GPU memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Self-contained episode processing function ---
def get_vggt_tokens_for_episode(model, device, dtype, all_images_np: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Takes all images for an episode and returns all VGGT tokens.
    This function handles its own resizing, normalization, batching, and concatenation internally.
    """
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
        
        # Use the `dtype` (bfloat16 on your A100) for fast GPU inference
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            model(images_torch)
        
        hook_handle.remove()

        if not captured_output_list:
            raise RuntimeError("VGGT hook failed to capture output!")
        
        captured_tensor = captured_output_list[0]
        tokens_batch = captured_tensor.squeeze(1)
        
        # Convert to standard float16 for saving to disk (halves file size)
        all_tokens_list.append(tokens_batch.to(torch.float16).cpu().numpy())
    
    if not all_tokens_list:
        return np.array([])
    
    return np.concatenate(all_tokens_list, axis=0)

# --- Helper Functions (Unchanged) ---
def _bytes_feature(value):
    if isinstance(value, np.bool_): value = b'\x01' if value else b'\x00'
    elif isinstance(value, type(tf.constant(0))): value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    # This now correctly handles float16 input
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

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

# --- Flags and Main Script ---
FLAGS = flags.FLAGS
flags.DEFINE_string("input_data_dir", None, "Path to the root directory containing the sub-datasets.", required=True)
flags.DEFINE_string("output_data_dir", None, "Path where the new TFDS datasets will be written.", required=True)
flags.DEFINE_integer("vggt_batch_size", 32, "Batch size for running VGGT inference.")
flags.DEFINE_bool("overwrite", False, "Whether to overwrite existing files.")

def main(_):
    logging.set_verbosity(logging.INFO)
    
    logging.info("Initializing VGGT model...")
    vggt_device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use the logic you found to set the BEST dtype for inference
    vggt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(vggt_device)
    vggt_model.eval()
    logging.info(f"VGGT model loaded to {vggt_device}. Using {vggt_dtype} for inference.")

    input_root = FLAGS.input_data_dir
    output_root = FLAGS.output_data_dir
    output_dir_name = os.path.basename(os.path.normpath(output_root))
    dataset_names = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d)) and d != output_dir_name]
    logging.info(f"Found {len(dataset_names)} sub-datasets to process: {dataset_names}")

    # For testing, you can uncomment this to process only one dataset
    # dataset_names = ['libero_spatial_no_noops']

    for dataset_name in dataset_names:
        logging.info(f"###### PROCESSING DATASET: {dataset_name} ######")
        builder = tfds.builder(dataset_name, data_dir=input_root)
        ds = builder.as_dataset(split='train')
        new_dataset_version = str(builder.info.version) + "_vggt"
        output_path = os.path.join(output_root, dataset_name, new_dataset_version)
        if not FLAGS.overwrite and os.path.exists(output_path):
            logging.info(f"Output path {output_path} already exists. Skipping.")
            continue
        os.makedirs(output_path, exist_ok=True)

        num_episodes = builder.info.splits['train'].num_examples
        for i, episode in enumerate(tqdm.tqdm(ds, total=num_episodes, desc=f"Episodes in {dataset_name}")):
            
            steps_ds = episode['steps']
            steps_list_of_dicts = list(tfds.as_numpy(steps_ds))
            if not steps_list_of_dicts: continue
            steps = transpose_list_of_dicts(steps_list_of_dicts)
            if 'image' not in steps['observation'] or len(steps['observation']['image']) == 0: continue
            
            images_np = steps['observation']['image']
            
            all_vggt_tokens = get_vggt_tokens_for_episode(
                vggt_model, vggt_device, vggt_dtype, images_np, FLAGS.vggt_batch_size
            )

            if all_vggt_tokens.size == 0 or len(all_vggt_tokens) != len(steps['action']):
                logging.warning(f"Token generation failed or length mismatch for episode {i} in {dataset_name}. Skipping.")
                continue

            output_filename = os.path.join(output_path, f"{dataset_name}-train.tfrecord-{i:05d}-of-{num_episodes:05d}")
            with tf.io.TFRecordWriter(output_filename) as writer:
                num_steps = len(steps['action'])
                for t in range(num_steps):
                    feature = {
                        'observation/image': _bytes_feature(tf.io.encode_jpeg(steps['observation']['image'][t])),
                        'observation/state': _float_feature(steps['observation']['state'][t]),
                        'action': _float_feature(steps['action'][t]),
                        'discount': _float_feature(np.array([steps['discount'][t]], dtype=np.float32)),
                        'reward': _float_feature(np.array([steps['reward'][t]], dtype=np.float32)),
                        'is_first': _bytes_feature(steps['is_first'][t]),
                        'is_last': _bytes_feature(steps['is_last'][t]),
                        'is_terminal': _bytes_feature(steps['is_terminal'][t]),
                        'observation/vggt_tokens': _float_feature(all_vggt_tokens[t]),
                    }
                    if 'language_instruction' in steps['observation']:
                         feature['observation/language_instruction'] = _bytes_feature(steps['observation']['language_instruction'][t])
                    
                    writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

        original_info_dir = os.path.join(input_root, dataset_name, str(builder.info.version))
        for f_name in os.listdir(original_info_dir):
            if f_name.endswith(".json"):
                tf.io.gfile.copy(os.path.join(original_info_dir, f_name), os.path.join(output_path, f_name), overwrite=True)
        logging.info(f"Finished processing {dataset_name}. Metadata copied.")

    logging.info(f"--- ALL DATASETS PROCESSED ---")
    logging.info(f"New datasets with VGGT tokens written to: {FLAGS.output_data_dir}")

if __name__ == "__main__":
    app.run(main)