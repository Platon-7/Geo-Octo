# The FINAL, CORRECTED, and WORKING preprocess_vggt.py
import os
import numpy as np
import tensorflow as tf
import torch
import tqdm
from absl import app, flags, logging
from ml_collections import config_flags
import tensorflow_datasets as tfds

from vggt.models.vggt import VGGT

# VGGT inference function is correct and unchanged.
VGGT_MODEL_STATE = {"model": None, "device": None, "dtype": None, "captured_output": None}
def run_vggt_inference(image_horizon_tensor: tf.Tensor) -> np.ndarray:
    if VGGT_MODEL_STATE["model"] is None:
        logging.info("Initializing VGGT model and hook...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        VGGT_MODEL_STATE.update({"device": device, "dtype": dtype})
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        model.eval()
        VGGT_MODEL_STATE["model"] = model
        def hook_fn(module, input, output):
            VGGT_MODEL_STATE["captured_output"] = output[0][0]
        model.aggregator.register_forward_hook(hook_fn)
        logging.info("VGGT model loaded.")
    model, device, dtype = VGGT_MODEL_STATE["model"], VGGT_MODEL_STATE["device"], VGGT_MODEL_STATE["dtype"]
    images_np = image_horizon_tensor.numpy()
    b, h, H, W, C = images_np.shape
    images_reshaped = images_np.reshape(b * h, H, W, C).transpose(0, 3, 1, 2)
    images_torch = torch.from_numpy(images_reshaped).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        model(images_torch)
    captured_tensor = VGGT_MODEL_STATE["captured_output"]
    vggt_tokens = captured_tensor.reshape(b, h, 261, 2048)
    return vggt_tokens.to(torch.float32).cpu().numpy()

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "scripts/configs/preprocess_libero_config.py")
flags.DEFINE_string("data_dir", None, "Path to the root directory containing TFDS datasets.", required=True)
flags.DEFINE_integer("vggt_batch_size", 32, "Batch size for running VGGT inference on windows.")

def main(_):
    logging.set_verbosity(logging.INFO)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logging.error(e)

    # File discovery logic is correct.
    all_tfrecord_paths = []
    data_dir = FLAGS.data_dir
    dataset_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for name in dataset_names:
        try:
            builder = tfds.builder(name, data_dir=data_dir)
            dataset_files_dir = builder.info.data_dir
            for split_name in builder.info.splits.keys():
                split_filenames = builder.info.splits[split_name].filenames
                absolute_paths = [os.path.join(dataset_files_dir, fname) for fname in split_filenames]
                all_tfrecord_paths.extend(absolute_paths)
        except Exception as e:
            logging.warning(f"Could not process directory '{name}' as a TFDS dataset. Error: {e}")

    logging.info(f"SUCCESS: Found {len(all_tfrecord_paths)} tfrecord files. Starting processing.")

    resize_size = tuple(FLAGS.config.resize_size)
    window_size = FLAGS.config.window_size

    def safe_parse_and_decode(raw_record_tensor):
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_record_tensor.numpy())
            if 'steps/observation/image' in example.features.feature:
                image_bytes = example.features.feature['steps/observation/image'].bytes_list.value[0]
                image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
                image_float = tf.image.resize(image, resize_size)
                # Cast the float32 image back to uint8 to match the Tout declaration.
                image_uint8 = tf.cast(image_float, tf.uint8)
                image_uint8.set_shape([*resize_size, 3])
                return image_uint8, True
            else:
                return tf.zeros([*resize_size, 3], dtype=tf.uint8), False
        except Exception:
            return tf.zeros([*resize_size, 3], dtype=tf.uint8), False

    def tf_safe_parser(raw_record_tensor):
        image, is_valid = tf.py_function(
            safe_parse_and_decode,
            inp=[raw_record_tensor],
            Tout=[tf.uint8, tf.bool]
        )
        image.set_shape([*resize_size, 3])
        is_valid.set_shape([])
        return image, is_valid

    for filepath in tqdm.tqdm(all_tfrecord_paths, desc="Processing TFRecord Files"):
        try:
            raw_dataset = tf.data.TFRecordDataset([filepath])
            parsed_dataset = raw_dataset.map(tf_safe_parser, num_parallel_calls=tf.data.AUTOTUNE)
            filtered_dataset = parsed_dataset.filter(lambda image, is_valid: is_valid)
            clean_image_dataset = filtered_dataset.map(lambda image, is_valid: image)
            images_in_episode = list(clean_image_dataset)

            if len(images_in_episode) < window_size:
                logging.warning(f"Skipping {filepath}, not enough valid image steps ({len(images_in_episode)}) for a window of size {window_size}.")
                continue
            
            images_tensor = tf.stack(images_in_episode)
            image_windows = tf.signal.frame(images_tensor, frame_length=window_size, frame_step=1, axis=0)
            # Cast to float32 right before inference.
            image_windows = tf.cast(image_windows, tf.float32)

            window_dataset = tf.data.Dataset.from_tensor_slices(image_windows)
            batched_window_dataset = window_dataset.batch(FLAGS.vggt_batch_size).prefetch(tf.data.AUTOTUNE)

            all_vggt_tokens = []
            for image_batch in batched_window_dataset:
                vggt_tokens_batch = run_vggt_inference(image_batch)
                all_vggt_tokens.append(vggt_tokens_batch)
            if not all_vggt_tokens:
                continue
            all_vggt_tokens = np.concatenate(all_vggt_tokens, axis=0)
            for i in range(all_vggt_tokens.shape[0]):
                vggt_tokens = all_vggt_tokens[i]
                start_timestep = i
                # Get the original tfrecord filename without extension
                base_filename = os.path.basename(filepath)
                
                tokens_dir = os.path.join(os.path.dirname(filepath), "vggt_tokens")
                tf.io.gfile.makedirs(tokens_dir)
                
                # The output name is now built correctly.
                output_filename = f"{base_filename}.step_{start_timestep:06d}.vggt_tokens.npy"
                output_path = os.path.join(tokens_dir, output_filename)
                if tf.io.gfile.exists(output_path):
                    continue
                with tf.io.gfile.GFile(output_path, "wb") as f:
                    np.save(f, vggt_tokens)
        except Exception as e:
            logging.error(f"A critical error occurred while processing file {filepath}: {e}")
            
    logging.info(f"--- FINISHED PROCESSING ---")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    app.run(main)