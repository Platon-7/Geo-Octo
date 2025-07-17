import os
import numpy as np
import tensorflow as tf

from octo.data.oxe.oxe_standardization_transforms import bridge_dataset_transform

def load_vggt_and_standardize(trajectory: dict) -> dict:
    # First, apply the original standardization transform
    trajectory = bridge_dataset_transform(trajectory)

    # Get the source path to find our corresponding .npy file
    source_path = trajectory["dataset_name"].numpy().decode('utf-8')
    tokens_dir = os.path.join(os.path.dirname(source_path), "vggt_tokens")
    filename = os.path.basename(source_path).replace(".tfrecord", ".npy")
    tokens_path = os.path.join(tokens_dir, filename)

    # Load the pre-computed tokens using tf.py_function to wrap np.load
    def _load_npy(path):
        with tf.io.gfile.GFile(path, "rb") as f:
            return np.load(f)

    vggt_tokens = tf.py_function(_load_npy, [tokens_path], tf.float32)

    # Ensure the shape is set correctly
    vggt_tokens.set_shape([trajectory["observation"]["image_primary"].shape[0], 261, 2048])
    
    # Add the tokens to the observation dictionary
    trajectory["observation"]["vggt_tokens"] = vggt_tokens
    
    return trajectory