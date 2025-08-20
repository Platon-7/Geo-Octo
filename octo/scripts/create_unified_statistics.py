import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
import os
import glob
import argparse

# --- Configuration ---
# BASE_DATA_DIR = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_compressed"
# DATASET_NAMES = [
#     "libero_object_vggt_compressed",
#     "libero_spatial_vggt_compressed",
#     "libero_goal_vggt_compressed",
#     "libero_10_vggt_compressed",
# ]

# --- CLI-configured; see argparse in main() ---

# TFRecord file discovery happens in main() based on CLI args

# Keys for parsing the TFRecord files - FIXED to use 'state' instead of 'joint_state'
ACTION_KEY = 'steps/action'
PROP_KEY = 'steps/observation/state'  # Use 'state' not 'joint_state'

FEATURE_DESCRIPTION = {
    ACTION_KEY: tf.io.VarLenFeature(tf.float32),
    PROP_KEY: tf.io.VarLenFeature(tf.float32),
}

# Define the output directory and filename
# Output path configured via CLI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-data-dir', required=True, type=str)
    parser.add_argument('--dataset-names', required=True, nargs='+', type=str, help='One or more dataset directory names under base-dir')
    parser.add_argument('--output-dir', required=True, type=str)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-name', type=str, help='Output filename (e.g., stats.json)')
    group.add_argument('--output-file', type=str, help='Full output file path')
    args = parser.parse_args()

    base_data_dir = args.base_data_dir
    dataset_names = args.dataset_names
    output_dir = args.output_dir
    output_stats_file = args.output_file if args.output_file else os.path.join(output_dir, args.output_name)
    os.makedirs(output_dir, exist_ok=True)

    tfrecord_files = []
    for name in dataset_names:
        path_pattern = os.path.join(base_data_dir, name, '*', '*.tfrecord*')
        tfrecord_files.extend(glob.glob(path_pattern))

    if not tfrecord_files:
        raise RuntimeError(f"No TFRecord files found! Path tried: {os.path.join(base_data_dir, dataset_names[0], '*', '*.tfrecord*')}")

    print(f"Found {len(tfrecord_files)} TFRecord files to process.")
    print(f"Datasets: {dataset_names} base: {base_data_dir}")

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

    action_dim = 7
    raw_proprio_dim = 8  # Raw state is 8D
    final_proprio_dim = 7  # Will be sliced to 7D to match standardization
    
    action_stats = {'count': 0, 'mean': np.zeros(action_dim), 'M2': np.zeros(action_dim), 'min': np.full(action_dim, np.inf), 'max': np.full(action_dim, -np.inf)}
    prop_stats = {'count': 0, 'mean': np.zeros(final_proprio_dim), 'M2': np.zeros(final_proprio_dim), 'min': np.full(final_proprio_dim, np.inf), 'max': np.full(final_proprio_dim, -np.inf)}

    # Initialize counters for trajectories and transitions
    num_trajectories = 0
    num_transitions = 0

    def update_stats(existing_stats, new_value):
        existing_stats['count'] += 1
        delta = new_value - existing_stats['mean']
        existing_stats['mean'] += delta / existing_stats['count']
        delta2 = new_value - existing_stats['mean']
        existing_stats['M2'] += delta * delta2
        existing_stats['min'] = np.minimum(existing_stats['min'], new_value)
        existing_stats['max'] = np.maximum(existing_stats['max'], new_value)

    print("Iterating through dataset to compute statistics... (This may take a while)")
    for raw_record in tqdm(raw_dataset):
        example = tf.io.parse_single_example(raw_record, FEATURE_DESCRIPTION)
        
        actions = tf.reshape(tf.sparse.to_dense(example[ACTION_KEY]), [-1, action_dim])
        proprios = tf.reshape(tf.sparse.to_dense(example[PROP_KEY]), [-1, raw_proprio_dim])
        
        # Slice proprios from 8D to 7D to match standardization function
        proprios = proprios[:, :7]

        # Update counters
        num_trajectories += 1
        num_transitions += actions.shape[0]
        
        for action_step in actions:
            update_stats(action_stats, action_step.numpy())
        for prop_step in proprios:
            update_stats(prop_stats, prop_step.numpy())
            
    print("Finalizing statistics...")
    action_stats['std'] = np.sqrt(action_stats['M2'] / action_stats['count'])
    prop_stats['std'] = np.sqrt(prop_stats['M2'] / prop_stats['count'])

    del action_stats['M2']
    del prop_stats['M2']
    
    # Need to compute quantiles from the data
    print("Computing quantiles...")
    
    # We need to iterate through the dataset again to collect all data for quantile computation
    all_actions = []
    all_proprios = []
    
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    for raw_record in tqdm(raw_dataset):
        example = tf.io.parse_single_example(raw_record, FEATURE_DESCRIPTION)
        
        actions = tf.reshape(tf.sparse.to_dense(example[ACTION_KEY]), [-1, action_dim])
        proprios = tf.reshape(tf.sparse.to_dense(example[PROP_KEY]), [-1, raw_proprio_dim])
        
        # Slice proprios from 8D to 7D to match standardization function
        proprios = proprios[:, :7]
        
        all_actions.append(actions.numpy())
        all_proprios.append(proprios.numpy())
    
    # Concatenate all data
    all_actions = np.concatenate(all_actions, axis=0)
    all_proprios = np.concatenate(all_proprios, axis=0)
    
    # Compute quantiles
    action_p99 = np.quantile(all_actions, 0.99, 0)
    action_p01 = np.quantile(all_actions, 0.01, 0)
    proprio_p99 = np.quantile(all_proprios, 0.99, 0)
    proprio_p01 = np.quantile(all_proprios, 0.01, 0)

    final_statistics = {
        'action': {
            'mean': action_stats['mean'].tolist(),
            'std': action_stats['std'].tolist(),
            'max': action_stats['max'].tolist(),
            'min': action_stats['min'].tolist(),
            'p99': action_p99.tolist(),
            'p01': action_p01.tolist(),
        },
        'proprio': {
            'mean': prop_stats['mean'].tolist(),
            'std': prop_stats['std'].tolist(),
            'max': prop_stats['max'].tolist(),
            'min': prop_stats['min'].tolist(),
            'p99': proprio_p99.tolist(),
            'p01': proprio_p01.tolist(),
        },
        'num_transitions': num_transitions,
        'num_trajectories': num_trajectories,
    }

    with open(output_stats_file, 'w') as f:
        json.dump(final_statistics, f, indent=4)
        
    print(f"\nDone! Unified statistics saved to:\n{output_stats_file}")

if __name__ == "__main__":
    main()