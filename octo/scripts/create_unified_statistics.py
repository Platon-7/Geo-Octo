import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
import os
import glob

# --- Configuration ---
BASE_DATA_DIR = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2"
DATASET_NAMES = [
    "libero_object_vggt",
    "libero_spatial_vggt",
    "libero_goal_vggt",
    "liber_o10_vggt",
]

TFRECORD_FILES = []
for name in DATASET_NAMES:
    path_pattern = os.path.join(BASE_DATA_DIR, name, '*', '*.tfrecord*')
    TFRECORD_FILES.extend(glob.glob(path_pattern))

if not TFRECORD_FILES:
    raise RuntimeError(f"No TFRecord files found! Path tried: {os.path.join(BASE_DATA_DIR, DATASET_NAMES[0], '*', '*.tfrecord*')}")

print(f"Found {len(TFRECORD_FILES)} TFRecord files to process.")


# Keys for parsing the TFRecord files
ACTION_KEY = 'steps/action'
PROP_KEY = 'steps/observation/joint_state'

FEATURE_DESCRIPTION = {
    ACTION_KEY: tf.io.VarLenFeature(tf.float32),
    PROP_KEY: tf.io.VarLenFeature(tf.float32),
}

# Define the output directory and filename
OUTPUT_DIR = "/home/pkarageorgis/geo_octo/libero_datasets/unified_stats"
OUTPUT_STATS_FILE = os.path.join(OUTPUT_DIR, "unified_dataset_statistics.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    raw_dataset = tf.data.TFRecordDataset(TFRECORD_FILES)

    action_dim = 7
    proprio_dim = 7
    
    action_stats = {'count': 0, 'mean': np.zeros(action_dim), 'M2': np.zeros(action_dim), 'min': np.full(action_dim, np.inf), 'max': np.full(action_dim, -np.inf)}
    prop_stats = {'count': 0, 'mean': np.zeros(proprio_dim), 'M2': np.zeros(proprio_dim), 'min': np.full(proprio_dim, np.inf), 'max': np.full(proprio_dim, -np.inf)}

    # ADDED: Initialize counters for trajectories and transitions
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
        proprios = tf.reshape(tf.sparse.to_dense(example[PROP_KEY]), [-1, proprio_dim])

        # ADDED: Update counters
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
    
    final_statistics = {
        'action': {key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in action_stats.items()},
        'proprio': {key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in prop_stats.items()},
        'num_transitions': num_transitions,  # ADDED: Save counts to the final JSON
        'num_trajectories': num_trajectories, # ADDED: Save counts to the final JSON
    }

    with open(OUTPUT_STATS_FILE, 'w') as f:
        json.dump(final_statistics, f, indent=4)
        
    print(f"\nDone! Unified statistics saved to:\n{OUTPUT_STATS_FILE}")

if __name__ == "__main__":
    main()