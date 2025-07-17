# octo/utils/offline_dataset_transforms.py (Corrected)
import tensorflow as tf
from typing import Dict

VGGT_TOKEN_SHAPE = [261, 2048]

def decode_vggt_tfrecord(record: tf.Tensor) -> Dict[str, tf.Tensor]:
    """
    Decodes a single TFRecord example using the correct flat keys.
    """
    feature_description = {
        'observation/image': tf.io.FixedLenFeature([], tf.string),
        'observation/state': tf.io.FixedLenFeature([8], tf.float32),
        'observation/vggt_tokens': tf.io.FixedLenFeature([261 * 2048], tf.float32),
        'action': tf.io.FixedLenFeature([7], tf.float32),
        'discount': tf.io.FixedLenFeature([1], tf.float32),
        'reward': tf.io.FixedLenFeature([1], tf.float32),
        'is_first': tf.io.FixedLenFeature([], tf.string),
        'is_last': tf.io.FixedLenFeature([], tf.string),
        'is_terminal': tf.io.FixedLenFeature([], tf.string),
        'observation/language_instruction': tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    # 'example' is a FLAT dictionary with keys like 'observation/state'
    example = tf.io.parse_single_example(record, feature_description)

    vggt_tokens = tf.reshape(example['observation/vggt_tokens'], VGGT_TOKEN_SHAPE)
    
    image = tf.io.decode_jpeg(example['observation/image'])
    image = tf.image.resize(image, [224, 224], method=tf.image.ResizeMethod.BICUBIC)
    image = tf.cast(image, tf.uint8)

    ### THE FIX IS HERE ###
    # We now correctly access the FLAT keys from the 'example' dictionary
    # to build the desired NESTED output dictionary.
    return {
        'observation': {
            'image_primary': image,
            'proprio': example['observation/state'], # CORRECTED
            'vggt_tokens': vggt_tokens,
        },
        'task': {
            'language_instruction': example['observation/language_instruction'], # CORRECTED
        },
        'action': example['action'],
        'reward': example['reward'][0],
        'discount': example['discount'][0],
        'is_first': tf.equal(example['is_first'], b'\x01'),
        'is_last': tf.equal(example['is_last'], b'\x01'),
        'is_terminal': tf.equal(example['is_terminal'], b'\x01'),
    }