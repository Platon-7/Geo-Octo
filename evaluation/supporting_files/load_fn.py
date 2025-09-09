# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modifications by Adrian Kretz <me@akretz.com>

from PIL import Image
import numpy as np


def load_and_preprocess_images(images_or_paths, target_size=224):
    """
    Load and preprocess images for VGGT input, matching the offline pipeline.

    Accepts either:
      - a list of file paths (strings/Paths), or
      - a list of in-memory NumPy arrays with shape (H, W, C)

    Args:
        images_or_paths (list): List of file paths or HWC NumPy arrays

    Returns:
        numpy.ndarray: Batched array of preprocessed images with shape (N, 3, target_size, target_size)

    Raises:
        ValueError: If the input list is empty

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
    """
    # Check for empty list
    if len(images_or_paths) == 0:
        raise ValueError("At least 1 image is required")

    images = []

    # Process all images
    for item in images_or_paths:
        # Open image from path or NumPy
        if isinstance(item, str):
            img = Image.open(item)
        elif hasattr(item, "__fspath__"):
            img = Image.open(item.__fspath__())
        elif isinstance(item, np.ndarray):
            # Expect HWC; if CHW, transpose
            arr = item
            if arr.ndim != 3:
                raise ValueError(f"Expected image array with 3 dims (HWC), got shape {arr.shape}")
            if arr.shape[-1] in (1, 3, 4):
                # HWC
                img = Image.fromarray(arr)
            elif arr.shape[0] in (1, 3, 4):
                # CHW -> HWC
                img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
            else:
                raise ValueError(f"Unrecognized image array shape {arr.shape}; expected HWC/CHW with C in (1,3,4)")
        else:
            raise ValueError("Each item must be a file path or a NumPy array")

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        # Make the largest dimension `target_size` while maintaining aspect ratio
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0   # Convert to numpy array (0, 1)
        img_array = np.transpose(img_array, (2, 0, 1))        # Convert from HWC to CHW format

        # Pad to make a square of target_size x target_size
        h_padding = target_size - img_array.shape[1]
        w_padding = target_size - img_array.shape[2]

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            # Pad with white (value=1.0)
            img_array = np.pad(
                img_array,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=1.0
            )

        images.append(img_array)

    images = np.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(images_or_paths) == 1:
        # Verify shape is (1, C, H, W)
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)

    return images
