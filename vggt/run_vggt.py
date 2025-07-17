import numpy as np
from PIL import Image
import os
import shutil
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# --- Step 1: Create Toy Images ---
print("--- Creating temporary toy images ---")
# Create a temporary directory to store the images
temp_dir = "temp_vggt_images"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# List to hold the paths of the created images
image_paths = []

for i in range(3):
    # Generate a random 256x256 RGB image
    random_image_array = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(random_image_array)
    
    # Define the path and save the image
    image_path = os.path.join(temp_dir, f"toy_image_{i}.png")
    image.save(image_path)
    image_paths.append(image_path)
    print(f"Saved: {image_path}")

print("--- Toy images created successfully ---\n")


# --- Step 2: Run the Original VGGT Code ---
print("--- Running VGGT inference ---")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Check for bfloat16 support, otherwise fallback to float16 or float32
if device == "cuda":
    try:
        # Check for Ampere or newer architecture (Compute Capability 8.0+)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    except Exception:
        # Fallback if there's an issue getting device capability (e.g., no GPU detected by PyTorch)
        dtype = torch.float32
else:
    # CPU does not support bfloat16 or float16 for this operation
    dtype = torch.float32

print(f"Using device: {device}, dtype: {dtype}")

# Initialize the model and load the pretrained weights.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess the toy images we just created
images = load_and_preprocess_images(image_paths).to(device)

with torch.no_grad():
    # Use autocast for mixed-precision inference if on GPU
    with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

print("--- VGGT inference completed successfully ---")

# Inspect the predictions
# 'predictions' is a dictionary containing the model's 3D outputs
print("\nPredicted attributes (keys):", predictions.keys())
print("Shape of predicted depth map for first image:", predictions['depth'][0].shape)
print("Shape of predicted camera pose encoding for first image:", predictions['pose_enc'][0].shape)
print("\nTest complete!")


# --- Step 3: Clean up the temporary images ---
print("\n--- Cleaning up temporary files ---")
shutil.rmtree(temp_dir)
print(f"Removed directory: {temp_dir}")