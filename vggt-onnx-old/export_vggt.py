import os

import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


# Create the output directories if they don't exist
os.makedirs("onnx", exist_ok=True)
os.makedirs("onnx_fp16", exist_ok=True)
# ---------------------

device = "cpu"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
image_names = [os.path.join("vggt", "examples", "kitchen", "images", f"{i:02}.png") for i in [0, 1]]
images = load_and_preprocess_images(image_names, "pad").to(device)

input_names = ["input_images"]
output_names = [
    "pose_enc",
    "depth",
    "depth_conf",
    "world_points",
    "world_points_conf",
    "images",
    "aggregated_tokens",
]
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    with torch.no_grad():
        print("\nExporting FP16 ONNX model...")
        with torch.amp.autocast(device, dtype=torch.float16):
            torch.onnx.export(
                model,
                images,
                "onnx_fp16/vggt_fp16.onnx",
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    name: {0: "num_images"} for name in input_names + output_names
                },
                opset_version=17,
            )
        print("FP16 ONNX model exported to onnx_fp16/vggt_fp16.onnx")
