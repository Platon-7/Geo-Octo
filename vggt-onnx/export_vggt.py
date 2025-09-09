import os
import torch
import torch.nn as nn

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

os.makedirs("onnx_fp16", exist_ok=True)

class VggtOnnxWrapper(nn.Module):
    def __init__(self, vggt_model: nn.Module):
        super().__init__()
        self.vggt = vggt_model
        # Fixed, canonical order
        self.output_names = [
            "layer_patch_tokens",     # [B,S,L,N,2048]
            "pose_enc",               # [B,S,9]
            "depth",                  # [B,S,H,W,1]
            "depth_conf",             # [B,S,H,W]
            "world_points",           # [B,S,H,W,3]
            "world_points_conf",      # [B,S,H,W]
            "images",                 # [B,S,3,H,W]
        ]

    def forward(self, images: torch.Tensor):
        pred = self.vggt(images)  # dict, includes "layer_patch_tokens"
        return (
            pred["layer_patch_tokens"],
            pred["pose_enc"],
            pred["depth"],
            pred["depth_conf"],
            pred["world_points"],
            pred["world_points_conf"],
            pred["images"],
        )

device = "cpu"
base = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
wrapped = VggtOnnxWrapper(base).to(device)

image_names = [os.path.join("vggt", "examples", "kitchen", "images", f"{i:02}.png") for i in [0, 1]]
images = load_and_preprocess_images(image_names, "pad").to(device)

input_names = ["input_images"]
output_names = wrapped.output_names

images = images.unsqueeze(0)  # [1, S, 3, H, W]

dynamic_axes = {
  "input_images": {0:"batch_size", 1:"num_frames", 3:"height", 4:"width"},
  "layer_patch_tokens": {0:"batch_size", 1:"num_frames", 3:"num_patches"},  # no num_layers here
  "pose_enc": {0:"batch_size", 1:"num_frames"},
  "depth": {0:"batch_size", 1:"num_frames", 2:"height", 3:"width"},
  "depth_conf": {0:"batch_size", 1:"num_frames", 2:"height", 3:"width"},
  "world_points": {0:"batch_size", 1:"num_frames", 2:"height", 3:"width"},
  "world_points_conf": {0:"batch_size", 1:"num_frames", 2:"height", 3:"width"},
  "images": {0:"batch_size", 1:"num_frames", 3:"height", 4:"width"},
}

with torch.no_grad():
    with torch.amp.autocast(device, dtype=torch.float16):
        torch.onnx.export(
            wrapped,
            images,
            "onnx_fp16/vggt_fp16.onnx",
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )