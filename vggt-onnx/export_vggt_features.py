import os
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.models.feature_export import VGGTAllLayersFeatures, VGGTSelectedLayersFeatures


def main() -> None:
    os.makedirs("onnx_fp16_features", exist_ok=True)

    device = "cpu"
    base = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    image_names = [os.path.join("vggt", "examples", "kitchen", "images", f"{i:02}.png") for i in [0, 1]]
    images = load_and_preprocess_images(image_names, "pad").to(device)

    input_names = ["input_images"]

    # Case 1: all 24 layers -> output [24, 261, 2048]
    model_all = VGGTAllLayersFeatures(base, num_patch_tokens=261).to(device)
    output_names_all = ["all_layer_features"]

    # Case 2: selected layers [3, 10, 16, 22] (0-based) -> [4, 261, 2048]
    selected_indices = [3, 10, 16, 22]
    model_sel = VGGTSelectedLayersFeatures(base, selected_indices, num_patch_tokens=261).to(device)
    output_names_sel = ["selected_layer_features"]

    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.float16):
            # Export all-layers
            torch.onnx.export(
                model_all,
                images,
                "onnx_fp16_features/vggt_all_layers_fp16.onnx",
                input_names=input_names,
                output_names=output_names_all,
                dynamic_axes={
                    "input_images": {0: "num_images"},
                },
            )

            # Export selected-layers
            torch.onnx.export(
                model_sel,
                images,
                "onnx_fp16_features/vggt_selected_layers_fp16.onnx",
                input_names=input_names,
                output_names=output_names_sel,
                dynamic_axes={
                    "input_images": {0: "num_images"},
                },
            )


if __name__ == "__main__":
    main()

