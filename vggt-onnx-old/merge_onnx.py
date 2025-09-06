import os
from shutil import rmtree

import onnx

for in_dir, out_name in [("onnx", "vggt.onnx"), ("onnx_fp16", "vggt_fp16.onnx")]:
    model = onnx.load(os.path.join(in_dir, out_name), load_external_data=True)
    rmtree(in_dir)

    onnx.save_model(
        model,
        out_name,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{out_name}_data",
        size_threshold=0,
    )
