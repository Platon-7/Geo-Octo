import os

import numpy as np
import onnxruntime as ort
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def assert_similar(a, b, delta=1e-3):
    assert a.shape == b.shape
    diff_99p = np.percentile(np.abs(a - b), 99.0)
    print(f"    {diff_99p}")
    assert diff_99p < delta


def compare_torch_onnx(torch_pred, onnx_pred, ort_sess, delta, conf_delta):
    for output_name in torch_pred:
        print(f"    Checking for similar {output_name}")
        idx = [x.name for x in ort_sess._outputs_meta].index(output_name)
        assert_similar(
            torch_pred[output_name],
            onnx_pred[idx],
            conf_delta if output_name.endswith("_conf") else delta,
        )


MAX_NUM_IMAGES = 3

image_names = [
    os.path.join("vggt", "examples", "kitchen", "images", f"{i:02}.png")
    for i in range(MAX_NUM_IMAGES)
]
images = load_and_preprocess_images(image_names, "pad")

print("Loading PyTorch model")
model = VGGT.from_pretrained("facebook/VGGT-1B")
expected = []
with torch.no_grad():
    for num_images in range(1, MAX_NUM_IMAGES + 1):
        input_images = images[:num_images]
        print(f"  Running on {num_images} input images")
        expected.append(model(input_images))
del model

for onnx_model, tol, conf_tol in [("vggt.onnx", 1e-4, 1e-3), ("vggt_fp16.onnx", 1e-2, 2e-1)]:
    print(f"Loading ONNX model {onnx_model}")
    ort_sess = ort.InferenceSession(onnx_model)
    for num_images in range(1, MAX_NUM_IMAGES + 1):
        input_images = images[:num_images]
        print(f"  Running on {num_images} input images")
        output = ort_sess.run(None, {"input_images": input_images.numpy()})
        compare_torch_onnx(expected[num_images - 1], output, ort_sess, tol, conf_tol)
    del ort_sess
