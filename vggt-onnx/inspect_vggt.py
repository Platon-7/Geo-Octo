import os
import sys
import argparse
from typing import Optional, Tuple, List

import numpy as np

try:
    import onnx
    import onnx.shape_inference
except Exception:
    onnx = None

try:
    import onnxruntime as ort
except Exception as e:
    ort = None


def find_onnx_model(path: str) -> str:
    """Return a concrete .onnx file path from a file or directory path."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        # Prefer common names first
        preferred = [
            "model.onnx",
            "vggt.onnx",
            "vggt_fp16.onnx",
            "onnx_fp16.onnx",
        ]
        for name in preferred:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        # Fallback: first .onnx file in directory (non-recursive, then recursive)
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith(".onnx"):
                return os.path.join(path, fname)
        for root, _dirs, files in os.walk(path):
            for fname in files:
                if fname.lower().endswith(".onnx"):
                    return os.path.join(root, fname)
    raise FileNotFoundError(f"Could not find an ONNX model at '{path}'.")


def dtype_from_onnx_str(t: str):
    if "float16" in t:
        return np.float16
    if "float" in t:
        return np.float32
    if "double" in t:
        return np.float64
    if "int64" in t:
        return np.int64
    if "int32" in t:
        return np.int32
    return np.float32


def pretty_shape(dims: List[Optional[int]]) -> str:
    return "x".join(str(d) if (d is not None and d != 0) else "?" for d in dims)


def get_io_details(session: "ort.InferenceSession") -> Tuple[List[dict], List[dict]]:
    def to_dict(io):
        shape = [d.dim_value if (d.dim_value is not None and d.dim_value != 0) else None for d in io.shape]
        return {
            "name": io.name,
            "type": io.type,
            "shape": shape,
        }

    inputs = [to_dict(i) for i in session.get_inputs()]
    outputs = [to_dict(o) for o in session.get_outputs()]
    return inputs, outputs


def infer_graph_shapes(model_path: str):
    if onnx is None:
        return None
    try:
        model = onnx.load(model_path)
        inferred = onnx.shape_inference.infer_shapes(model)
        return inferred
    except Exception:
        return None


def run_dummy_inference(session: "ort.InferenceSession", input_name: str, input_spec: dict, input_res: int):
    dtype = dtype_from_onnx_str(input_spec["type"]) if "type" in input_spec else np.float32
    shape = input_spec.get("shape", [])
    # Heuristics: expect NCHW, default N=1, C=3, H=W=input_res
    n = shape[0] if (len(shape) >= 1 and isinstance(shape[0], int) and shape[0] > 0) else 1
    c = shape[1] if (len(shape) >= 2 and isinstance(shape[1], int) and shape[1] > 0) else 3
    h = shape[2] if (len(shape) >= 3 and isinstance(shape[2], int) and shape[2] > 0) else input_res
    w = shape[3] if (len(shape) >= 4 and isinstance(shape[3], int) and shape[3] > 0) else input_res
    dummy = np.ones((n, c, h, w), dtype=dtype)
    outputs = session.run(None, {input_name: dummy})
    return outputs


def analyze_output_shape(arr: np.ndarray) -> str:
    shape = list(arr.shape)
    if len(shape) == 2 and shape == [261, 2048]:
        return "Detected single-layer tokens (261, 2048) — previous export style."
    if len(shape) == 3 and shape[1:] == [261, 2048]:
        if shape[0] == 24:
            return "Matches Case 1: (24, 261, 2048) — all layers."
        if shape[0] == 4:
            return "Matches Case 2: (4, 261, 2048) — layers 4,11,17,23."
        return f"3D tokens with unknown layer count {shape[0]} (expected 24 or 4)."
    return f"Unexpected token shape {tuple(shape)}."


def main():
    parser = argparse.ArgumentParser(description="Inspect a VGGT ONNX model for multi-layer token outputs.")
    parser.add_argument("--onnx_path", type=str, default="/home/pkarageorgis/geo_octo/temp/vggt-onnx/onnx_fp16",
                        help="Path to ONNX file or directory containing it.")
    parser.add_argument("--input_res", type=int, default=224, help="Square input resolution for dummy run.")
    parser.add_argument("--use_cuda", type=str, default="auto", choices=["auto", "true", "false"],
                        help="Whether to prefer CUDAExecutionProvider if available.")
    args = parser.parse_args()

    if ort is None:
        print("ERROR: onnxruntime is not installed in this environment.")
        sys.exit(1)

    try:
        model_path = find_onnx_model(args.onnx_path)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    providers = ort.get_available_providers()
    use_cuda = ("CUDAExecutionProvider" in providers) if args.use_cuda == "auto" else (args.use_cuda == "true")
    ep = ["CUDAExecutionProvider"] if use_cuda and "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
    print(f"Using providers: {ep}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=ep)

    inputs, outputs = get_io_details(session)
    print("\nModel path:", model_path)
    print("Inputs:")
    for i, info in enumerate(inputs):
        print(f"  [{i}] name={info['name']} type={info['type']} shape={pretty_shape(info['shape'])}")
    print("Outputs:")
    for i, info in enumerate(outputs):
        print(f"  [{i}] name={info['name']} type={info['type']} shape={pretty_shape(info['shape'])}")

    # Try ONNX shape inference for additional hints
    inferred = infer_graph_shapes(model_path)
    if inferred is not None:
        try:
            value_infos = list(inferred.graph.value_info)
            print(f"\nInferred intermediate tensors: {len(value_infos)} (showing up to 10)")
            for i, vi in enumerate(value_infos[:10]):
                dims = [d.dim_value if (d.dim_value is not None and d.dim_value != 0) else None for d in vi.type.tensor_type.shape.dim]
                print(f"  - {vi.name}: {pretty_shape(dims)}")
        except Exception:
            pass

    # Run a tiny dummy inference to see actual output tensor ranks/shapes
    try:
        input_name = inputs[0]["name"]
        outputs_np = run_dummy_inference(session, input_name, inputs[0], args.input_res)
        print("\nRan dummy inference successfully.")
        for idx, out in enumerate(outputs_np):
            analysis = analyze_output_shape(np.asarray(out))
            print(f"  Output[{idx}] shape={out.shape}: {analysis}")
    except Exception as e:
        print(f"\nWARNING: Dummy inference failed: {e}")

    # Final conclusion based on declared outputs if any
    if len(outputs) == 1:
        oshape = outputs[0]["shape"]
        if len(oshape) == 3 and oshape[1:] == [261, 2048] and oshape[0] in (4, 24):
            print("\nConclusion: Model likely supports multi-layer tokens natively.")
        else:
            print("\nConclusion: Model likely exports a single feature map (not multi-layer). Changes to export may be required.")
    else:
        print("\nConclusion: Multiple outputs present; inspect above to confirm layers.")


if __name__ == "__main__":
    main()

