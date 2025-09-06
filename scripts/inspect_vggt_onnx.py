#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import json
from typing import Any, Dict, List, Optional


def _install_if_missing(packages: List[str]) -> None:
    """Best-effort install of required packages if missing."""
    import importlib
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    if missing:
        import subprocess
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *missing]
        try:
            subprocess.check_call(cmd)
        except Exception as exc:
            print(f"[WARN] Failed to auto-install {missing}: {exc}")


def _safe_shape_proto_to_list(type_proto) -> List[Optional[int]]:
    dims: List[Optional[int]] = []
    if type_proto is None:
        return dims
    try:
        shape = type_proto.tensor_type.shape
        for d in shape.dim:
            if d.dim_value:
                dims.append(int(d.dim_value))
            elif d.dim_param:
                dims.append(None)
            else:
                dims.append(None)
    except Exception:
        pass
    return dims


def _dtype_from_onnx_type(elem_type: str) -> str:
    # elem_type example: 'tensor(float16)'
    open_paren = elem_type.find("(")
    close_paren = elem_type.find(")")
    if open_paren != -1 and close_paren != -1:
        return elem_type[open_paren + 1:close_paren]
    return elem_type


def find_onnx_file(path: str) -> str:
    """Return a concrete .onnx file path. If a directory is provided, pick the first matching file."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        # Prefer common file names
        preferred = [
            os.path.join(path, "vggt_fp16.onnx"),
            os.path.join(path, "model.onnx"),
        ]
        for p in preferred:
            if os.path.isfile(p):
                return p
        matches = sorted(glob.glob(os.path.join(path, "*.onnx")))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No .onnx file found at {path}")


def describe_model_graph(onnx_model: "onnx.ModelProto") -> Dict[str, Any]:
    g = onnx_model.graph
    outputs: List[Dict[str, Any]] = []
    for o in g.output:
        outputs.append({
            "name": o.name,
            "shape": _safe_shape_proto_to_list(o.type),
            "elem_type": getattr(o.type.tensor_type, "elem_type", None),
        })
    inputs: List[Dict[str, Any]] = []
    for i in g.input:
        inputs.append({
            "name": i.name,
            "shape": _safe_shape_proto_to_list(i.type),
            "elem_type": getattr(i.type.tensor_type, "elem_type", None),
        })
    return {"inputs": inputs, "outputs": outputs}


def try_infer(model_path: str, num_images: int = 2, height: int = 518, width: int = 518) -> Dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    providers = [
        ("CUDAExecutionProvider", {}),
        ("CPUExecutionProvider", {}),
    ]
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=[p[0] for p in providers])

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_type = input_meta.type  # e.g., 'tensor(float16)'
    dtype_name = _dtype_from_onnx_type(input_type)

    np_dtype = {
        "float16": np.float16,
        "float": np.float32,
        "float32": np.float32,
        "double": np.float64,
    }.get(dtype_name, np.float32)

    # Expecting [S, 3, H, W]
    dummy = np.random.rand(num_images, 3, height, width).astype(np_dtype)

    outputs = session.run(None, {input_name: dummy})
    output_names = [o.name for o in session.get_outputs()]
    result = {name: (getattr(arr, "shape", None)) for name, arr in zip(output_names, outputs)}
    return {
        "providers": session.get_providers(),
        "input_name": input_name,
        "input_dtype": dtype_name,
        "output_shapes": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect VGGT ONNX model for available outputs and shapes.")
    parser.add_argument("path", help="Path to .onnx file or directory containing it")
    parser.add_argument("--num-images", type=int, default=2, help="Number of images for a dry run inference")
    parser.add_argument("--height", type=int, default=518, help="Input height for dummy data")
    parser.add_argument("--width", type=int, default=518, help="Input width for dummy data")
    parser.add_argument("--no-run", action="store_true", help="Only print graph IO, skip inference")
    args = parser.parse_args()

    # Ensure runtime dependencies are present
    _install_if_missing(["onnx", "onnxruntime"])

    import onnx

    onnx_path = find_onnx_file(args.path)
    print(f"[INFO] Using ONNX model: {onnx_path}")

    model = onnx.load(onnx_path)
    io_desc = describe_model_graph(model)
    print("[INFO] Graph IO summary (dim None means dynamic/unknown):")
    print(json.dumps(io_desc, indent=2))

    # Heuristic check for desired feature outputs
    desired_feature_dims = {2048}
    found_feature_like = False
    for out in io_desc.get("outputs", []):
        shape = out.get("shape", [])
        if len(shape) >= 2 and any(d in desired_feature_dims for d in shape if isinstance(d, int)):
            found_feature_like = True
            break

    if not args.no_run:
        try:
            run_info = try_infer(onnx_path, num_images=args.num_images, height=args.height, width=args.width)
            print("[INFO] Inference output shapes:")
            print(json.dumps(run_info, indent=2))
        except Exception as exc:
            print(f"[WARN] Inference run failed: {exc}")

    # Conclusion message regarding feature availability
    if found_feature_like:
        print("[NOTE] Found outputs whose dimensions include 2048. Inspect the names above.")
    else:
        print("[NOTE] No outputs appear to expose token features (e.g., 2048-dim). Current model likely does not export per-layer tokens.")


if __name__ == "__main__":
    main()

