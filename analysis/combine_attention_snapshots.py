#!/usr/bin/env python3
"""
Combine multiple attention snapshot .npz files into a single archive.

Each input must be provided as ``label:path`` (or ``label=path``). Only keys that
start with the given label prefix will be copied into the combined archive to
avoid accidental collisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def _parse_inputs(specs) -> Dict[str, Path]:
    inputs: Dict[str, Path] = {}
    for spec in specs:
        if "=" in spec:
            label, path_str = spec.split("=", 1)
        elif ":" in spec:
            label, path_str = spec.split(":", 1)
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid input specification '{spec}'. Expected format label:path"
            )
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"Empty label in specification '{spec}'")
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Snapshot file '{path}' does not exist for label '{label}'.")
        if label in inputs:
            raise argparse.ArgumentTypeError(f"Duplicate label '{label}' provided.")
        inputs[label] = path
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine attention snapshot .npz files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Snapshot specifications of the form label:path (e.g. baseline:baseline_snapshot.npz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the combined snapshot (.npz).",
    )
    args = parser.parse_args()

    input_map = _parse_inputs(args.inputs)
    combined: Dict[str, np.ndarray] = {}

    for label, path in input_map.items():
        with np.load(path, allow_pickle=True) as data:
            prefix = f"{label}_"
            label_keys = [k for k in data.files if k.startswith(prefix)]
            if not label_keys:
                print(
                    f"[combine_snapshots] Warning: no keys starting with '{prefix}' found in {path}. "
                    "This label will be skipped."
                )
                continue
            for key in label_keys:
                if key in combined:
                    raise ValueError(
                        f"Key '{key}' already present in combined archive. "
                        f"Conflict caused by label '{label}' from {path}."
                    )
                combined[key] = data[key]

    if not combined:
        raise RuntimeError("No snapshot data collected; ensure inputs contain expected keys.")

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **combined)
    print(
        f"[combine_snapshots] Wrote combined snapshot with {len(combined)} arrays to {output_path}"
    )


if __name__ == "__main__":
    main()
