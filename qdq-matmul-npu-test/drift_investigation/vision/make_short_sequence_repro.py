"""Resize static sequence metadata and slice QKV/mask inputs for a smaller attention repro."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx


def resize_model(source: Path, output: Path, source_length: int, length: int) -> None:
    if not source.is_file():
        raise ValueError(f"source model does not exist: {source}")
    model = onnx.load(source, load_external_data=False)
    changed = 0
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.dim_value == source_length:
                dimension.dim_value = length
                changed += 1
    if changed == 0:
        raise ValueError(f"{source}: no static dimensions equal --source-length={source_length}")
    model.metadata_props.add(key="synthetic_sequence_length", value=str(length))
    model.metadata_props.add(key="synthetic_source_sequence_length", value=str(source_length))
    onnx.save(model, output)
    print(f"Saved {output} after changing {changed} sequence dimensions")


def _slice_axis(value: np.ndarray, axis: int, length: int, name: str) -> np.ndarray:
    normalized_axis = axis if axis >= 0 else value.ndim + axis
    if normalized_axis < 0 or normalized_axis >= value.ndim:
        raise ValueError(f"{name}: axis {axis} is invalid for rank {value.ndim}")
    if value.shape[normalized_axis] < length:
        raise ValueError(
            f"{name}: axis {normalized_axis} has length {value.shape[normalized_axis]}, less than {length}"
        )
    slices = [slice(None)] * value.ndim
    slices[normalized_axis] = slice(0, length)
    return value[tuple(slices)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("length", type=int, help="New query/key/value sequence length.")
    parser.add_argument("--source-length", type=int, default=2520)
    parser.add_argument("--model", type=Path, action="append", required=True, help="Model to resize (repeatable).")
    parser.add_argument("--inputs", type=Path, required=True, help="Source NPZ containing QKV and pre-QDQ mask.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-name", default="query")
    parser.add_argument("--key-name", default="key")
    parser.add_argument("--value-name", default="value")
    parser.add_argument("--mask-name", default="mask_pre_qdq")
    parser.add_argument("--query-axis", type=int, default=2)
    parser.add_argument("--key-axis", type=int, default=3)
    parser.add_argument("--value-axis", type=int, default=2)
    parser.add_argument("--mask-axis", type=int, default=3)
    parser.add_argument(
        "--allow-nonzero-mask",
        action="store_true",
        help="Allow a sliced mask containing nonzero values; default requires an unpadded all-zero mask.",
    )
    args = parser.parse_args()

    if args.source_length < 2 or args.length < 2 or args.length > args.source_length:
        raise ValueError("require 2 <= length <= source-length")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    model_paths = [path.resolve() for path in args.model]
    output_names = [path.name for path in model_paths]
    if len(set(output_names)) != len(output_names):
        raise ValueError("--model basenames must be unique")
    inputs_path = args.inputs.resolve()
    if not inputs_path.is_file():
        raise ValueError(f"input archive does not exist: {inputs_path}")

    output_dir.mkdir(parents=True)
    for source, name in zip(model_paths, output_names, strict=True):
        resize_model(source, output_dir / name, args.source_length, args.length)

    names_and_axes = (
        (args.query_name, args.query_axis),
        (args.key_name, args.key_axis),
        (args.value_name, args.value_axis),
        (args.mask_name, args.mask_axis),
    )
    with np.load(inputs_path, allow_pickle=False) as archive:
        missing = [name for name, _ in names_and_axes if name not in archive]
        if missing:
            raise ValueError(f"{inputs_path}: missing arrays {missing}")
        values = {
            name: _slice_axis(archive[name], axis, args.length, name)
            for name, axis in names_and_axes
        }
    if not args.allow_nonzero_mask and np.any(values[args.mask_name] != 0):
        raise ValueError("expected an all-zero unpadded mask; use --allow-nonzero-mask to override")
    np.savez_compressed(output_dir / "inputs.npz", **values)
    print(f"Saved length={args.length} attention repro and inputs at {output_dir}")


if __name__ == "__main__":
    main()
