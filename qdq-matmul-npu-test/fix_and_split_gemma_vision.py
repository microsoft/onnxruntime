#!/usr/bin/env python3
"""Repair Gemma 4 vision padding-mask QDQ and split off the dynamic pooler."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx_ir as ir

from split_gemma_vision_pooler import describe_component, save_component, split_components


_MASK_CONSTANT = "v_vision_encoder.encoder.CastLike_29"
_MASK_QDQ = "v_vision_encoder.encoder.Where_31"
_EXPECTED_MASK_USERS = {
    f"{_MASK_QDQ}_zero_point": {
        f"{_MASK_QDQ}_QuantizeLinear",
        f"{_MASK_QDQ}_DequantizeLinear",
        "v_vision_encoder.encoder.Unsqueeze_32_QuantizeLinear",
        "v_vision_encoder.encoder.Unsqueeze_32_DequantizeLinear",
    },
    f"{_MASK_CONSTANT}_scale": {f"{_MASK_CONSTANT}_DequantizeLinear"},
    f"{_MASK_CONSTANT}_quantized": {f"{_MASK_CONSTANT}_DequantizeLinear"},
    f"{_MASK_CONSTANT}_zero_point": {f"{_MASK_CONSTANT}_DequantizeLinear"},
    f"{_MASK_QDQ}_scale": {
        f"{_MASK_QDQ}_QuantizeLinear",
        f"{_MASK_QDQ}_DequantizeLinear",
        "v_vision_encoder.encoder.Unsqueeze_32_QuantizeLinear",
        "v_vision_encoder.encoder.Unsqueeze_32_DequantizeLinear",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Original Mobius Gemma 4 vision QDQ model.")
    parser.add_argument("--encoder-output", type=Path, help="Corrected encoder ONNX output path.")
    parser.add_argument("--pooler-output", type=Path, help="Corrected CPU pooler/projector ONNX output path.")
    parser.add_argument(
        "--mask-value",
        type=int,
        default=-100,
        help="Finite padding bias represented exactly by uint16 mask QDQ; default: -100.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the selected output files.")
    return parser.parse_args()


def _scalar_initializer(graph: ir.Graph, name: str, dtype: np.dtype) -> ir.Value:
    value = graph.initializers.get(name)
    if value is None or value.const_value is None:
        raise ValueError(f"expected scalar initializer {name!r}")
    data = value.const_value.numpy()
    if data.shape != () or data.dtype != dtype:
        raise ValueError(f"{name!r} must be a scalar {dtype}; got {data.shape} {data.dtype}")
    return value


def _check_mask_graph(graph: ir.Graph) -> None:
    expected = {
        f"{_MASK_CONSTANT}_quantized": (np.dtype(np.uint16), 0),
        f"{_MASK_CONSTANT}_scale": (np.dtype(np.float32), np.float32(1e9 / 65535).item()),
        f"{_MASK_CONSTANT}_zero_point": (np.dtype(np.uint16), 65535),
        f"{_MASK_QDQ}_scale": (np.dtype(np.float32), 1.0),
        f"{_MASK_QDQ}_zero_point": (np.dtype(np.uint16), 0),
    }
    for name, (dtype, original_value) in expected.items():
        value = _scalar_initializer(graph, name, dtype)
        if value.const_value.numpy().item() != original_value:
            raise ValueError(f"{name!r} no longer has the expected original value {original_value}")

    for name, expected_users in _EXPECTED_MASK_USERS.items():
        value = graph.initializers[name]
        actual_users = {use.node.name for use in value.uses()}
        if actual_users != expected_users:
            raise ValueError(f"{name!r} has unexpected consumers: {sorted(actual_users)}")

    mask = next((node for node in graph if node.name == "vision_encoder/encoder/Where_node_31"), None)
    if mask is None or mask.op_type != "Where" or mask.inputs[1].name != f"{_MASK_CONSTANT}_DequantizeLinear_Output":
        raise ValueError("expected Mobius padding-mask Where node was not found")


def fix_mask_qdq(model: ir.Model, mask_value: int = -100) -> None:
    if not -65535 <= mask_value < 0:
        raise ValueError("--mask-value must be a negative integer in [-65535, -1]")

    graph = model.graph
    _check_mask_graph(graph)
    # The original uint16 zero point of 0 clips negative masks to zero. Shift it
    # so both the masked value and zero are exactly representable at scale 1.
    graph.initializers[f"{_MASK_QDQ}_zero_point"].const_value = ir.tensor(
        np.asarray(-mask_value, dtype=np.uint16)
    )
    # The preceding quantized constant must yield the same finite mask value.
    graph.initializers[f"{_MASK_CONSTANT}_scale"].const_value = ir.tensor(
        np.asarray(-mask_value / 65535, dtype=np.float32)
    )
    model.metadata_props["gemma_vision_padding_mask"] = str(mask_value)


def main() -> None:
    args = parse_args()
    input_path = args.model.resolve()
    if not input_path.is_file():
        raise ValueError(f"input model does not exist: {input_path}")

    encoder_path = (
        args.encoder_output or input_path.with_name(f"{input_path.stem}_mask_fixed_encoder.onnx")
    ).resolve()
    pooler_path = (
        args.pooler_output or input_path.with_name(f"{input_path.stem}_mask_fixed_pooler_projector.onnx")
    ).resolve()
    if len({input_path, encoder_path, pooler_path}) != 3:
        raise ValueError("input, encoder output, and pooler output must all have distinct paths")
    for path in (encoder_path, pooler_path):
        if not args.overwrite:
            for candidate in (path, path.parent / f"{path.name}.data"):
                if candidate.exists():
                    raise ValueError(f"output already exists: {candidate}")

    model = ir.load(input_path)
    fix_mask_qdq(model, args.mask_value)
    encoder_model, pooler_model = split_components(model)
    save_component(encoder_model, encoder_path, args.overwrite)
    save_component(pooler_model, pooler_path, args.overwrite)
    print(f"Padding mask: {args.mask_value} (uint16 scale=1, zero point={-args.mask_value})")
    describe_component("NPU encoder", encoder_model, encoder_path)
    describe_component("CPU pooler/projector", pooler_model, pooler_path)


if __name__ == "__main__":
    main()
