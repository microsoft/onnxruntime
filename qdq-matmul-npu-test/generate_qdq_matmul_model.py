#!/usr/bin/env python3
"""Generate an asymmetric CLIP uint16-activation DQ/MatMul/Q test model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


OPSET_VERSION = 21


@dataclass(frozen=True)
class Projection:
    input_shape: tuple[int, int]
    weight_shape: tuple[int, int]
    activation_scale: float
    activation_zero_point: int
    weight_scale: float
    weight_zero_point: int
    output_scale: float
    output_zero_point: int


PROJECTIONS = {
    "visual": Projection(
        (1, 768),
        (768, 512),
        0.0001934150350280106,
        36400,
        0.0009047564235515893,
        113,
        0.00015079299919307232,
        16116,
    ),
    "text": Projection(
        (10, 512),
        (512, 512),
        0.0006906900089234114,
        21264,
        0.0010325711918994784,
        143,
        0.00008716459706192836,
        14058,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output ONNX model path; defaults under unit-models.",
    )
    parser.add_argument(
        "--projection",
        choices=tuple(PROJECTIONS),
        default="visual",
        help="CLIP projection shapes to use; default: visual.",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        metavar="DIM",
        help=(
            "Override the input shape. The final dimension must match the "
            "weight rows."
        ),
    )
    parser.add_argument(
        "--weight-shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLUMNS"),
        help="Override the selected projection's 2D weight shape.",
    )
    parser.add_argument(
        "--weight-quantization",
        choices=("per-tensor", "per-channel", "blockwise"),
        default="per-tensor",
        help="Asymmetric uint8 weight quantization mode; default: per-tensor.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        choices=(32, 128),
        default=32,
        help="Input-axis block size for blockwise weights; default: 32.",
    )
    parser.add_argument(
        "--add-boundary-nodes",
        action="store_true",
        help="Add Relu nodes outside the DQ-MatMul-Q subgraph boundaries.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-scale",
        type=float,
        help="Override the reference model's per-tensor output scale.",
    )
    args = parser.parse_args()
    if args.output_scale is not None and args.output_scale <= 0:
        parser.error("--output-scale must be greater than zero")
    projection = PROJECTIONS[args.projection]
    weight_shape = (
        projection.weight_shape
        if args.weight_shape is None
        else tuple(args.weight_shape)
    )
    if any(dimension <= 0 for dimension in weight_shape):
        parser.error("--weight-shape dimensions must be positive")
    if args.input_shape is not None:
        if any(dimension <= 0 for dimension in args.input_shape):
            parser.error("--input-shape dimensions must be positive")
        if args.input_shape[-1] != weight_shape[0]:
            parser.error(
                "--input-shape's final dimension must be "
                f"{weight_shape[0]} to match the weight rows"
            )
    if args.output is None:
        args.output = (
            Path(__file__).resolve().parent
            / "unit-models"
            / f"clip_{args.projection}_dq_matmul_q.onnx"
        )
    return args


def asymmetric_params(
    values: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.minimum(np.min(values, axis=axis), 0.0)
    maximum = np.maximum(np.max(values, axis=axis), 0.0)
    scale = np.maximum((maximum - minimum) / 255.0, 1e-8).astype(np.float32)
    zero_point = np.clip(np.rint(-minimum / scale), 0, 255).astype(np.uint8)
    return scale, zero_point


def quantize_weight(
    values: np.ndarray,
    projection: Projection,
    mode: str,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    attributes: dict[str, int] = {}
    if mode == "per-tensor":
        scale = np.asarray(projection.weight_scale, dtype=np.float32)
        zero_point = np.asarray(projection.weight_zero_point, dtype=np.uint8)
        quantized = np.rint(values / scale) + zero_point
    elif mode == "per-channel":
        scale, zero_point = asymmetric_params(values, axis=0)
        quantized = np.rint(values / scale[None, :]) + zero_point[None, :]
        attributes["axis"] = 1
    else:
        rows, columns = values.shape
        block_count = (rows + block_size - 1) // block_size
        scale = np.empty((block_count, columns), dtype=np.float32)
        zero_point = np.empty((block_count, columns), dtype=np.uint8)
        quantized = np.empty(values.shape, dtype=np.float32)
        for block_index in range(block_count):
            start = block_index * block_size
            end = min(start + block_size, rows)
            block_scale, block_zero_point = asymmetric_params(
                values[start:end, :], axis=0
            )
            scale[block_index, :] = block_scale
            zero_point[block_index, :] = block_zero_point
            quantized[start:end, :] = (
                np.rint(values[start:end, :] / block_scale[None, :])
                + block_zero_point[None, :]
            )
        attributes.update(axis=0, block_size=block_size)

    return (
        np.clip(quantized, 0, 255).astype(np.uint8),
        scale,
        zero_point,
        attributes,
    )


def build_model(args: argparse.Namespace) -> onnx.ModelProto:
    projection = PROJECTIONS[args.projection]
    weight_shape = (
        projection.weight_shape
        if args.weight_shape is None
        else tuple(args.weight_shape)
    )
    input_shape = (
        projection.input_shape[:-1] + (weight_shape[0],)
        if args.input_shape is None
        else tuple(args.input_shape)
    )
    output_shape = input_shape[:-1] + (weight_shape[1],)
    rng = np.random.default_rng(args.seed)
    weight = rng.normal(0.0, 0.02, weight_shape).astype(np.float32)
    quantized_weight, weight_scale, weight_zero_point, weight_attributes = (
        quantize_weight(
            weight,
            projection,
            args.weight_quantization,
            args.block_size,
        )
    )
    output_scale = (
        projection.output_scale
        if args.output_scale is None
        else args.output_scale
    )

    initializers = [
        numpy_helper.from_array(
            np.asarray(projection.activation_scale, dtype=np.float32),
            "activation_scale",
        ),
        numpy_helper.from_array(
            np.asarray(projection.activation_zero_point, dtype=np.uint16),
            "activation_zero_point",
        ),
        numpy_helper.from_array(quantized_weight, "weight_quantized"),
        numpy_helper.from_array(weight_scale, "weight_scale"),
        numpy_helper.from_array(weight_zero_point, "weight_zero_point"),
        numpy_helper.from_array(
            np.asarray(output_scale, dtype=np.float32), "output_scale"
        ),
        numpy_helper.from_array(
            np.asarray(projection.output_zero_point, dtype=np.uint16),
            "output_zero_point",
        ),
    ]
    nodes: list[onnx.NodeProto] = []
    quantize_input = "input"
    if args.add_boundary_nodes:
        nodes.append(
            helper.make_node(
                "Relu",
                [quantize_input],
                ["input_relu"],
                name="InputBoundaryRelu",
            )
        )
        quantize_input = "input_relu"

    nodes.extend(
        [
            helper.make_node(
                "QuantizeLinear",
                [quantize_input, "activation_scale", "activation_zero_point"],
                ["activation_quantized"],
                name="QuantizeActivation",
            ),
            helper.make_node(
                "DequantizeLinear",
                [
                    "activation_quantized",
                    "activation_scale",
                    "activation_zero_point",
                ],
                ["activation"],
                name="DequantizeActivation",
            ),
            helper.make_node(
                "DequantizeLinear",
                ["weight_quantized", "weight_scale", "weight_zero_point"],
                ["weight"],
                name="DequantizeWeight",
                **weight_attributes,
            ),
        ]
    )

    nodes.append(
        helper.make_node(
            "MatMul", ["activation", "weight"], ["matmul"], name="MatMul"
        )
    )

    dequantized_output = (
        "output_dequantized" if args.add_boundary_nodes else "output"
    )
    nodes.extend(
        (
            helper.make_node(
                "QuantizeLinear",
                ["matmul", "output_scale", "output_zero_point"],
                ["output_quantized"],
                name="QuantizeOutput",
            ),
            helper.make_node(
                "DequantizeLinear",
                ["output_quantized", "output_scale", "output_zero_point"],
                [dequantized_output],
                name="DequantizeOutput",
            ),
        )
    )
    graph_output = dequantized_output
    if args.add_boundary_nodes:
        nodes.append(
            helper.make_node(
                "Relu",
                [graph_output],
                ["output"],
                name="OutputBoundaryRelu",
            )
        )
        graph_output = "output"

    graph = helper.make_graph(
        nodes,
        f"clip_{args.projection}_dq_matmul_q",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, input_shape
            )
        ],
        [
            helper.make_tensor_value_info(
                graph_output, TensorProto.FLOAT, output_shape
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name=Path(__file__).name,
        opset_imports=[helper.make_opsetid("", OPSET_VERSION)],
    )
    model.ir_version = 8
    model.metadata_props.add(
        key="reference_model", value="amd-clip/clip_vit_base_patch16_amd.onnx"
    )
    model.metadata_props.add(key="projection", value=args.projection)
    model.metadata_props.add(
        key="weight_quantization", value=args.weight_quantization
    )
    model.metadata_props.add(
        key="weight_shape", value="x".join(str(dimension) for dimension in weight_shape)
    )
    if args.weight_quantization == "blockwise":
        model.metadata_props.add(key="block_size", value=str(args.block_size))
    model.metadata_props.add(
        key="boundary_nodes", value=str(args.add_boundary_nodes).lower()
    )
    model = shape_inference.infer_shapes(model)
    checker.check_model(model, full_check=True)
    return model


def main() -> None:
    args = parse_args()
    model = build_model(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)
    print(f"Saved validated ONNX model to {args.output}")


if __name__ == "__main__":
    main()
