#!/usr/bin/env python3
"""Generate an asymmetric opset-17 QDQ Gemm model without a bias input."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper

from generate_qdq_matmul_model import PROJECTIONS, Projection, asymmetric_params


OPSET_VERSION = 17
MICROSOFT_DOMAIN = "com.microsoft"


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
        help="CLIP quantization profile and default shapes; default: visual.",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLUMNS"),
        help="Override the selected projection's 2D input shape.",
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
        choices=("per-tensor", "per-channel"),
        default="per-tensor",
        help="Asymmetric uint8 weight quantization mode; default: per-tensor.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-scale",
        type=float,
        help="Override the reference model's per-tensor output scale.",
    )
    args = parser.parse_args()

    projection = PROJECTIONS[args.projection]
    weight_shape = resolve_weight_shape(args, projection)
    input_shape = resolve_input_shape(args, projection, weight_shape)
    if any(dimension <= 0 for dimension in (*input_shape, *weight_shape)):
        parser.error("input and weight dimensions must be positive")
    if input_shape[1] != weight_shape[0]:
        parser.error(
            "input columns must equal weight rows: "
            f"{input_shape[1]} != {weight_shape[0]}"
        )
    if args.output_scale is not None and args.output_scale <= 0:
        parser.error("--output-scale must be greater than zero")
    if args.output is None:
        args.output = (
            Path(__file__).resolve().parent
            / "unit-models"
            / f"clip_{args.projection}_qdq_gemm.onnx"
        )
    return args


def resolve_weight_shape(
    args: argparse.Namespace, projection: Projection
) -> tuple[int, int]:
    return (
        projection.weight_shape
        if args.weight_shape is None
        else tuple(args.weight_shape)
    )


def resolve_input_shape(
    args: argparse.Namespace,
    projection: Projection,
    weight_shape: tuple[int, int],
) -> tuple[int, int]:
    return (
        (projection.input_shape[0], weight_shape[0])
        if args.input_shape is None
        else tuple(args.input_shape)
    )


def quantize_weight(
    values: np.ndarray,
    projection: Projection,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    attributes: dict[str, int] = {}
    if mode == "per-tensor":
        scale = np.asarray(projection.weight_scale, dtype=np.float32)
        zero_point = np.asarray(projection.weight_zero_point, dtype=np.uint8)
        quantized = np.rint(values / scale) + zero_point
    else:
        scale, zero_point = asymmetric_params(values, axis=0)
        quantized = np.rint(values / scale[None, :]) + zero_point[None, :]
        attributes["axis"] = 1
    return (
        np.clip(quantized, 0, 255).astype(np.uint8),
        scale,
        zero_point,
        attributes,
    )


def build_model(args: argparse.Namespace) -> onnx.ModelProto:
    projection = PROJECTIONS[args.projection]
    weight_shape = resolve_weight_shape(args, projection)
    input_shape = resolve_input_shape(args, projection, weight_shape)
    output_shape = (input_shape[0], weight_shape[1])
    rng = np.random.default_rng(args.seed)
    weight = rng.normal(0.0, 0.02, weight_shape).astype(np.float32)
    quantized_weight, weight_scale, weight_zero_point, weight_attributes = (
        quantize_weight(weight, projection, args.weight_quantization)
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
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "activation_scale", "activation_zero_point"],
            ["activation_quantized"],
            name="QuantizeActivation",
            domain=MICROSOFT_DOMAIN,
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
            domain=MICROSOFT_DOMAIN,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_quantized", "weight_scale", "weight_zero_point"],
            ["weight"],
            name="DequantizeWeight",
            domain=MICROSOFT_DOMAIN,
            **weight_attributes,
        ),
        helper.make_node(
            "Gemm",
            ["activation", "weight"],
            ["gemm"],
            name="Gemm",
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0,
        ),
        helper.make_node(
            "QuantizeLinear",
            ["gemm", "output_scale", "output_zero_point"],
            ["output_quantized"],
            name="QuantizeOutput",
            domain=MICROSOFT_DOMAIN,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_quantized", "output_scale", "output_zero_point"],
            ["output"],
            name="DequantizeOutput",
            domain=MICROSOFT_DOMAIN,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        f"clip_{args.projection}_qdq_gemm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name=Path(__file__).name,
        opset_imports=[
            helper.make_opsetid("", OPSET_VERSION),
            helper.make_opsetid(MICROSOFT_DOMAIN, 1),
        ],
    )
    model.ir_version = 8
    model.metadata_props.add(key="operation", value="Gemm")
    model.metadata_props.add(key="bias", value="none")
    model.metadata_props.add(
        key="weight_quantization", value=args.weight_quantization
    )
    model.metadata_props.add(
        key="weight_shape", value="x".join(str(dimension) for dimension in weight_shape)
    )
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
