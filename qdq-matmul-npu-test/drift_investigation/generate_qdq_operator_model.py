#!/usr/bin/env python3
"""Generate a small float operator with input/output activation QDQ and stressed inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


OPERATORS = (
    "Add", "Clip", "Concat", "Gather", "Gelu", "LpNormalization", "Mul",
    "Neg", "Reshape", "Slice", "Softmax", "Sub", "Transpose", "Unsqueeze", "Where",
)
ACTIVATION_TYPES = ("uint16", "uint8")
REGIMES = ("input_clipped", "output_clipped")


def operator_case(
    op_type: str, seed: int,
) -> tuple[onnx.NodeProto, dict[str, np.ndarray], list[onnx.TensorProto], np.ndarray]:
    if op_type not in OPERATORS:
        raise ValueError(f"unsupported operator: {op_type}")
    rng = np.random.default_rng(seed)
    shape = (1, 2, 32, 32) if op_type == "Softmax" else (1, 16, 32)
    if op_type == "Gather":
        shape = (32, 32)
    elif op_type == "Concat":
        shape = (1, 16, 16)
    x = rng.normal(0, 2 if op_type == "Softmax" else 1, size=shape).astype(np.float32)
    inputs = {"x": x}
    initializers: list[onnx.TensorProto] = []
    node_inputs = ["activation"]
    attributes: dict[str, int | str | list[int]] = {}

    if op_type in ("Add", "Mul", "Sub", "Concat", "Where"):
        inputs["y"] = rng.normal(0, 1, size=shape).astype(np.float32)
        node_inputs.append("y")
    if op_type == "Add":
        raw = x + inputs["y"]
    elif op_type == "Mul":
        raw = x * inputs["y"]
    elif op_type == "Sub":
        raw = x - inputs["y"]
    elif op_type == "Neg":
        raw = -x
    elif op_type == "Clip":
        initializers.extend([
            numpy_helper.from_array(np.asarray(-2.0, dtype=np.float32), "minimum"),
            numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), "maximum"),
        ])
        node_inputs.extend(("minimum", "maximum"))
        raw = np.clip(x, -2, 2)
    elif op_type == "Where":
        inputs["condition"] = rng.integers(0, 2, size=shape, dtype=np.uint8).astype(np.bool_)
        node_inputs = ["condition", "activation", "y"]
        raw = np.where(inputs["condition"], x, inputs["y"])
    elif op_type == "Concat":
        attributes["axis"] = 2
        raw = np.concatenate((x, inputs["y"]), axis=2)
    elif op_type == "Transpose":
        attributes["perm"] = [0, 2, 1]
        raw = np.transpose(x, (0, 2, 1))
    elif op_type == "Reshape":
        initializers.append(numpy_helper.from_array(np.asarray([1, 8, 64], dtype=np.int64), "new_shape"))
        node_inputs.append("new_shape")
        raw = x.reshape(1, 8, 64)
    elif op_type == "Slice":
        for name, value in (("starts", [0]), ("ends", [16]), ("axes", [-1])):
            initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name))
            node_inputs.append(name)
        raw = x[..., :16]
    elif op_type == "Gather":
        indices = rng.integers(0, x.shape[0], size=(16,), dtype=np.int64)
        inputs["indices"] = indices
        node_inputs.append("indices")
        attributes["axis"] = 0
        raw = x[indices]
    elif op_type == "Unsqueeze":
        initializers.append(numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axes"))
        node_inputs.append("axes")
        raw = x[:, None, :, :]
    elif op_type == "LpNormalization":
        attributes.update(axis=-1, p=2)
        raw = x / np.linalg.norm(x, axis=-1, keepdims=True)
    elif op_type == "Softmax":
        attributes["axis"] = -1
        centered = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(centered)
        raw = exp / np.sum(exp, axis=-1, keepdims=True)
    elif op_type == "Gelu":
        attributes["approximate"] = "tanh"
        raw = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    else:
        raise ValueError(f"no implementation for {op_type}")
    node = helper.make_node(op_type, node_inputs, ["operator_result"], name="TargetOp", **attributes)
    return node, inputs, initializers, raw


def qdq_parameters(
    x: np.ndarray, raw_output: np.ndarray, activation_type: str, regime: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if activation_type not in ACTIVATION_TYPES or regime not in REGIMES:
        raise ValueError(f"unsupported activation type or regime: {activation_type}, {regime}")
    dtype = np.dtype(activation_type)
    limit = np.iinfo(dtype).max
    midpoint = (limit + 1) // 2
    symmetric_range = min(midpoint, limit - midpoint)
    input_bound = 0.5 if regime == "input_clipped" else 2 * float(np.max(np.abs(x)))
    output_nonnegative = bool(np.min(raw_output) >= 0)
    output_zero = 0 if output_nonnegative else midpoint
    output_range = limit if output_nonnegative else symmetric_range
    target = float(np.percentile(np.abs(raw_output), 70))
    output_bound = target if regime == "output_clipped" else 2 * float(np.max(np.abs(raw_output)))
    if input_bound <= 0 or output_bound <= 0:
        raise ValueError("QDQ reference inputs must have nonzero magnitude")
    return (
        np.asarray(input_bound / symmetric_range, dtype=np.float32),
        np.asarray(midpoint, dtype=dtype),
        np.asarray(output_bound / output_range, dtype=np.float32),
        np.asarray(output_zero, dtype=dtype),
    )


def build_model(
    op_type: str, activation_type: str, regime: str, seed: int,
) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    op, inputs, constants, raw_output = operator_case(op_type, seed)
    input_scale, input_zero, output_scale, output_zero = qdq_parameters(
        inputs["x"], raw_output, activation_type, regime,
    )
    initializers = [
        *constants,
        numpy_helper.from_array(input_scale, "activation_scale"),
        numpy_helper.from_array(input_zero, "activation_zero_point"),
        numpy_helper.from_array(output_scale, "output_scale"),
        numpy_helper.from_array(output_zero, "output_zero_point"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["x", "activation_scale", "activation_zero_point"],
            ["input_quantized"], name="QuantizeActivation",
        ),
        helper.make_node(
            "DequantizeLinear", ["input_quantized", "activation_scale", "activation_zero_point"],
            ["activation"], name="DequantizeActivation",
        ),
        op,
        helper.make_node(
            "QuantizeLinear", ["operator_result", "output_scale", "output_zero_point"],
            ["output_quantized"], name="QuantizeOutput",
        ),
        helper.make_node(
            "DequantizeLinear", ["output_quantized", "output_scale", "output_zero_point"],
            ["output"], name="DequantizeOutput",
        ),
    ]
    input_types = {
        np.dtype(np.float32): TensorProto.FLOAT,
        np.dtype(np.bool_): TensorProto.BOOL,
        np.dtype(np.int64): TensorProto.INT64,
    }
    graph_inputs = [
        helper.make_tensor_value_info(name, input_types[value.dtype], value.shape)
        for name, value in inputs.items()
    ]
    graph = helper.make_graph(
        nodes,
        f"{op_type}_activation_qdq_{activation_type}_{regime}",
        graph_inputs,
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, raw_output.shape)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 21)], ir_version=10)
    model.metadata_props.add(key="operator", value=op_type)
    model.metadata_props.add(key="activation_type", value=activation_type)
    model.metadata_props.add(key="regime", value=regime)
    model = shape_inference.infer_shapes(model)
    checker.check_model(model, full_check=True)
    return model, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=OPERATORS, required=True)
    parser.add_argument("--activation-type", choices=ACTIVATION_TYPES, default="uint16")
    parser.add_argument("--regime", choices=REGIMES, default="output_clipped")
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.inputs_output.exists() or args.output.resolve() == args.inputs_output.resolve():
        raise ValueError("output ONNX and input archive must be new, distinct files")
    model, inputs = build_model(args.op, args.activation_type, args.regime, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.inputs_output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    np.savez_compressed(args.inputs_output, **inputs)
    print(f"Saved {args.op} QDQ unit model: {args.output}, inputs: {args.inputs_output}")


if __name__ == "__main__":
    main()
