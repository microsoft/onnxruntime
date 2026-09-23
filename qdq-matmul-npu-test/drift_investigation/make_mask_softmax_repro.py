"""Generate matched QDQ-mask and float-mask Add/Softmax controls without learned weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper
import onnxruntime as ort


def make_model(quantized_mask: bool, scale: float, zero_point: int, width: int) -> onnx.ModelProto:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("mask QDQ scale must be positive and finite")
    if not 0 <= zero_point <= np.iinfo(np.uint16).max:
        raise ValueError("mask QDQ zero point must fit uint16")
    if width < 2:
        raise ValueError("Softmax width must be at least two")

    nodes = []
    initializers = []
    mask = "mask"
    if quantized_mask:
        initializers = [
            numpy_helper.from_array(np.array(scale, dtype=np.float32), name="mask_scale"),
            numpy_helper.from_array(np.array(zero_point, dtype=np.uint16), name="mask_zero_point"),
        ]
        nodes = [
            helper.make_node("QuantizeLinear", ["mask", "mask_scale", "mask_zero_point"], ["mask_quantized"]),
            helper.make_node(
                "DequantizeLinear", ["mask_quantized", "mask_scale", "mask_zero_point"], ["mask_dequantized"]
            ),
        ]
        mask = "mask_dequantized"
    nodes.extend([
        helper.make_node("Add", ["scores", mask], ["masked_scores"]),
        helper.make_node("Softmax", ["masked_scores"], ["output"], axis=-1),
    ])
    shape = [1, 1, width]
    graph = helper.make_graph(
        nodes, "qdq_mask_softmax" if quantized_mask else "float_mask_softmax",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ("scores", "mask")],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    checker.check_model(model)
    return model


def write_repro(output_dir: Path, scale: float, zero_point: int, width: int, mask_value: float) -> None:
    mask_quantized = mask_value / scale + zero_point
    if not np.isfinite(mask_value) or not 0 <= mask_quantized <= np.iinfo(np.uint16).max:
        raise ValueError("mask value is outside the selected uint16 QDQ range")
    if mask_quantized != round(mask_quantized):
        raise ValueError("mask value must be exactly representable on the selected QDQ grid")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    scores = np.linspace(-1.0, 1.0, width, dtype=np.float32).reshape(1, 1, width)
    mask = np.zeros_like(scores)
    mask[..., -1] = np.float32(mask_value)
    inputs = {"scores": scores, "mask": mask}
    qdq = output_dir / "qdq_mask.onnx"
    control = output_dir / "float_mask.onnx"
    onnx.save(make_model(True, scale, zero_point, width), qdq)
    onnx.save(make_model(False, scale, zero_point, width), control)
    np.savez_compressed(output_dir / "inputs.npz", **inputs)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    outputs = [
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"]).run(None, inputs)[0]
        for path in (qdq, control)
    ]
    if not np.array_equal(*outputs):
        raise ValueError("QDQ and float-mask controls do not have bitwise-identical CPU output")
    print(f"Saved CPU-equivalent controls and inputs in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--zero-point", type=int, default=100)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--mask-value", type=float, default=0.0)
    args = parser.parse_args()
    write_repro(args.output_dir, args.scale, args.zero_point, args.width, args.mask_value)


if __name__ == "__main__":
    main()
