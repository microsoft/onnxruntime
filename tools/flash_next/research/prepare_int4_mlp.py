import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def prepare_case(out, rows, k, n):
    rng = np.random.default_rng(712 + rows + k + n)
    a = (rng.integers(-1, 2, (1, rows, k)) / 16).astype(np.float16)
    tensors = [numpy_helper.from_array(a, "a")]
    projections = []
    for name in ("gate", "up"):
        quantized = rng.integers(0, 16, (n, k), dtype=np.uint8)
        packed = quantized[:, 0::2] | (quantized[:, 1::2] << 4)
        scales = np.exp2(rng.integers(-7, -5, (n, k // 32))).astype(np.float16)
        weights = ((quantized.astype(np.float64) - 8) *
                   np.repeat(scales.astype(np.float64), 32, axis=1)).astype(np.float16)
        projection = (a.astype(np.float64) @ weights.astype(np.float64).T).astype(np.float16)
        projections.append(projection)
        tensors.extend([
            numpy_helper.from_array(packed.reshape(n, k // 32, 16), name + "_b"),
            numpy_helper.from_array(scales, name + "_scales"),
        ])
    gate, up = projections
    sigmoid = (1 / (1 + np.exp(-gate.astype(np.float64)))).astype(np.float16)
    activation = (gate * sigmoid).astype(np.float16)
    expected = (activation * up).astype(np.float16)
    if not np.isfinite(expected).all() or np.count_nonzero(expected) < expected.size // 4:
        raise RuntimeError("Fixture must contain finite, nontrivial outputs")
    attrs = dict(K=k, N=n, bits=4, block_size=32, accuracy_level=4)
    unfused = [
        helper.make_node("MatMulNBits", ["a", "gate_b", "gate_scales"], ["gate"],
                         domain="com.microsoft", **attrs),
        helper.make_node("MatMulNBits", ["a", "up_b", "up_scales"], ["up"],
                         domain="com.microsoft", **attrs),
        helper.make_node("Sigmoid", ["gate"], ["sigmoid"]),
        helper.make_node("Mul", ["gate", "sigmoid"], ["activation"]),
        helper.make_node("Mul", ["activation", "up"], ["y"]),
    ]
    fused = [helper.make_node(
        "MatMulNBitsMlp",
        ["a", "", "", "gate_b", "gate_scales", "", "up_b", "up_scales", ""],
        ["y"], domain="com.microsoft", activation="silu", **attrs)]
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT16, list(expected.shape))
    for variant, nodes in (("unfused", unfused), ("fused", fused)):
        name = f"mlp_m{rows}_k{k}_n{n}_{variant}"
        model = helper.make_model(
            helper.make_graph(nodes, name, [], [output], tensors),
            opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)])
        model.ir_version = 10
        onnx.checker.check_model(model)
        onnx.save(model, out / f"{name}.onnx")
        reference = helper.make_model(
            helper.make_graph([helper.make_node("Identity", ["expected"], ["y"])],
                              name + "_reference", [], [output],
                              [numpy_helper.from_array(expected, "expected")]),
            opset_imports=[helper.make_opsetid("", 22)])
        reference.ir_version = 10
        onnx.checker.check_model(reference)
        onnx.save(reference, out / f"{name}_reference.onnx")
        print(name, "outputs", expected.size, "nonzero", np.count_nonzero(expected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    for k, n in ((32, 8), (2560, 640)):
        for rows in (1, 2, 16):
            prepare_case(args.output, rows, k, n)
