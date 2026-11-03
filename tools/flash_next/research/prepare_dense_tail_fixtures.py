from pathlib import Path
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import load_external_data_for_tensor


parser = argparse.ArgumentParser()
parser.add_argument("--skinny", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent / ("skinny-fixtures" if args.skinny else "dense-tail-fixtures")
root.mkdir(exist_ok=True)
cases = [
    (4, 256, 132, 4, False, False, False),
    (16, 256, 260, 4, False, False, False),
    (129, 256, 516, 4, False, False, False),
    (4, 10240, 324, 4, False, False, False),
    (16, 10240, 324, 4, False, False, False),
    (128, 256, 324, 4, False, False, False),
    (129, 256, 324, 4, False, False, False),
    (16, 256, 324, 4, True, False, False),
    (16, 256, 324, 4, True, True, False),
    (16, 256, 324, 8, True, False, False),
    (16, 256, 324, 4, False, False, True),
    (16, 256, 323, 4, False, False, False),
    (1, 256, 324, 4, False, False, False),
    (16, 256, 128, 4, False, False, False),
]
if args.skinny:
    cases = [
        (m, 2048, n, 4, False, False, False)
        for m in (1, 2, 4, 5) for n in (1, 323, 324, 640, 1024, 1025)
    ] + [
        (1, k, 324, bits, bias, zp, fp32)
        for k, bits, bias, zp, fp32 in (
            (2016, 4, False, False, False),
            (2080, 4, False, False, False),
            (2048, 4, True, False, False),
            (2048, 4, True, True, False),
            (2048, 8, False, False, False),
            (2048, 4, False, False, True),
        )
    ]
for index, (m, k, n, bits, bias, explicit_zero, fp32) in enumerate(cases):
    rng = np.random.default_rng(9100 + index)
    dtype = np.float32 if fp32 else np.float16
    proto_dtype = TensorProto.FLOAT if fp32 else TensorProto.FLOAT16
    x = rng.uniform(-.25, .25, (m, k)).astype(dtype)
    if fp32:
        x = rng.integers(-127, 128, (m, k)).astype(np.float32) / 512
        x[:, ::128] = 127 / 512
    codes = rng.integers(0, 16, (n, k), dtype=np.uint8)
    if bits == 8:
        codes += 120
    scales = np.exp2(rng.integers(-11, -8, (n, k // 32))).astype(dtype)
    zeros = rng.integers(4, 12, (n, k // 32), dtype=np.uint8) if explicit_zero else np.full(
        (n, k // 32), 8 if bits == 4 else 128, np.uint8,
    )
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)) if bits == 4 else codes
    packed = packed.reshape(n, k // 32, 32 * bits // 8)
    weights = (codes.astype(np.float64) - np.repeat(zeros, 32, axis=1)) * np.repeat(scales, 32, axis=1)
    expected = x.astype(np.float64) @ weights.T
    tensors = [numpy_helper.from_array(x, "x"), numpy_helper.from_array(packed, "weights"),
               numpy_helper.from_array(scales, "scales")]
    inputs = ["x", "weights", "scales"]
    if explicit_zero:
        packed_zeros = zeros[:, ::2] | (zeros[:, 1::2] << 4)
        tensors.append(numpy_helper.from_array(packed_zeros, "zeros"))
        inputs.append("zeros")
    elif bias:
        inputs.append("")
    if bias:
        values = rng.uniform(-.01, .01, n).astype(dtype)
        tensors.append(numpy_helper.from_array(values, "bias"))
        inputs.append("")
        inputs.append("bias")
        expected += values
    expected = expected.astype(dtype)
    name = f"dense_m{m}_k{k}_n{n}_bits{bits}_bias{int(bias)}_zp{int(explicit_zero)}_f32{int(fp32)}"
    output = helper.make_tensor_value_info("y", proto_dtype, [m, n])
    node = helper.make_node("MatMulNBits", inputs, ["y"], domain="com.microsoft",
                           K=k, N=n, bits=bits, block_size=32, accuracy_level=4)
    model = helper.make_model(helper.make_graph([node], name, [], [output], tensors),
                              opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, root / f"{name}.onnx")
    reference = helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["expected"], ["y"])], "reference", [], [output],
        [numpy_helper.from_array(expected, "expected")]), opset_imports=[helper.make_opsetid("", 22)])
    reference.ir_version = 10
    onnx.save(reference, root / f"{name}_reference.onnx")
source = Path("/home/sayanshaw/repos/qwen38-webgpu-raw")
graph_path = source / "model.onnx"
if args.skinny:
    source = Path(__file__).resolve().parent / "dense-tail-fixtures"
    graph_path = source / "dense_actual_hyper_m1.onnx"
graph = onnx.load(graph_path, load_external_data=False)
original_node = next(node for node in graph.graph.node
                     if node.op_type == "MatMulNBits" and "attn_hyper_connection/input_mix_down_block_inject" in node.name)
initializers = {tensor.name: tensor for tensor in graph.graph.initializer}
originals = []
for name in original_node.input[1:]:
    if not name:
        continue
    tensor = TensorProto()
    tensor.CopyFrom(initializers[name])
    if tensor.external_data:
        load_external_data_for_tensor(tensor, str(source))
        tensor.ClearField("external_data")
        tensor.data_location = TensorProto.DEFAULT
    originals.append(tensor)
arrays = [numpy_helper.to_array(tensor) for tensor in originals]
n, blocks, _ = arrays[0].shape
k = blocks * 32
packed = arrays[0].reshape(n, k // 2)
codes = np.empty((n, k), np.uint8)
codes[:, ::2], codes[:, 1::2] = packed & 15, packed >> 4
weights = (codes.astype(np.float64) - 8) * np.repeat(arrays[1], 32, axis=1)
actual_rows = (1, 2, 4, 5, 16) if args.skinny else (1, 16)
for m in actual_rows:
    bound = .03125 if m in (1, 5) else .25
    x = np.random.default_rng(9500 + m).uniform(-bound, bound, (m, k)).astype(np.float16)
    expected = (x.astype(np.float64) @ weights.T).astype(np.float16)
    name = f"dense_actual_hyper_m{m}"
    node = onnx.NodeProto()
    node.CopyFrom(original_node)
    node.input[0], node.output[0] = "x", "y"
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [m, n])
    model = helper.make_model(helper.make_graph(
        [node], name, [], [output], [*originals, numpy_helper.from_array(x, "x")]),
        opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, root / f"{name}.onnx")
    reference = helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["expected"], ["y"])], "reference", [], [output],
        [numpy_helper.from_array(expected, "expected")]), opset_imports=[helper.make_opsetid("", 22)])
    reference.ir_version = 10
    onnx.save(reference, root / f"{name}_reference.onnx")
print(f"Prepared {len(cases)} independent and {len(actual_rows)} original-weight fixtures in {root}")
