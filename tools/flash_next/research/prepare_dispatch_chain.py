from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


root = Path(__file__).resolve().parent
out = root / "dispatch-chain-fixtures"
out.mkdir(exist_ok=True)
for rows in (1, 16):
    source = onnx.load(root / "dense-tail-fixtures" / f"dense_actual_hyper_m{rows}.onnx")
    reference = onnx.load(root / "dense-tail-fixtures" / f"dense_actual_hyper_m{rows}_reference.onnx")
    projection = numpy_helper.to_array(reference.graph.initializer[0]).copy()
    for layers in (32, 256):
        tensors = list(source.graph.initializer)
        tensors += [numpy_helper.from_array(np.array(4, np.float16), "four"),
                    numpy_helper.from_array(np.array(2, np.float16), "two"),
                    numpy_helper.from_array(np.zeros_like(projection), "initial")]
        nodes = []
        previous = "initial"
        expected = np.zeros_like(projection)
        for layer in range(layers):
            node = onnx.NodeProto()
            node.CopyFrom(source.graph.node[0])
            node.name = f"projection_{layer}"
            node.output[0] = f"projected_{layer}"
            nodes += [
                node,
                helper.make_node("Add", [previous, node.output[0]], [f"add_{layer}"]),
                helper.make_node("Div", [f"add_{layer}", "four"], [f"div_{layer}"]),
                helper.make_node("Sigmoid", [f"div_{layer}"], [f"sigmoid_{layer}"]),
                helper.make_node("Mul", [f"sigmoid_{layer}", "two"], [f"state_{layer}"]),
            ]
            expected = (expected + projection).astype(np.float16)
            expected = (expected / np.float16(4)).astype(np.float16)
            expected = (1 / (1 + np.exp(-expected.astype(np.float32)))).astype(np.float16)
            expected = (expected * np.float16(2)).astype(np.float16)
            previous = f"state_{layer}"
        nodes.append(helper.make_node("Identity", [previous], ["y"]))
        output = helper.make_tensor_value_info("y", TensorProto.FLOAT16, list(expected.shape))
        name = f"chain_m{rows}_layers{layers}"
        model = helper.make_model(helper.make_graph(nodes, name, [], [output], tensors),
                                  opset_imports=list(source.opset_import))
        model.ir_version = 10
        onnx.checker.check_model(model)
        onnx.save(model, out / f"{name}.onnx")
        reference = helper.make_model(helper.make_graph(
            [helper.make_node("Identity", ["expected"], ["y"])], "reference", [], [output],
            [numpy_helper.from_array(expected, "expected")]),
            opset_imports=[helper.make_opsetid("", 22)])
        reference.ir_version = 10
        onnx.save(reference, out / f"{name}_reference.onnx")
        print(name, "dispatches", layers * 5 + 1)

source = np.random.default_rng(37).uniform(-1, 1, 512).astype(np.float32)
tensors = [numpy_helper.from_array(source, "source"),
           numpy_helper.from_array(np.zeros(64, np.float32), "initial")]
nodes = []
expected = np.zeros(64, np.float32)
previous = "initial"
for layer in range(320):
    start = layer * 7 % 449
    tensors += [numpy_helper.from_array(np.array([start], np.int64), f"start_{layer}"),
                numpy_helper.from_array(np.array([start + 64], np.int64), f"end_{layer}")]
    nodes += [
        helper.make_node("Slice", ["source", f"start_{layer}", f"end_{layer}"], [f"slice_{layer}"]),
        helper.make_node("Add", [previous, f"slice_{layer}"], [f"state_{layer}"]),
    ]
    expected = (expected + source[start:start + 64]).astype(np.float32)
    previous = f"state_{layer}"
nodes.append(helper.make_node("Identity", [previous], ["y"]))
output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [64])
model = helper.make_model(helper.make_graph(nodes, "uniform_offsets", [], [output], tensors),
                          opset_imports=[helper.make_opsetid("", 22)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, out / "uniform_offsets.onnx")
reference = helper.make_model(helper.make_graph(
    [helper.make_node("Identity", ["expected"], ["y"])], "reference", [], [output],
    [numpy_helper.from_array(expected, "expected")]),
    opset_imports=[helper.make_opsetid("", 22)])
reference.ir_version = 10
onnx.save(reference, out / "uniform_offsets_reference.onnx")
print("uniform_offsets", "nominal_compute_nodes", len(nodes))
