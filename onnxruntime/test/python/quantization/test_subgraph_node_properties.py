# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

import onnxruntime as ort
from onnxruntime.quantization.onnx_model import ONNXModel

DOMAIN = "test.quantization.functions"


def _branch(name, weight):
    return helper.make_graph(
        [helper.make_node("Gemm", ["x", "weight"], [name], name=name + "_gemm")],
        name,
        [],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])],
        [
            numpy_helper.from_array(np.asarray(weight, dtype=np.float32), "weight"),
            numpy_helper.from_array(np.asarray([17], dtype=np.float32), "unused"),
        ],
    )


def _function_model(overload=""):
    select = helper.make_node("If", ["condition"], ["y"])
    for name in ("then_branch", "else_branch"):
        attribute = AttributeProto()
        attribute.name = name
        attribute.type = AttributeProto.GRAPH
        attribute.ref_attr_name = name
        select.attribute.append(attribute)
    function = helper.make_function(
        DOMAIN,
        "Conditional",
        ["condition", "x"],
        ["y"],
        [select],
        [helper.make_opsetid("", 17)],
        attributes=["then_branch", "else_branch"],
        overload=overload,
    )
    node = helper.make_node(
        "Conditional",
        ["condition", "x"],
        ["y"],
        name="function_call",
        domain=DOMAIN,
        overload=overload,
        doc_string="Preserve this custom function dispatch and provenance.",
        then_branch=_branch("positive", [[2, 0], [0, 3]]),
        else_branch=_branch("negative", [[-1, 0], [0, -2]]),
    )
    node.metadata_props.add(key="origin", value="exported-layer")
    node.metadata_props.add(key="scope", value="vision/block")
    graph = helper.make_graph(
        [node],
        "function_graph",
        [
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
    )
    return helper.make_model(
        graph,
        functions=[function],
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(DOMAIN, 1)],
        ir_version=10,
    )


def _properties(node):
    result = copy.deepcopy(node)
    for attribute in result.attribute:
        if attribute.type == AttributeProto.GRAPH:
            attribute.ClearField("g")
        elif attribute.type == AttributeProto.GRAPHS:
            attribute.ClearField("graphs")
    return result.SerializeToString()


def _apply(model, operation):
    wrapped = ONNXModel(model)
    getattr(wrapped, operation)()
    return wrapped.model


def _infer(model, condition=True):
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session.run(
        None,
        {"condition": np.asarray(condition), "x": np.asarray([[2, 5]], dtype=np.float32)},
    )[0]


class TestSubgraphNodeProperties(unittest.TestCase):
    def test_custom_function_dispatch_and_inference_survive(self):
        for operation in ("clean_initializers", "replace_gemm_with_matmul"):
            for condition in (False, True):
                with self.subTest(operation=operation, condition=condition):
                    model = _function_model()
                    expected = np.asarray([[4, 15] if condition else [-2, -10]], dtype=np.float32)
                    np.testing.assert_array_equal(_infer(model, condition), expected)
                    _apply(model, operation)
                    self.assertEqual(model.graph.node[0].domain, DOMAIN)
                    np.testing.assert_array_equal(_infer(model, condition), expected)

    def test_overload_survives_save_and_reload(self):
        for operation in ("clean_initializers", "replace_gemm_with_matmul"):
            with self.subTest(operation=operation):
                model = _function_model(overload="float32")
                expected = _infer(model)
                _apply(model, operation)
                self.assertEqual(model.graph.node[0].overload, "float32")
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.onnx"
                    onnx.save_model(model, path)
                    restored = onnx.load_model(path)
                self.assertEqual(restored.graph.node[0].overload, "float32")
                np.testing.assert_array_equal(_infer(restored), expected)

    def test_all_non_graph_node_properties_are_preserved(self):
        for operation in ("clean_initializers", "replace_gemm_with_matmul"):
            with self.subTest(operation=operation):
                model = _function_model(overload="float32")
                node = model.graph.node[0]
                node.attribute[0].doc_string = "branch attribute documentation"
                expected = _properties(node)
                _apply(model, operation)
                self.assertEqual(_properties(model.graph.node[0]), expected)

    def test_nested_standard_if_nodes_keep_metadata(self):
        for operation in ("clean_initializers", "replace_gemm_with_matmul"):
            with self.subTest(operation=operation):
                model = _function_model()
                custom = model.graph.node[0]
                inner = helper.make_node(
                    "If",
                    ["condition"],
                    ["nested_y"],
                    name="inner_if",
                    doc_string="inner condition",
                    then_branch=_branch("inner_positive", [[2, 0], [0, 3]]),
                    else_branch=_branch("inner_negative", [[-1, 0], [0, -2]]),
                )
                inner.metadata_props.add(key="layer", value="inner")
                nested = helper.make_graph(
                    [inner], "nested", [], [helper.make_tensor_value_info("nested_y", TensorProto.FLOAT, [1, 2])]
                )
                outer = helper.make_node(
                    "If",
                    ["condition"],
                    ["y"],
                    name="outer_if",
                    doc_string="outer condition",
                    then_branch=nested,
                    else_branch=copy.deepcopy(custom.attribute[0].g),
                )
                outer.metadata_props.add(key="layer", value="outer")
                model.graph.ClearField("node")
                model.graph.node.append(outer)
                expected = _infer(model)
                expected_fields = {
                    "outer_if": _properties(outer),
                    "inner_if": _properties(inner),
                }
                _apply(model, operation)
                graphs = [model.graph]
                seen = set()
                while graphs:
                    graph = graphs.pop()
                    for node in graph.node:
                        if node.name in expected_fields:
                            self.assertEqual(_properties(node), expected_fields[node.name])
                            seen.add(node.name)
                        graphs.extend(attr.g for attr in node.attribute if attr.type == AttributeProto.GRAPH)
                self.assertEqual(seen, set(expected_fields))
                np.testing.assert_array_equal(_infer(model), expected)

    def test_repeated_graph_attributes_keep_the_original_node(self):
        for operation in ("clean_initializers", "replace_gemm_with_matmul"):
            with self.subTest(operation=operation):
                node = helper.make_node(
                    "GraphContainer",
                    ["x"],
                    ["y"],
                    name="container",
                    domain=DOMAIN,
                    doc_string="multiple graph alternatives",
                    overload="variant",
                    branches=[
                        _branch("first", [[2, 0], [0, 3]]),
                        _branch("second", [[-1, 0], [0, -2]]),
                    ],
                    selection=2,
                    label="source",
                )
                node.metadata_props.add(key="exporter", value="test")
                model = helper.make_model(helper.make_graph([node], "structural", [], []))
                expected = _properties(node)
                _apply(model, operation)
                result = model.graph.node[0]
                self.assertEqual(_properties(result), expected)
                branches = next(attr.graphs for attr in result.attribute if attr.name == "branches")
                self.assertEqual(len(branches), 2)
                for graph in branches:
                    if operation == "clean_initializers":
                        self.assertEqual([tensor.name for tensor in graph.initializer], ["weight"])
                    else:
                        self.assertEqual([child.op_type for child in graph.node], ["MatMul"])

    def test_graph_contents_still_receive_the_requested_transformation(self):
        for operation in ("clean_initializers", "replace_gemm_with_matmul"):
            with self.subTest(operation=operation):
                model = _function_model()
                _apply(model, operation)
                for attribute in model.graph.node[0].attribute:
                    graph = attribute.g
                    if operation == "clean_initializers":
                        self.assertEqual([tensor.name for tensor in graph.initializer], ["weight"])
                        self.assertEqual(graph.node[0].op_type, "Gemm")
                    else:
                        self.assertEqual([child.op_type for child in graph.node], ["MatMul"])
                np.testing.assert_array_equal(_infer(model), np.asarray([[4, 15]], dtype=np.float32))

    def test_plain_node_without_subgraphs_is_unchanged(self):
        for operation in ("clean_initializers", "replace_gemm_with_matmul"):
            with self.subTest(operation=operation):
                node = helper.make_node("Identity", ["x"], ["y"], name="identity", doc_string="plain")
                node.metadata_props.add(key="origin", value="plain")
                model = helper.make_model(
                    helper.make_graph(
                        [node],
                        "plain",
                        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
                        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
                    ),
                    opset_imports=[helper.make_opsetid("", 17)],
                    ir_version=10,
                )
                expected = node.SerializeToString()
                _apply(model, operation)
                self.assertEqual(model.graph.node[0].SerializeToString(), expected)
                onnx.checker.check_model(model)


if __name__ == "__main__":
    unittest.main()
