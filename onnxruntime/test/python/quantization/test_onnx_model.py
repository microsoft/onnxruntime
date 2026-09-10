#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import check_op_type_order

from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.tools.onnx_graph_utils import (
    get_children,
    get_parent,
    get_parents,
    input_name_to_nodes,
    output_name_to_node,
)


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """
    Helper function to generate initializers for test inputs
    """
    tensor = np.random.normal(0, 0.3, tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


def construct_model_for_topo_sort(model_path):
    #    (input)
    #       |
    #      GRU
    #      /  \
    #  Conv(1) \
    #     |     \
    #    Relu  Conv(2)
    #     |     |
    #     \     /
    #       Add
    #        |
    #       (output)
    initializers = []
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 8, 12])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 2, 8, 8])

    # make GRU
    initializers.append(generate_input_initializer([2, 24, 12], np.float32, "W_GRU"))
    initializers.append(generate_input_initializer([2, 24, 8], np.float32, "R_GRU"))
    initializers.append(generate_input_initializer([2, 8, 8], np.float32, "H_GRU"))
    gru_node = helper.make_node(
        "GRU",
        ["input", "W_GRU", "R_GRU", "", "", "H_GRU"],
        ["GRU_O"],
        hidden_size=8,
        direction="bidirectional",
    )

    initializers.append(generate_input_initializer([2, 2, 1, 1], np.float32, "W1"))
    initializers.append(generate_input_initializer([2, 2, 1, 1], np.float32, "W2"))
    initializers.append(generate_input_initializer([2], np.float32, "B1"))
    initializers.append(generate_input_initializer([2], np.float32, "B2"))
    conv_node_1 = helper.make_node("Conv", ["GRU_O", "W1", "B1"], ["Conv1_O"], name="Conv1")
    conv_node_2 = helper.make_node("Conv", ["GRU_O", "W2", "B2"], ["Conv2_O"], name="Conv2")
    relu_node = helper.make_node("Relu", ["Conv1_O"], ["Relu_O"], name="Relu")
    add_node = helper.make_node("Add", ["Relu_O", "Conv2_O"], ["output"], name="Add")
    graph = helper.make_graph(
        [conv_node_1, relu_node, conv_node_2, gru_node, add_node],
        "onnx_model_test",
        [input],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, model_path)


def construct_model_for_topo_sort_constant(model_path):
    #    (input)    Constant
    #       \         /
    #        \       /
    #         \     /
    #          \   /
    #           Add
    #            |
    #         (output)

    initializers = []
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 8, 12])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 8, 12])

    # make nodes
    constant_node = helper.make_node("Constant", [], ["const_output"], value_float=42.0)
    add_node = helper.make_node("Add", ["input", "const_output"], ["output"], name="Add")
    graph = helper.make_graph(
        [add_node, constant_node],
        "onnx_model_test",
        [input],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, model_path)


def construct_model_for_topo_sort_empty_input_output(model_path):
    #    (input1)    (input2)
    #       |           |
    #      Op1         Op1
    #       \         /
    #        \       /
    #         \     /
    #          \   /
    #           Op2
    #            |
    #           Op3
    #            |
    #         (output)

    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [4, 8, 12])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [4, 8, 12])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 8, 12])

    # make nodes
    op1_node_1 = helper.make_node("Op1", ["input1"], ["", "", "Op1_1_output"], name="op1_1", domain="Test")
    op1_node_2 = helper.make_node("Op1", ["input2"], ["", "", "Op1_2_output"], name="op1_2", domain="Test")
    op2_node = helper.make_node("Op2", ["Op1_1_output", "Op1_2_output"], ["op2_output"], name="op2", domain="Test")
    op3_node = helper.make_node("Op3", ["", "op2_output"], ["output"], name="op3", domain="Test")
    graph = helper.make_graph(
        [op1_node_1, op1_node_2, op3_node, op2_node],
        "onnx_model_topo_test",
        [input1, input2],
        [output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("Test", 1), helper.make_opsetid("", 13)])
    onnx.save(model, model_path)


class TestOnnxGraphUtils(unittest.TestCase):
    def test_name_maps_ignore_empty_names_and_preserve_order(self):
        first = helper.make_node("First", ["input", ""], ["shared", "", "first_output"], name="first")
        second = helper.make_node("Second", ["shared", "shared"], ["shared"], name="second")

        input_map = input_name_to_nodes([first, second])
        self.assertNotIn("", input_map)
        self.assertEqual([node.name for node in input_map["shared"]], ["second", "second"])

        output_map = output_name_to_node([first, second])
        self.assertNotIn("", output_map)
        self.assertIs(output_map["shared"], second)
        self.assertIs(output_map["first_output"], first)

    def test_children_preserve_order_duplicates_and_output_selection(self):
        parent = helper.make_node("Parent", [], ["first", "", "second"], name="parent")
        first_child = helper.make_node("FirstChild", ["second", "first"], ["first_child"], name="first_child")
        second_child = helper.make_node(
            "SecondChild",
            ["first", "", "first"],
            ["second_child"],
            name="second_child",
        )
        input_map = input_name_to_nodes([first_child, second_child])

        self.assertEqual(
            [node.name for node in get_children(parent, input_map)],
            ["first_child", "second_child", "second_child", "first_child"],
        )
        self.assertEqual(
            [node.name for node in get_children(parent, input_map, output_index=0)],
            ["first_child", "second_child", "second_child"],
        )
        self.assertEqual(get_children(parent, input_map, output_index=1), [])
        self.assertEqual([node.name for node in get_children(parent, input_map, output_index=2)], ["first_child"])
        self.assertEqual(get_children(parent, input_map, output_index=3), [])

    def test_parents_preserve_duplicates_and_handle_missing_inputs(self):
        first_parent = helper.make_node("FirstParent", [], ["first"], name="first_parent")
        second_parent = helper.make_node("SecondParent", [], ["second"], name="second_parent")
        child = helper.make_node(
            "Child",
            ["second", "missing", "", "first", "first"],
            ["child"],
            name="child",
        )
        output_map = output_name_to_node([first_parent, second_parent])

        self.assertEqual(
            [node.name for node in get_parents(child, output_map)],
            ["second_parent", "first_parent", "first_parent"],
        )
        self.assertIs(get_parent(child, 0, output_map), second_parent)
        self.assertIsNone(get_parent(child, 1, output_map))
        self.assertIsNone(get_parent(child, 2, output_map))
        self.assertIs(get_parent(child, 3, output_map), first_parent)
        self.assertIsNone(get_parent(child, 5, output_map))


class TestONNXModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_onnx_model.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def test_topo_sort(self):
        test_model_path = str(Path(self._tmp_model_dir.name) / "onnx_model_topo_sort.onnx")
        construct_model_for_topo_sort(test_model_path)
        onnx_model = ONNXModel(onnx.load(test_model_path))
        check_op_type_order(self, onnx_model.model, ["Conv", "Relu", "Conv", "GRU", "Add"])
        onnx_model.topological_sort()
        check_op_type_order(self, onnx_model.model, ["GRU", "Conv", "Conv", "Relu", "Add"])

    def test_topo_sort_constant(self):
        test_model_path = str(Path(self._tmp_model_dir.name) / "onnx_model_topo_sort_constant.onnx")
        construct_model_for_topo_sort_constant(test_model_path)
        onnx_model = ONNXModel(onnx.load(test_model_path))
        check_op_type_order(self, onnx_model.model, ["Add", "Constant"])
        onnx_model.topological_sort()
        check_op_type_order(self, onnx_model.model, ["Constant", "Add"])

    def test_topo_sort_empty_input_output(self):
        test_model_path = str(Path(self._tmp_model_dir.name) / "onnx_model_topo_empty_input_output.onnx")
        construct_model_for_topo_sort_empty_input_output(test_model_path)
        onnx_model = ONNXModel(onnx.load(test_model_path))
        check_op_type_order(self, onnx_model.model, ["Op1", "Op1", "Op3", "Op2"])
        onnx_model.topological_sort()
        check_op_type_order(self, onnx_model.model, ["Op1", "Op1", "Op2", "Op3"])

    def test_navigation_searches_only_main_graph(self):
        subgraph_node = helper.make_node("Identity", ["subgraph_input", ""], ["subgraph_output", ""], name="subgraph")
        subgraph = helper.make_graph(
            [subgraph_node],
            "subgraph",
            [],
            [helper.make_tensor_value_info("subgraph_output", TensorProto.FLOAT, [1])],
        )
        main_node = helper.make_node(
            "If",
            ["condition", ""],
            ["main_output", ""],
            name="main",
            then_branch=subgraph,
            else_branch=subgraph,
        )
        graph = helper.make_graph(
            [main_node],
            "main_graph",
            [helper.make_tensor_value_info("condition", TensorProto.BOOL, [])],
            [helper.make_tensor_value_info("main_output", TensorProto.FLOAT, [1])],
        )
        onnx_model = ONNXModel(helper.make_model(graph))

        input_map = onnx_model.input_name_to_nodes()
        output_map = onnx_model.output_name_to_node()
        self.assertEqual(set(input_map), {"condition"})
        self.assertEqual(set(output_map), {"main_output"})
        self.assertEqual(input_map["condition"][0].name, main_node.name)
        self.assertEqual(output_map["main_output"].name, main_node.name)


if __name__ == "__main__":
    unittest.main()
