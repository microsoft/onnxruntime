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
from onnxruntime.quantization.utils import update_tensor_metadata_for_permutation


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

class TestUpdateTensorMetadataForPermutation(unittest.TestCase):
    def _make_dim_value(self, val: int):
        d = onnx.TensorShapeProto.Dimension()
        d.dim_value = val
        return d
    def _make_dim_param(self, name: str):
        d = onnx.TensorShapeProto.Dimension()
        d.dim_param = name
        return d
    def _make_value_info(self, name: str, dims):
        vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        # overwrite with explicit dims to mix value/param
        shape = onnx.TensorShapeProto()
        for d in dims:
            shape.dim.append(d)
        vi.type.tensor_type.shape.CopyFrom(shape)
        return vi
    def test_updates_value_info_input_output(self):
        # Create a graph where the same tensor name B is input, output, and has a value_info
        dims = [self._make_dim_value(2), self._make_dim_value(3)]
        g_in = self._make_value_info("B", dims)
        g_out = self._make_value_info("B", dims)
        vinfo = self._make_value_info("B", dims)
        graph = helper.make_graph(nodes=[], name="g", inputs=[g_in], outputs=[g_out], initializer=[])
        graph.value_info.extend([vinfo])
        pre_counts = (len(graph.value_info), len(graph.input), len(graph.output))
        updated = update_tensor_metadata_for_permutation(graph, "B", [1, 0])
        self.assertEqual(updated, 3)
        for coll in (graph.value_info, graph.input, graph.output):
            vi = next(v for v in coll if v.name == "B")
            self.assertEqual(vi.type.tensor_type.shape.dim[0].dim_value, 3)
            self.assertEqual(vi.type.tensor_type.shape.dim[1].dim_value, 2)
        post_counts = (len(graph.value_info), len(graph.input), len(graph.output))
        self.assertEqual(pre_counts, post_counts)
    def test_preserves_dim_semantics(self):
        # First dim symbolic M, second is concrete 3 -> after perm becomes [3, M]
        dims = [self._make_dim_param("M"), self._make_dim_value(3)]
        vinfo = self._make_value_info("B", dims)
        graph = helper.make_graph([], "g", [], [])
        graph.value_info.extend([vinfo])
        updated = update_tensor_metadata_for_permutation(graph, "B", [1, 0])
        self.assertEqual(updated, 1)
        vi = next(v for v in graph.value_info if v.name == "B")
        self.assertTrue(vi.type.tensor_type.shape.dim[0].HasField("dim_value"))
        self.assertEqual(vi.type.tensor_type.shape.dim[0].dim_value, 3)
        self.assertTrue(vi.type.tensor_type.shape.dim[1].HasField("dim_param"))
        self.assertEqual(vi.type.tensor_type.shape.dim[1].dim_param, "M")
    def test_invalid_perm_strict(self):
        dims = [self._make_dim_value(2), self._make_dim_value(3)]
        vinfo = self._make_value_info("B", dims)
        graph = helper.make_graph([], "g", [], [])
        graph.value_info.extend([vinfo])
        with self.assertRaises(ValueError):
            update_tensor_metadata_for_permutation(graph, "B", [0, 2], strict=True)
        with self.assertRaises(ValueError):
            update_tensor_metadata_for_permutation(graph, "B", [0, 0], strict=True)
        # Non-strict: invalid indices out of range -> no update
        self.assertEqual(update_tensor_metadata_for_permutation(graph, "B", [0, 2], strict=False), 0)
    def test_rank_mismatch_behavior(self):
        # Rank 1 value for B, perm requires rank 2
        dims = [self._make_dim_value(5)]
        vinfo = self._make_value_info("B", dims)
        graph = helper.make_graph([], "g", [], [])
        graph.value_info.extend([vinfo])
        # Non-strict: no-op
        self.assertEqual(update_tensor_metadata_for_permutation(graph, "B", [1, 0], strict=False), 0)
        # Strict: error
        with self.assertRaises(ValueError):
            update_tensor_metadata_for_permutation(graph, "B", [1, 0], strict=True)
    def test_scoped_updates_only(self):
        dims_b = [self._make_dim_value(2), self._make_dim_value(3)]
        dims_c = [self._make_dim_value(4), self._make_dim_value(5)]
        vb = self._make_value_info("B", dims_b)
        vc = self._make_value_info("C", dims_c)
        graph = helper.make_graph([], "g", [], [])
        graph.value_info.extend([vb, vc])
        updated = update_tensor_metadata_for_permutation(graph, "B", [1, 0])
        self.assertEqual(updated, 1)
        b = next(v for v in graph.value_info if v.name == "B")
        c = next(v for v in graph.value_info if v.name == "C")
        self.assertEqual([d.dim_value for d in b.type.tensor_type.shape.dim], [3, 2])
        self.assertEqual([d.dim_value for d in c.type.tensor_type.shape.dim], [4, 5])
    def test_subgraph_update(self):
        # Create a subgraph on its own and update that, verifying parent graph is unchanged.
        dims = [self._make_dim_value(2), self._make_dim_value(3)]
        sub_vi = self._make_value_info("B", dims)
        subgraph = helper.make_graph([], "sub", [], [])
        subgraph.value_info.extend([sub_vi])
        parent_vi = self._make_value_info("B", [self._make_dim_value(2), self._make_dim_value(3)])
        parent = helper.make_graph([], "parent", [], [])
        parent.value_info.extend([parent_vi])
        updated = update_tensor_metadata_for_permutation(subgraph, "B", [1, 0])
        self.assertEqual(updated, 1)
        # subgraph updated
        sv = next(v for v in subgraph.value_info if v.name == "B")
        self.assertEqual([d.dim_value for d in sv.type.tensor_type.shape.dim], [3, 2])
        # parent unchanged
        pv = next(v for v in parent.value_info if v.name == "B")
        self.assertEqual([d.dim_value for d in pv.type.tensor_type.shape.dim], [2, 3])
    def test_repeat_application_safe(self):
        # Applying permutation twice with [1,0] returns to original shape
        dims = [self._make_dim_value(2), self._make_dim_value(3)]
        vinfo = self._make_value_info("B", dims)
        graph = helper.make_graph([], "g", [], [])
        graph.value_info.extend([vinfo])
        update_tensor_metadata_for_permutation(graph, "B", [1, 0])
        vi = next(v for v in graph.value_info if v.name == "B")
        self.assertEqual([d.dim_value for d in vi.type.tensor_type.shape.dim], [3, 2])
        update_tensor_metadata_for_permutation(graph, "B", [1, 0])
        vi2 = next(v for v in graph.value_info if v.name == "B")
        self.assertEqual([d.dim_value for d in vi2.type.tensor_type.shape.dim], [2, 3])


if __name__ == "__main__":
    unittest.main()
