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


class TestReplaceGemmWithMatmul(unittest.TestCase):
    def test_replace_gemm_with_matmul_trans_b_initializer_metadata_updated(self):
        # Build minimal Gemm with transB=1 and B as initializer with value_info
        a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 2])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        weight = numpy_helper.from_array(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), name="B"
        )
        # ValueInfo for B with original dims [2, 3]
        b_vi = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])
        gemm = helper.make_node("Gemm", ["A", "B"], ["Y"], transB=1, alpha=1.0, beta=1.0, name="Gemm0")
        graph = helper.make_graph([gemm], "g", [a], [y], initializer=[weight])
        graph.value_info.extend([b_vi])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        onnx_model = ONNXModel(model)
        onnx_model.replace_gemm_with_matmul()

        # Confirm Gemm was replaced by MatMul
        ops = [n.op_type for n in onnx_model.model.graph.node]
        assert "MatMul" in ops and "Gemm" not in ops

        # Initializer B should now have transposed dims [3, 2]
        b_init = next(i for i in onnx_model.model.graph.initializer if i.name == "B")
        assert list(b_init.dims) == [3, 2]

        # ValueInfo for B should be updated to [3, 2]
        b_vi = next(vi for vi in onnx_model.model.graph.value_info if vi.name == "B")
        assert [d.dim_value for d in b_vi.type.tensor_type.shape.dim] == [3, 2]

        # Shape inference should succeed without mismatch
        onnx.shape_inference.infer_shapes(onnx_model.model)

    def test_replace_gemm_with_matmul_transb_dynamic_b_inserts_transpose(self):
        # B is a graph input (not initializer) so a Transpose should be inserted
        a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 2])
        b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        gemm = helper.make_node("Gemm", ["A", "B"], ["Y"], transB=1, alpha=1.0, beta=1.0, name="Gemm0")
        graph = helper.make_graph([gemm], "g", [a, b], [y], initializer=[])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        onnx_model = ONNXModel(model)
        onnx_model.replace_gemm_with_matmul()

        ops = [n.op_type for n in onnx_model.model.graph.node]
        assert "Transpose" in ops and "MatMul" in ops and "Gemm" not in ops

        # Shape inference should succeed
        onnx.shape_inference.infer_shapes(onnx_model.model)


if __name__ == "__main__":
    unittest.main()
