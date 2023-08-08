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
from op_test_utils import (
    TestDataFeeds,
    check_model_correctness,
    check_op_type_count,
    check_op_type_order,
    create_clip_node,
)

from onnxruntime.quantization import QDQQuantizer, QuantFormat, QuantizationMode, QuantType, quantize_static


class TestQDQFormat(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr


class TestQDQExtraOptions(unittest.TestCase):
    def test_qdq_extra_options(self):
        #   (input)
        #      |
        #     Add
        #      |
        #     ReduceMean
        #      |
        #     Add
        #      |
        #   (output)

        initializers = []

        input_tensor = helper.make_tensor_value_info("L", TensorProto.FLOAT, [5, 5])
        output_tensor = helper.make_tensor_value_info("O", TensorProto.FLOAT, [5, 5])

        add_weight_data_1 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(add_weight_data_1, name="M"))
        add_weight_data_2 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(add_weight_data_2, name="N"))

        add_node_1 = onnx.helper.make_node("Add", ["L", "M"], ["P"], name="Add1")
        reduce_mean_node = onnx.helper.make_node("ReduceMean", ["P"], ["Q"], keepdims=1, name="ReduceMean")
        add_node_2 = onnx.helper.make_node("Add", ["Q", "N"], ["O"], name="Add2")

        graph = helper.make_graph(
            [add_node_1, reduce_mean_node, add_node_2],
            "QDQ_Test_Finetune",
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        test_model_path = "./test_qdq_finetune.onnx"
        onnx.save(model, test_model_path)

        compute_range = {
            "P": [0.1, 0.1],
            "Q": [0.1, 0.1],
            "M": [0.1, 0.1],
            "N": [0.1, 0.1],
            "L": [0.1, 0.1],
            "O": [0.1, 0.1],
        }

        op_types_to_quantize = ["Add"]

        mode = QuantizationMode.QLinearOps
        model = onnx.load_model(test_model_path)
        quantizer = QDQQuantizer(
            model,
            True,  # per_channel
            False,  # reduce_range
            mode,
            True,  # static
            QuantType.QInt8,  # weight_type
            QuantType.QInt8,  # activation_type
            compute_range,
            [],  # nodes_to_quantize
            ["Add2"],  # nodes_to_exclude
            op_types_to_quantize,
            {
                "ActivationSymmetric": True,
                "AddQDQPairToWeight": True,
                "OpTypesToExcludeOutputQuantization": [],
            },
        )  # extra_options
        quantizer.quantize_model()
        qdq_model_path = "./test_qdq_finetune_qdq.onnx"
        quantizer.model.save_model_to_file(qdq_model_path, False)

        # QDQ pair should be added to Add1 but not Add2
        # QDQ pair shoud be added to Add1 output as well.
        qdq_added_to_node_output_flag = False
        for node in quantizer.model.nodes():
            if node.name == "Add1":
                for input in node.input:
                    self.assertTrue("DequantizeLinear" in input)
                for output in node.output:
                    self.assertTrue("QuantizeLinear" not in output)

            if node.name == "Add2":
                for input in node.input:
                    self.assertTrue("DequantizeLinear" not in input)
                for output in node.output:
                    self.assertTrue("QuantizeLinear" not in output)

            # This QuantizeLinear node should be followed by Add1
            if node.name == "P_QuantizeLinear":
                qdq_added_to_node_output_flag = True
                self.assertTrue(node.input[0] == "P")

        self.assertTrue(qdq_added_to_node_output_flag)

    def test_qdq_extra_options_2(self):
        #         (input)
        #           |
        #          Add
        #       /   |   \
        #  MatMul MatMul MatMul
        #     |     |      |
        # (output)(output)(output)

        initializers = []

        input_tensor = helper.make_tensor_value_info("L", TensorProto.FLOAT, [5, 5])
        output_tensor1 = helper.make_tensor_value_info("M", TensorProto.FLOAT, [5, 5])
        output_tensor2 = helper.make_tensor_value_info("N", TensorProto.FLOAT, [5, 5])
        output_tensor3 = helper.make_tensor_value_info("O", TensorProto.FLOAT, [5, 5])

        add_weight_data = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(add_weight_data, name="P"))
        matmul_weight_data_1 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data_1, name="Q"))
        matmul_weight_data_2 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data_2, name="R"))
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data_2, name="S"))

        add_node = onnx.helper.make_node("Add", ["L", "P"], ["T"], name="Add")
        matmul_node_1 = onnx.helper.make_node("MatMul", ["T", "Q"], ["M"], name="MatMul1")
        matmul_node_2 = onnx.helper.make_node("MatMul", ["T", "R"], ["N"], name="MatMul2")
        matmul_node_3 = onnx.helper.make_node("MatMul", ["T", "S"], ["O"], name="MatMul3")

        graph = helper.make_graph(
            [add_node, matmul_node_1, matmul_node_2, matmul_node_3],
            "QDQ_Test_Finetune_2",
            [input_tensor],
            [output_tensor1, output_tensor2, output_tensor3],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        test_model_path = "./test_qdq_finetune_2.onnx"
        onnx.save(model, test_model_path)

        compute_range = {
            "L": [0.1, 0.1],
            "M": [0.1, 0.1],
            "N": [0.1, 0.1],
            "O": [0.1, 0.1],
            "P": [0.1, 0.1],
            "Q": [0.1, 0.1],
            "R": [0.1, 0.1],
            "S": [0.1, 0.1],
            "T": [0.1, 0.1],
        }

        op_types_to_quantize = ["Add", "MatMul"]

        mode = QuantizationMode.QLinearOps
        model = onnx.load_model(test_model_path)
        quantizer = QDQQuantizer(
            model,
            True,  # per_channel
            False,  # reduce_range
            mode,
            True,  # static
            QuantType.QInt8,  # weight_type
            QuantType.QInt8,  # activation_type
            compute_range,
            [],  # nodes_to_quantize
            ["Add"],  # nodes_to_exclude
            op_types_to_quantize,
            {
                "ActivationSymmetric": True,
                "AddQDQPairToWeight": True,
                "OpTypesToExcludeOutputQuantization": op_types_to_quantize,
                "DedicatedQDQPair": True,
            },
        )  # extra_options
        quantizer.quantize_model()
        qdq_model_path = "./test_qdq_finetune_qdq_2.onnx"
        quantizer.model.save_model_to_file(qdq_model_path, False)

        # Three dedicated QDQ pair should be generated and feed into each MatMul node
        # Also QDQ pair should not be added to Add node
        # QDQ pair shoud not be added to node's output
        for node in quantizer.model.nodes():
            if node.name == "MatMul1":
                self.assertIn("T_DequantizeLinear_Output_1", node.input)
            elif node.name == "MatMul2":
                self.assertIn("T_DequantizeLinear_Output_2", node.input)
            elif node.name == "MatMul3":
                self.assertIn("T_DequantizeLinear_Output_3", node.input)
            elif node.name == "Add":
                self.assertNotIn("L_DequantizeLinear_Output", node.input)

            # QDQ pair shoud not be added to MatMul's output
            if node.op_type == "QuantizeLinear":
                self.assertNotIn(
                    node.input[0],
                    {
                        "M_QuantizeLinear_Input",
                        "N_QuantizeLinear_Input",
                        "O_QuantizeLinear_Input",
                    },
                )


class TestQDQFormatConv(TestQDQFormat):
    def check_per_channel_counts(self, model_path, channel_count: int, axis: int = 0):
        model = onnx.load(Path(model_path))
        for initializer in model.graph.initializer:
            dims = initializer.dims
            # skip if initializer is not a weight
            if len(dims) > 0:
                self.assertGreater(len(dims), axis)
                self.assertEqual(channel_count, dims[axis])

    def construct_model_conv(self, output_model_path, input_shape, weight_shape, output_shape, has_bias):
        #    (input)
        #      |
        #     Conv
        #      |
        #    (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        # make Conv node
        weight_name = "conv_weight"
        bias_name = "conv_bias"
        conv_inputs = [input_name, weight_name]
        conv_outputs = [output_name]
        conv_name = "conv_node"
        conv_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(conv_weight_data, name=weight_name))
        if has_bias:
            conv_inputs.append(bias_name)
            bias_data = np.random.normal(0, 0.05, (weight_shape[0])).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))
        conv_node = onnx.helper.make_node("Conv", conv_inputs, conv_outputs, name=conv_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = "QDQ_Test_Conv"
        graph = helper.make_graph(
            [conv_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def verify_quantize_conv(self, has_bias, per_channel, is_quant_type_int8=False):
        np.random.seed(1)
        model_fp32_path = f"conv_fp32.{has_bias}.{per_channel}.onnx"
        model_int8_qdq_path = f"conv_quant_qdq.{has_bias}.{per_channel}.onnx"
        model_int8_qop_path = f"conv_quant_qop.{has_bias}.{per_channel}.onnx"
        channel_count = 16
        data_reader = self.input_feeds(1, {"input": [1, 8, 33, 33]})
        self.construct_model_conv(model_fp32_path, [1, 8, 33, 33], [channel_count, 8, 3, 3], [1, 16, 31, 31], has_bias)
        quantize_static(
            model_fp32_path,
            model_int8_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            reduce_range=per_channel,
            activation_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
            weight_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
        )
        data_reader.rewind()
        qdq_nodes = {
            "Conv": 1,
            "QuantizeLinear": 2,
            "DequantizeLinear": 4 if has_bias else 3,
        }
        check_op_type_count(self, model_int8_qdq_path, **qdq_nodes)
        if per_channel:
            self.check_per_channel_counts(model_int8_qdq_path, channel_count)
        check_model_correctness(self, model_fp32_path, model_int8_qdq_path, data_reader.get_next())

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_qop_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            per_channel=per_channel,
            reduce_range=per_channel,
            activation_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
            weight_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
        )
        data_reader.rewind()
        qop_nodes = {"QLinearConv": 1, "QuantizeLinear": 1, "DequantizeLinear": 1}
        check_op_type_count(self, model_int8_qop_path, **qop_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_qop_path, data_reader.get_next())

    def test_quantize_conv_without_bias(self):
        # only test cases per_channel=True and reduce_range=True to avoid saturation on avx2 and avx512 for weight type int8
        self.verify_quantize_conv(False, True, True)  # has_bias:False, per_channel:True, is_quant_type_int8:True
        self.verify_quantize_conv(True, True, True)  # has_bias:True, per_channel:True, is_quant_type_int8:True

        self.verify_quantize_conv(False, False, False)  # has_bias:False, per_channel:False, is_quant_type_int8:False
        self.verify_quantize_conv(True, False, False)  # has_bias:True, per_channel:False, is_quant_type_int8:False
        self.verify_quantize_conv(False, True, False)  # has_bias:False, per_channel:True, is_quant_type_int8:False
        self.verify_quantize_conv(True, True, False)  # has_bias:True, per_channel:True, is_quant_type_int8:False


class TestQDQFormatConvClip(TestQDQFormat):
    def construct_model_conv_clip(self, output_model_path, input_shape, weight_shape, output_shape):
        #    (input)
        #      |
        #     Conv
        #      |
        #     Clip
        #      |
        #   Reshape
        #      |
        #    (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        # make Conv node
        weight_name = "conv_weight"
        conv_inputs = [input_name, weight_name]
        conv_outputs = ["conv_output"]
        conv_name = "conv_node"
        conv_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(conv_weight_data, name=weight_name))
        conv_node = onnx.helper.make_node("Conv", conv_inputs, conv_outputs, name=conv_name)

        # make Clip node
        clip_node = create_clip_node(conv_outputs[0], "clip_output", "clip_node", initializers)

        # make Identity node
        reshape_name = "reshape_node"
        reshape_shape = "reshape_shape"
        initializers.append(onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), name=reshape_shape))
        reshape_node = onnx.helper.make_node(
            "Reshape", ["clip_output", reshape_shape], [output_name], name=reshape_name
        )

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = "QDQ_Test_Conv_clip"
        graph = helper.make_graph(
            [conv_node, clip_node, reshape_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def verify(self, per_channel, is_quant_type_int8):
        np.random.seed(1)
        model_fp32_path = f"conv_clip_fp32.{per_channel}.onnx"
        model_int8_qdq_path = f"conv_clip_quant_qdq.{per_channel}.onnx"
        model_int8_qop_path = f"conv_clip_quant_qop.{per_channel}.onnx"
        data_reader = self.input_feeds(1, {"input": [1, 8, 33, 33]})
        self.construct_model_conv_clip(model_fp32_path, [1, 8, 33, 33], [16, 8, 3, 3], [15376])
        quantize_static(
            model_fp32_path,
            model_int8_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            reduce_range=per_channel,
            activation_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
            weight_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
        )
        data_reader.rewind()
        # topo sort check
        check_op_type_order(
            self,
            model_int8_qdq_path,
            [
                "DequantizeLinear",
                "QuantizeLinear",
                "DequantizeLinear",
                "Conv",
                "QuantizeLinear",
                "DequantizeLinear",
                "Reshape",
                "QuantizeLinear",
                "DequantizeLinear",
            ],
        )
        check_model_correctness(self, model_fp32_path, model_int8_qdq_path, data_reader.get_next())

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_qop_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            per_channel=per_channel,
            reduce_range=per_channel,
            activation_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
            weight_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
        )
        data_reader.rewind()
        qop_nodes = {"QLinearConv": 1, "QuantizeLinear": 1, "DequantizeLinear": 1}
        check_op_type_count(self, model_int8_qop_path, **qop_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_qop_path, data_reader.get_next())

    def test_quantize_conv_without_bias(self):
        # only test cases per_channel=True and reduce_range=True to avoid saturation on avx2 and avx512 for weight type int8
        self.verify(True, True)  # per_channel:False, is_quant_type_int8:True

        self.verify(False, False)  # per_channel:False, is_quant_type_int8:False
        self.verify(True, False)  # per_channel:True, is_quant_type_int8:False


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """
    Helper function to generate initializers for test inputs
    """
    tensor = np.random.normal(0, 0.3, tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


def construct_relu_conv_model(test_model_path: str) -> None:
    """ Create an ONNX model shaped as:
    ```
       (input)
          |
         Relu1
         /   \
      Conv1   \
        |      \
      Relu2  Conv3
        |      |
      Conv2    |
        \\      /
          Add
           |
          (AddOut)
    ```
    """

    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1, 3])
    output_vi = helper.make_tensor_value_info("AddOut", TensorProto.FLOAT, [1, 3, 1, 3])
    w1 = generate_input_initializer([3, 3, 1, 1], np.float32, "W1")
    b1 = generate_input_initializer([3], np.float32, "B1")
    w3 = generate_input_initializer([3, 3, 1, 1], np.float32, "W3")
    b3 = generate_input_initializer([3], np.float32, "B3")
    w5 = generate_input_initializer([3, 3, 1, 1], np.float32, "W5")
    b5 = generate_input_initializer([3], np.float32, "B5")
    relu_node_1 = helper.make_node("Relu", ["input"], ["Relu1Out"], name="Relu1")
    conv_node_1 = helper.make_node("Conv", ["Relu1Out", "W1", "B1"], ["Conv1Out"], name="Conv1")
    relu_node_2 = helper.make_node("Relu", ["Conv1Out"], ["Relu2Out"], name="Relu2")
    conv_node_2 = helper.make_node("Conv", ["Relu2Out", "W3", "B3"], ["Conv2Out"], name="Conv2")
    conv_node_3 = helper.make_node("Conv", ["Relu1Out", "W5", "B5"], ["Conv3Out"], name="Conv3")
    add_node = helper.make_node("Add", ["Conv2Out", "Conv3Out"], ["AddOut"], name="Add")

    graph = helper.make_graph(
        [relu_node_1, conv_node_1, relu_node_2, conv_node_2, conv_node_3, add_node],
        "test_graph_4",
        [input_vi],
        [output_vi],
    )
    graph.initializer.add().CopyFrom(w1)
    graph.initializer.add().CopyFrom(b1)
    graph.initializer.add().CopyFrom(w3)
    graph.initializer.add().CopyFrom(b3)
    graph.initializer.add().CopyFrom(w5)
    graph.initializer.add().CopyFrom(b5)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, test_model_path)


class TestQDQFormatConvRelu(TestQDQFormat):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_qdq_format_conv_relu")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def construct_model_conv_relu(self, output_model_path, input_shape, weight_shape, output_shape):
        #    (input)
        #      |
        #     Conv
        #      |
        #     Relu
        #      |
        #    (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        # make Conv node
        weight_name = "conv_weight"
        conv_inputs = [input_name, weight_name]
        conv_outputs = ["conv_output"]
        conv_name = "conv_node"
        conv_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(conv_weight_data, name=weight_name))
        conv_node = onnx.helper.make_node("Conv", conv_inputs, conv_outputs, name=conv_name)

        # make Relu node
        relu_node = onnx.helper.make_node("Relu", conv_outputs, [output_name], name="Relu")

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = "QDQ_Test_Conv_Relu"
        graph = helper.make_graph(
            [conv_node, relu_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def verify(self, per_channel, is_quant_type_int8):
        np.random.seed(1)
        model_fp32_path = str(Path(self._tmp_model_dir.name) / f"conv_relu_fp32.{per_channel}.onnx")
        model_int8_qdq_path = str(Path(self._tmp_model_dir.name) / f"conv_relu_quant_qdq.{per_channel}.onnx")
        model_int8_qop_path = str(Path(self._tmp_model_dir.name) / f"conv_relu_quant_qop.{per_channel}.onnx")
        data_reader = self.input_feeds(1, {"input": [1, 8, 33, 33]})
        self.construct_model_conv_relu(model_fp32_path, [1, 8, 33, 33], [16, 8, 3, 3], [1, 16, 31, 31])
        quantize_static(
            model_fp32_path,
            model_int8_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            reduce_range=per_channel,
            activation_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
            weight_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
        )
        data_reader.rewind()
        # topo sort check
        check_op_type_order(
            self,
            model_int8_qdq_path,
            [
                "DequantizeLinear",
                "QuantizeLinear",
                "DequantizeLinear",
                "Conv",
                "QuantizeLinear",
                "DequantizeLinear",
            ],
        )
        check_model_correctness(self, model_fp32_path, model_int8_qdq_path, data_reader.get_next())

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_qop_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            per_channel=per_channel,
            reduce_range=per_channel,
            activation_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
            weight_type=QuantType.QInt8 if is_quant_type_int8 else QuantType.QUInt8,
        )
        data_reader.rewind()
        qop_nodes = {"QLinearConv": 1, "QuantizeLinear": 1, "DequantizeLinear": 1}
        check_op_type_count(self, model_int8_qop_path, **qop_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_qop_path, data_reader.get_next())

    def test_quantize_conv_without_bias(self):
        # only test cases per_channel=True and reduce_range=True to avoid saturation on avx2 and avx512 for weight type int8
        self.verify(True, True)  # per_channel:False, is_quant_type_int8:True

        self.verify(False, False)  # per_channel:False, is_quant_type_int8:False
        self.verify(True, False)  # per_channel:True, is_quant_type_int8:False

    def test_quantize_relu_conv(self):
        float_model_path = str(Path(self._tmp_model_dir.name) / "float_relu_convs_model.onnx")
        construct_relu_conv_model(float_model_path)
        data_reader = self.input_feeds(2, {"input": [1, 3, 1, 3]})

        qdq_model_path = str(Path(self._tmp_model_dir.name) / "qdq_relu_convs_model.onnx")
        quantize_static(
            float_model_path,
            qdq_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            reduce_range=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )


class TestQDQRemovableActivation(TestQDQFormat):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.quant.activation")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def construct_model_clip_relu(self, output_model_path, input_shape, output_shape):
        #    (input)
        #      |
        #     Clip
        #      |
        #     Relu
        #      |
        #    (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        # make Clip node
        clip_output_name = "clip_output"
        clip_node = create_clip_node(input_name, clip_output_name, "clip_node", initializers)

        # make Relu node
        relu_node = onnx.helper.make_node("Relu", [clip_output_name], [output_name], name="Relu")

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = "QDQ_Test_Clip_Relu"
        graph = helper.make_graph(
            [clip_node, relu_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def test_activation_only(self):
        float_model_path = str(Path(self._tmp_model_dir.name) / "float_relu_convs_model.onnx")
        self.construct_model_clip_relu(float_model_path, [1, 3, 1, 3], [1, 3, 1, 3])
        data_reader = self.input_feeds(2, {"input": [1, 3, 1, 3]})

        qdq_model_path = str(Path(self._tmp_model_dir.name) / "qdq_relu_convs_model.onnx")
        quantize_static(float_model_path, qdq_model_path, data_reader)

        qop_nodes = {"Clip": 1, "Relu": 1, "QuantizeLinear": 0, "DequantizeLinear": 0}
        check_op_type_count(self, qdq_model_path, **qop_nodes)


if __name__ == "__main__":
    unittest.main()
