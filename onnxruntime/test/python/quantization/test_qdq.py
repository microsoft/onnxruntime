#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import os
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
    get_tensor_consumers_and_producers,
)

from onnxruntime.quantization import QDQQuantizer, QuantFormat, QuantType, quantize_static, write_calibration_table
from onnxruntime.quantization.calibrate import CalibrationMethod, TensorData, TensorsData
from onnxruntime.quantization.quant_utils import quantize_nparray


class TestQDQFormat(unittest.TestCase):
    def input_feeds(self, n, name2shape, np_float_type=np.float32):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np_float_type)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr


class TestQDQExtraOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.qdq.extra_options_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

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

        def td(vals):
            return TensorData(lowest=np.array(vals[0], dtype=np.float32), highest=np.array(vals[1], dtype=np.float32))

        compute_data = {
            "P": td([0.1, 0.1]),
            "Q": td([0.1, 0.1]),
            "M": td([0.1, 0.1]),
            "N": td([0.1, 0.1]),
            "L": td([0.1, 0.1]),
            "O": td([0.1, 0.1]),
        }

        op_types_to_quantize = ["Add"]

        model = onnx.load_model(test_model_path)
        quantizer = QDQQuantizer(
            model,
            True,  # per_channel
            False,  # reduce_range
            QuantType.QInt8,  # weight_type
            QuantType.QInt8,  # activation_type
            compute_data,
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

        def td(vals):
            return TensorData(lowest=np.array(vals[0], dtype=np.float32), highest=np.array(vals[1], dtype=np.float32))

        compute_data = {
            "L": td([0.1, 0.1]),
            "M": td([0.1, 0.1]),
            "N": td([0.1, 0.1]),
            "O": td([0.1, 0.1]),
            "P": td([0.1, 0.1]),
            "Q": td([0.1, 0.1]),
            "R": td([0.1, 0.1]),
            "S": td([0.1, 0.1]),
            "T": td([0.1, 0.1]),
        }

        op_types_to_quantize = ["Add", "MatMul"]

        model = onnx.load_model(test_model_path)
        quantizer = QDQQuantizer(
            model,
            True,  # per_channel
            False,  # reduce_range
            QuantType.QInt8,  # weight_type
            QuantType.QInt8,  # activation_type
            compute_data,
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

    def test_qdq_keep_removable_activations_option(self):
        #
        # Create f32 model with Relu and Clip.
        # input0 ---> Conv ---> Relu ---> Conv ---> Clip ---> output
        #
        shape1 = (1, 1, 3, 3)
        w_shape1 = (2, 1, 2, 2)
        w_shape2 = (2, 2, 2, 2)
        shape3 = (1, 2, 1, 1)

        input0 = onnx.helper.make_tensor_value_info("input0", onnx.TensorProto.FLOAT, shape1)
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape3)

        # Conv1
        weight1_data = np.random.normal(-1.0, 1.0, w_shape1).astype(np.float32)
        weight1_const = onnx.numpy_helper.from_array(weight1_data, "weight1_const")
        conv1_node = onnx.helper.make_node("Conv", ["input0", "weight1_const"], ["conv1_out"], name="conv1_node")

        # Relu1
        relu1_node = onnx.helper.make_node("Relu", ["conv1_out"], ["relu1_out"], name="relu1_node")

        # Conv2
        weight2_data = np.random.normal(-1.8, 1.8, w_shape2).astype(np.float32)
        weight2_const = onnx.numpy_helper.from_array(weight2_data, "weight2_const")
        conv2_node = onnx.helper.make_node("Conv", ["relu1_out", "weight2_const"], ["conv2_out"], name="conv2_node")

        # Clip1
        min_const = onnx.numpy_helper.from_array(np.array(0.0, dtype=np.float32), "min_const")
        max_const = onnx.numpy_helper.from_array(np.array(0.5, dtype=np.float32), "max_const")
        clip1_node = onnx.helper.make_node(
            "Clip", ["conv2_out", "min_const", "max_const"], ["output"], name="clip1_node"
        )

        graph = onnx.helper.make_graph(
            [conv1_node, relu1_node, conv2_node, clip1_node],
            "keep_qdq_activations",
            [input0],
            [output],
            initializer=[weight1_const, weight2_const, min_const, max_const],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        f32_model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        f32_model = onnx.shape_inference.infer_shapes(f32_model)
        f32_model_path = os.path.join(self._tmp_dir_path, "keep.act.model.onnx")
        onnx.save_model(f32_model, f32_model_path)

        # Create a data reader.
        input_data_list = []
        for _ in range(5):
            inputs = {"input0": np.random.randint(-10, 10, shape1).astype(np.float32)}
            input_data_list.extend([inputs])
        data_reader = TestDataFeeds(input_data_list)

        #
        # Quantize model with extra option to KEEP removable activations.
        #
        qdq_model_path = os.path.join(self._tmp_dir_path, "keep.act.model.qdq.onnx")

        # Create u8_act/u8_wgt qdq model
        quantize_static(
            f32_model_path,
            qdq_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=[node.op_type for node in f32_model.graph.node],
            extra_options={"QDQKeepRemovableActivations": True},
        )

        has_relu = False
        has_clip = False

        qdq_model = onnx.load_model(qdq_model_path)

        for node in qdq_model.graph.node:
            if node.op_type == "Relu":
                has_relu = True
            if node.op_type == "Clip":
                has_clip = True

        self.assertTrue(has_relu)
        self.assertTrue(has_clip)

        #
        # Quantize model without extra option. Clip and Relu should be removed by default.
        #
        qdq_model_path = os.path.join(self._tmp_dir_path, "nokeep.act.model.qdq.onnx")
        data_reader.rewind()

        # Create u8_act/u8_wgt qdq model
        quantize_static(
            f32_model_path,
            qdq_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=[node.op_type for node in f32_model.graph.node],
        )

        has_relu = False
        has_clip = False

        qdq_model = onnx.load_model(qdq_model_path)

        for node in qdq_model.graph.node:
            if node.op_type == "Relu":
                has_relu = True
            if node.op_type == "Clip":
                has_clip = True

        self.assertFalse(has_relu)
        self.assertFalse(has_clip)


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

    def construct_model_conv_relu(
        self, output_model_path, input_shape, weight_shape, output_shape, opset=13, ir_version=7
    ):
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
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        model.ir_version = ir_version

        onnx.save(model, output_model_path)

    def verify_qdq(
        self,
        per_channel,
        activation_type,
        weight_type,
        extra_options=None,
        opset=13,
        ir_version=7,
        rtol=1e-2,
        atol=0.05,
    ):
        np.random.seed(1)
        model_fp32_path = str(Path(self._tmp_model_dir.name) / f"conv_relu_fp32.{per_channel}.onnx")
        model_qdq_path = str(
            Path(self._tmp_model_dir.name) / f"conv_relu_quant_qdq.{activation_type}.{weight_type}.{per_channel}.onnx"
        )
        data_reader = self.input_feeds(1, {"input": [1, 8, 33, 33]})
        self.construct_model_conv_relu(
            model_fp32_path, [1, 8, 33, 33], [16, 8, 3, 3], [1, 16, 31, 31], opset=opset, ir_version=ir_version
        )
        quantize_static(
            model_fp32_path,
            model_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            reduce_range=per_channel,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        data_reader.rewind()
        # topo sort check
        check_op_type_order(
            self,
            model_qdq_path,
            [
                "DequantizeLinear",
                "QuantizeLinear",
                "DequantizeLinear",
                "Conv",
                "QuantizeLinear",
                "DequantizeLinear",
            ],
        )

        # checks that the qdq model has INT4 or INT16 types when expected
        with open(model_qdq_path, "rb") as f:
            qdq_model = onnx.load(f)
        inits = {init.name: init for init in qdq_model.graph.initializer}
        zero_types = []
        for node in qdq_model.graph.node:
            print(node.op_type)
            if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
                continue
            zp = inits[node.input[2]]
            zero_types.append(zp.data_type)

        to_tensor_types = {
            QuantType.QInt4: TensorProto.INT4,
            QuantType.QUInt4: TensorProto.UINT4,
            QuantType.QInt16: TensorProto.INT16,
            QuantType.QUInt16: TensorProto.UINT16,
        }
        assert weight_type not in to_tensor_types or to_tensor_types[weight_type] in zero_types, (
            f"weight_type={weight_type} not in zero_types={zero_types}"
        )
        assert activation_type not in to_tensor_types or to_tensor_types[activation_type] in zero_types, (
            f"activation_type={activation_type} not in zero_types={zero_types}"
        )

        check_model_correctness(self, model_fp32_path, model_qdq_path, data_reader.get_next(), rtol=rtol, atol=atol)

        # If the model uses Q/DQ ops with "com.microsoft" domain (e.g., for int16 support),
        # then ensure the model has the appropriate opset import.
        if extra_options and extra_options.get("UseQDQContribOps", False):
            qdq_model = onnx.load_model(model_qdq_path)
            ms_opset = next((opset for opset in qdq_model.opset_import if opset.domain == "com.microsoft"), None)
            self.assertIsNot(ms_opset, None)

    def verify_qop(self, per_channel, is_quant_type_int8):
        np.random.seed(1)
        model_fp32_path = str(Path(self._tmp_model_dir.name) / f"conv_relu_fp32.{per_channel}.onnx")
        model_int8_qop_path = str(Path(self._tmp_model_dir.name) / f"conv_relu_quant_qop.{per_channel}.onnx")
        data_reader = self.input_feeds(1, {"input": [1, 8, 33, 33]})
        self.construct_model_conv_relu(model_fp32_path, [1, 8, 33, 33], [16, 8, 3, 3], [1, 16, 31, 31])

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
        self.verify_qdq(True, QuantType.QInt8, QuantType.QInt8)  # per_channel:True
        self.verify_qop(True, True)  # per_channel:True, is_quant_type_int8:True

        self.verify_qdq(False, QuantType.QUInt8, QuantType.QUInt8)  # per_channel:False
        self.verify_qop(False, False)  # per_channel:False, is_quant_type_int8:False

        self.verify_qdq(True, QuantType.QUInt8, QuantType.QUInt8)  # per_channel:True
        self.verify_qop(True, False)  # per_channel:True, is_quant_type_int8:False

        # 16-bit QDQ via contrib ops
        self.verify_qdq(False, QuantType.QUInt16, QuantType.QUInt16, {"UseQDQContribOps": True})
        self.verify_qdq(False, QuantType.QInt16, QuantType.QInt16, {"UseQDQContribOps": True})
        self.verify_qdq(False, QuantType.QUInt16, QuantType.QUInt8, {"UseQDQContribOps": True})
        self.verify_qdq(False, QuantType.QInt16, QuantType.QInt8, {"UseQDQContribOps": True})

        self.verify_qdq(True, QuantType.QUInt16, QuantType.QUInt16, {"UseQDQContribOps": True})
        self.verify_qdq(True, QuantType.QInt16, QuantType.QInt16, {"UseQDQContribOps": True})
        self.verify_qdq(True, QuantType.QUInt16, QuantType.QUInt8, {"UseQDQContribOps": True})
        self.verify_qdq(True, QuantType.QInt16, QuantType.QInt8, {"UseQDQContribOps": True})

        # 4-bit QDQ - per tensor
        self.verify_qdq(False, QuantType.QInt16, QuantType.QInt4, opset=21, ir_version=10, atol=0.4)  # per-tensor
        self.verify_qdq(
            False, QuantType.QInt16, QuantType.QInt4, {"UseQDQContribOps": True}, opset=21, ir_version=10, atol=0.4
        )

        # 4-bit QDQ - per channel
        self.verify_qdq(True, QuantType.QInt16, QuantType.QInt4, opset=21, ir_version=10, atol=0.6)
        self.verify_qdq(
            True, QuantType.QInt16, QuantType.QInt4, {"UseQDQContribOps": True}, opset=21, ir_version=10, atol=0.6
        )

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


class TestQDQMixedPrecision(TestQDQFormat):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.qdq.mixed_prec_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def build_test_model_for_add_qdq_ops(
        self,
        num_consumers: int,
        is_graph_output: bool,
        float_type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT,
        op0_transpose: bool = False,
    ):
        """
        Builds a float32 model with a single producer node and a configurable number of consumer nodes.
        The tensor between the producer and consumers can be optionally made a graph output.
        op_0 can optionally be made a Transpose node to test sharing qparams across the input and output.

                           +-> op_0_out (optional graph output)
                           |
        input_0 --> op_0 --+-> op_1 --> output_0
                           |
                           +-> op_2 --> output_1
                           |
                           ...
                           |
                           +-> op_{n} --> output_{n-1}
        """
        shape = (1, 2, 3)
        shape_t = (1, 3, 2)
        input_0 = onnx.helper.make_tensor_value_info("input_0", float_type, shape)
        output_shape = shape if not op0_transpose else shape_t

        outputs = []
        for i in range(num_consumers):
            outputs.append(onnx.helper.make_tensor_value_info(f"output_{i}", float_type, output_shape))

        if is_graph_output:
            outputs.append(onnx.helper.make_tensor_value_info("op_0_out", float_type, output_shape))

        nodes = []
        if op0_transpose:
            nodes.append(onnx.helper.make_node("Transpose", ["input_0"], ["op_0_out"], perm=[0, 2, 1], name="op_0"))
        else:
            nodes.append(onnx.helper.make_node("Sigmoid", ["input_0"], ["op_0_out"], name="op_0"))

        for i in range(num_consumers):
            op_index = i + 1
            nodes.append(onnx.helper.make_node("Cos", ["op_0_out"], [f"output_{i}"], name=f"op_{op_index}"))

        graph = onnx.helper.make_graph(
            nodes,
            "test_add_qdq_ops_for_converted_activation",
            [input_0],
            outputs,
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        return onnx.shape_inference.infer_shapes(model)

    def test_add_tensor_qdq_ops_case_1(self):
        """
        Tensor T is not a graph output; all consumers use the converted type
        <Producer> ---> Q1 ---> DQ1 ---> Q2 ---> DQ2 ---> <Consumers>
        """
        # Test configurations (qparam_sharing, float_type)
        subtest_configs = [
            (False, onnx.TensorProto.FLOAT, np.float32),
            (False, onnx.TensorProto.FLOAT16, np.float16),
            (True, onnx.TensorProto.FLOAT, np.float32),
            (True, onnx.TensorProto.FLOAT16, np.float16),
        ]
        for test_qparam_sharing, float_type, np_float_type in subtest_configs:
            with self.subTest(test_qparam_sharing=test_qparam_sharing, float_type=float_type):
                label = f"_share{test_qparam_sharing}_f{float_type}"
                float_model_path = os.path.join(self._tmp_dir_path, f"case_1{label}.onnx")
                qdq_model_path = os.path.join(self._tmp_dir_path, f"case_1{label}.qdq.onnx")
                float_model = self.build_test_model_for_add_qdq_ops(
                    2, False, float_type=float_type, op0_transpose=test_qparam_sharing
                )
                onnx.save_model(float_model, float_model_path)

                data_reader = self.input_feeds(3, {"input_0": (1, 2, 3)}, np_float_type)

                mixed_prec_overrides = {
                    "op_0_out": [
                        {
                            "quant_type": QuantType.QUInt8,
                            "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"op_1", "op_2"}},
                        }
                    ],
                    "output_0": [{"quant_type": QuantType.QUInt16}],
                    "output_1": [{"quant_type": QuantType.QUInt16}],
                }
                quantize_static(
                    float_model_path,
                    qdq_model_path,
                    data_reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    op_types_to_quantize=[node.op_type for node in float_model.graph.node],
                    extra_options={
                        "TensorQuantOverrides": mixed_prec_overrides,
                        "ForceQuantizeNoInputCheck": test_qparam_sharing,  # To ensure Transpose is wrapped in DQ/Q
                    },
                )

                # Expect the following QDQ model:
                # input_0 --> Q --> DQ --> op_0 --> Q_8 --> DQ_8 --> Q_16 --> DQ_16 -+-> op_1 --> Q --> DQ --> output_0
                #                                                                    |
                #                                                                    +-> op_2 --> Q --> DQ --> output_1
                qdq_node_counts = {"QuantizeLinear": 5, "DequantizeLinear": 5}
                check_op_type_count(self, qdq_model_path, **qdq_node_counts)

                qdq_model = onnx.load_model(qdq_model_path)
                onnx.checker.check_model(qdq_model, True)

                initializers = {init.name: init for init in qdq_model.graph.initializer}

                # Check zero-point data types
                orig_zp_init = None
                if test_qparam_sharing:
                    # op_0_out_zero_point should not be in the model because the Transpose output is sharing
                    # qparams from the Transpose input.
                    self.assertNotIn("op_0_out_zero_point", initializers)
                    orig_zp_init = initializers["input_0_zero_point"]
                else:
                    orig_zp_init = initializers["op_0_out_zero_point"]

                self.assertEqual(orig_zp_init.data_type, onnx.TensorProto.UINT8)
                convert_zp_init = initializers["op_0_out_zero_point_convert"]
                self.assertEqual(convert_zp_init.data_type, onnx.TensorProto.UINT16)
                output_0_zp_init = initializers["output_0_zero_point"]
                self.assertEqual(output_0_zp_init.data_type, onnx.TensorProto.UINT16)
                output_1_zp_init = initializers["output_1_zero_point"]
                self.assertEqual(output_1_zp_init.data_type, onnx.TensorProto.UINT16)

                # Check scale data types
                orig_scale_init = None
                if test_qparam_sharing:
                    self.assertNotIn("op_0_out_scale", initializers)
                    orig_scale_init = initializers["input_0_scale"]
                else:
                    orig_scale_init = initializers["op_0_out_scale"]

                self.assertEqual(orig_scale_init.data_type, float_type)
                convert_scale_init = initializers["op_0_out_scale_convert"]
                self.assertEqual(convert_scale_init.data_type, float_type)
                output_0_scale_init = initializers["output_0_scale"]
                self.assertEqual(output_0_scale_init.data_type, float_type)
                output_1_scale_init = initializers["output_1_scale"]
                self.assertEqual(output_1_scale_init.data_type, float_type)

    def test_add_tensor_qdq_ops_case_2(self):
        """
        Tensor T is not a graph output; some consumers use the original type, others use the converted type
        <Producer> ---> Q1 -+-> DQ1 ---> <Consumers of original type>
                            |
                            +-> DQ1' ---> Q2 ---> DQ2 ---> <Consumers of converted type>
        """
        # Test configurations (qparam_sharing, float_type)
        subtest_configs = [
            (False, onnx.TensorProto.FLOAT, np.float32),
            (False, onnx.TensorProto.FLOAT16, np.float16),
            (True, onnx.TensorProto.FLOAT, np.float32),
            (True, onnx.TensorProto.FLOAT16, np.float16),
        ]
        for test_qparam_sharing, float_type, np_float_type in subtest_configs:
            with self.subTest(test_qparam_sharing=test_qparam_sharing, float_type=float_type):
                label = f"_share{test_qparam_sharing}_f{float_type}"
                float_model_path = os.path.join(self._tmp_dir_path, f"case_2{label}.onnx")
                qdq_model_path = os.path.join(self._tmp_dir_path, f"case_2{label}.qdq.onnx")
                float_model = self.build_test_model_for_add_qdq_ops(
                    4, False, float_type=float_type, op0_transpose=test_qparam_sharing
                )
                onnx.save_model(float_model, float_model_path)

                data_reader = self.input_feeds(3, {"input_0": (1, 2, 3)}, np_float_type)

                mixed_prec_overrides = {
                    "op_0_out": [
                        {
                            "quant_type": QuantType.QUInt8,
                            "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"op_3", "op_4"}},
                        }
                    ],
                    "output_2": [{"quant_type": QuantType.QUInt16}],
                    "output_3": [{"quant_type": QuantType.QUInt16}],
                }
                quantize_static(
                    float_model_path,
                    qdq_model_path,
                    data_reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    op_types_to_quantize=[node.op_type for node in float_model.graph.node],
                    extra_options={
                        "TensorQuantOverrides": mixed_prec_overrides,
                        "ForceQuantizeNoInputCheck": test_qparam_sharing,  # To ensure Transpose is wrapped in DQ/Q
                    },
                )

                # Expect the following QDQ model:
                # input_0 --> Q --> DQ --> op_0 --> Q_8 -+-> DQ_8 -+-> op_1 --> Q --> DQ --> output_0
                #                                        |         |
                #                                        |         +-> op_2 --> Q --> DQ --> output_1
                #                                        |
                #                                        +-> DQ_8' --> Q_16 --> DQ_16 -+-> op_3 --> Q --> DQ --> output_2
                #                                                                      |
                #                                                                      +-> op_4 --> Q --> DQ --> output_3
                qdq_node_counts = {"QuantizeLinear": 7, "DequantizeLinear": 8}
                check_op_type_count(self, qdq_model_path, **qdq_node_counts)

                qdq_model = onnx.load_model(qdq_model_path)
                onnx.checker.check_model(qdq_model, True)

                initializers = {init.name: init for init in qdq_model.graph.initializer}

                # Check zero-point data types
                orig_zp_init = None
                if test_qparam_sharing:
                    # op_0_out_zero_point should not be in the model because the Transpose output is sharing
                    # qparams from the Transpose input.
                    self.assertNotIn("op_0_out_zero_point", initializers)
                    orig_zp_init = initializers["input_0_zero_point"]
                else:
                    orig_zp_init = initializers["op_0_out_zero_point"]

                self.assertEqual(orig_zp_init.data_type, onnx.TensorProto.UINT8)
                convert_zp_init = initializers["op_0_out_zero_point_convert"]
                self.assertEqual(convert_zp_init.data_type, onnx.TensorProto.UINT16)
                output_0_zp_init = initializers["output_0_zero_point"]
                self.assertEqual(output_0_zp_init.data_type, onnx.TensorProto.UINT8)
                output_1_zp_init = initializers["output_1_zero_point"]
                self.assertEqual(output_1_zp_init.data_type, onnx.TensorProto.UINT8)
                output_2_zp_init = initializers["output_2_zero_point"]
                self.assertEqual(output_2_zp_init.data_type, onnx.TensorProto.UINT16)
                output_3_zp_init = initializers["output_3_zero_point"]
                self.assertEqual(output_3_zp_init.data_type, onnx.TensorProto.UINT16)

                # Check scale data types
                orig_scale_init = None
                if test_qparam_sharing:
                    self.assertNotIn("op_0_out_scale", initializers)
                    orig_scale_init = initializers["input_0_scale"]
                else:
                    orig_scale_init = initializers["op_0_out_scale"]

                self.assertEqual(orig_scale_init.data_type, float_type)
                convert_scale_init = initializers["op_0_out_scale_convert"]
                self.assertEqual(convert_scale_init.data_type, float_type)
                output_0_scale_init = initializers["output_0_scale"]
                self.assertEqual(output_0_scale_init.data_type, float_type)
                output_1_scale_init = initializers["output_1_scale"]
                self.assertEqual(output_1_scale_init.data_type, float_type)
                output_2_scale_init = initializers["output_2_scale"]
                self.assertEqual(output_2_scale_init.data_type, float_type)
                output_3_scale_init = initializers["output_3_scale"]
                self.assertEqual(output_3_scale_init.data_type, float_type)

    def test_add_tensor_qdq_ops_case_3(self):
        """
        Tensor T is a graph output; all consumers use the converted type
        <Producer> ---> Q1 ---> DQ1 ---> Q2 ---> DQ2 -+-> <Consumers>
                                                      |
                                                      +-> <Graph output>
        """
        # Test configurations (qparam_sharing, float_type)
        subtest_configs = [
            (False, onnx.TensorProto.FLOAT, np.float32),
            (False, onnx.TensorProto.FLOAT16, np.float16),
            (True, onnx.TensorProto.FLOAT, np.float32),
            (True, onnx.TensorProto.FLOAT16, np.float16),
        ]
        for test_qparam_sharing, float_type, np_float_type in subtest_configs:
            with self.subTest(test_qparam_sharing=test_qparam_sharing, float_type=float_type):
                label = f"_share{test_qparam_sharing}_f{float_type}"
                float_model_path = os.path.join(self._tmp_dir_path, f"case_3{label}.onnx")
                qdq_model_path = os.path.join(self._tmp_dir_path, f"case_3{label}.qdq.onnx")
                float_model = self.build_test_model_for_add_qdq_ops(
                    2, True, float_type=float_type, op0_transpose=test_qparam_sharing
                )
                onnx.save_model(float_model, float_model_path)

                data_reader = self.input_feeds(3, {"input_0": (1, 2, 3)}, np_float_type)

                mixed_prec_overrides = {
                    "op_0_out": [
                        {
                            "quant_type": QuantType.QUInt8,
                            "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"op_1", "op_2"}},
                        }
                    ],
                    "output_0": [{"quant_type": QuantType.QUInt16}],
                    "output_1": [{"quant_type": QuantType.QUInt16}],
                }
                quantize_static(
                    float_model_path,
                    qdq_model_path,
                    data_reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    op_types_to_quantize=[node.op_type for node in float_model.graph.node],
                    extra_options={
                        "TensorQuantOverrides": mixed_prec_overrides,
                        "ForceQuantizeNoInputCheck": test_qparam_sharing,  # To ensure Transpose is wrapped in DQ/Q
                    },
                )

                # Expect the following QDQ model:
                # input_0 --> Q --> DQ --> op_0 --> Q_8 --> DQ_8 --> Q_16 --> DQ_16 -+-> op_1 --> Q --> DQ --> output_0
                #                                                                    |
                #                                                                    +-> op_2 --> Q --> DQ --> output_1
                #                                                                    |
                #                                                                    +--> op_0_out (is graph output)
                qdq_node_counts = {"QuantizeLinear": 5, "DequantizeLinear": 5}
                check_op_type_count(self, qdq_model_path, **qdq_node_counts)

                qdq_model = onnx.load_model(qdq_model_path)
                onnx.checker.check_model(qdq_model, True)

                initializers = {init.name: init for init in qdq_model.graph.initializer}
                graph_outputs = {g_output.name: g_output for g_output in qdq_model.graph.output}

                # Check zero-point data types
                orig_zp_init = None
                if test_qparam_sharing:
                    # op_0_out_zero_point should not be in the model because the Transpose output is sharing
                    # qparams from the Transpose input.
                    self.assertNotIn("op_0_out_zero_point", initializers)
                    self.assertNotIn("op_0_out_scale", initializers)
                    orig_zp_init = initializers["input_0_zero_point"]
                else:
                    orig_zp_init = initializers["op_0_out_zero_point"]

                self.assertEqual(orig_zp_init.data_type, onnx.TensorProto.UINT8)
                convert_zp_init = initializers["op_0_out_zero_point_convert"]
                self.assertEqual(convert_zp_init.data_type, onnx.TensorProto.UINT16)
                output_0_zp_init = initializers["output_0_zero_point"]
                self.assertEqual(output_0_zp_init.data_type, onnx.TensorProto.UINT16)
                output_1_zp_init = initializers["output_1_zero_point"]
                self.assertEqual(output_1_zp_init.data_type, onnx.TensorProto.UINT16)

                # Check scale data types
                orig_scale_init = None
                if test_qparam_sharing:
                    self.assertNotIn("op_0_out_scale", initializers)
                    orig_scale_init = initializers["input_0_scale"]
                else:
                    orig_scale_init = initializers["op_0_out_scale"]

                self.assertEqual(orig_scale_init.data_type, float_type)
                convert_scale_init = initializers["op_0_out_scale_convert"]
                self.assertEqual(convert_scale_init.data_type, float_type)
                output_0_scale_init = initializers["output_0_scale"]
                self.assertEqual(output_0_scale_init.data_type, float_type)
                output_1_scale_init = initializers["output_1_scale"]
                self.assertEqual(output_1_scale_init.data_type, float_type)

                self.assertIn("op_0_out", graph_outputs)

    def test_add_tensor_qdq_ops_case_4(self):
        """
        Tensor T is a graph output; some consumers use the original type, others use the converted type
        <Producer> ---> Q1 -+-> DQ1 -+-> <Consumers of original type>
                            |        |
                            |        +-> <Graph output>
                            |
                            +-> DQ1' ---> Q2 ---> DQ2 ---> <Consumers of converted type>
        """
        # Test configurations (qparam_sharing, float_type)
        subtest_configs = [
            (False, onnx.TensorProto.FLOAT, np.float32),
            (False, onnx.TensorProto.FLOAT16, np.float16),
            (True, onnx.TensorProto.FLOAT, np.float32),
            (True, onnx.TensorProto.FLOAT16, np.float16),
        ]
        for test_qparam_sharing, float_type, np_float_type in subtest_configs:
            with self.subTest(test_qparam_sharing=test_qparam_sharing, float_type=float_type):
                label = f"_share{test_qparam_sharing}_f{float_type}"
                float_model_path = os.path.join(self._tmp_dir_path, f"case_4{label}.onnx")
                qdq_model_path = os.path.join(self._tmp_dir_path, f"case_4{label}.qdq.onnx")
                float_model = self.build_test_model_for_add_qdq_ops(
                    4, True, float_type=float_type, op0_transpose=test_qparam_sharing
                )
                onnx.save_model(float_model, float_model_path)

                data_reader = self.input_feeds(3, {"input_0": (1, 2, 3)}, np_float_type)

                mixed_prec_overrides = {
                    "op_0_out": [
                        {
                            "quant_type": QuantType.QUInt8,
                            "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"op_3", "op_4"}},
                        }
                    ],
                    "output_2": [{"quant_type": QuantType.QUInt16}],
                    "output_3": [{"quant_type": QuantType.QUInt16}],
                }
                quantize_static(
                    float_model_path,
                    qdq_model_path,
                    data_reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    op_types_to_quantize=[node.op_type for node in float_model.graph.node],
                    extra_options={
                        "TensorQuantOverrides": mixed_prec_overrides,
                        "ForceQuantizeNoInputCheck": test_qparam_sharing,  # To ensure Transpose is wrapped in DQ/Q
                    },
                )

                # Expect the following QDQ model:
                # input_0 --> Q --> DQ --> op_0 --> Q_8 -+-> DQ_8 -+-> op_1 --> Q --> DQ --> output_0
                #                                        |         |
                #                                        |         +-> op_2 --> Q --> DQ --> output_1
                #                                        |         |
                #                                        |         +-> op_0_out (is graph output)
                #                                        |
                #                                        +-> DQ_8' --> Q_16 --> DQ_16 -+-> op_3 --> Q --> DQ --> output_2
                #                                                                      |
                #                                                                      +-> op_4 --> Q --> DQ --> output_3
                qdq_node_counts = {"QuantizeLinear": 7, "DequantizeLinear": 8}
                check_op_type_count(self, qdq_model_path, **qdq_node_counts)

                qdq_model = onnx.load_model(qdq_model_path)
                onnx.checker.check_model(qdq_model, True)

                initializers = {init.name: init for init in qdq_model.graph.initializer}
                graph_outputs = {g_output.name: g_output for g_output in qdq_model.graph.output}

                # Check zero-point data types
                orig_zp_init = None
                if test_qparam_sharing:
                    # op_0_out_zero_point should not be in the model because the Transpose output is sharing
                    # qparams from the Transpose input.
                    self.assertNotIn("op_0_out_zero_point", initializers)
                    orig_zp_init = initializers["input_0_zero_point"]
                else:
                    orig_zp_init = initializers["op_0_out_zero_point"]

                self.assertEqual(orig_zp_init.data_type, onnx.TensorProto.UINT8)
                convert_zp_init = initializers["op_0_out_zero_point_convert"]
                self.assertEqual(convert_zp_init.data_type, onnx.TensorProto.UINT16)
                output_0_zp_init = initializers["output_0_zero_point"]
                self.assertEqual(output_0_zp_init.data_type, onnx.TensorProto.UINT8)
                output_1_zp_init = initializers["output_1_zero_point"]
                self.assertEqual(output_1_zp_init.data_type, onnx.TensorProto.UINT8)
                output_2_zp_init = initializers["output_2_zero_point"]
                self.assertEqual(output_2_zp_init.data_type, onnx.TensorProto.UINT16)
                output_3_zp_init = initializers["output_3_zero_point"]
                self.assertEqual(output_3_zp_init.data_type, onnx.TensorProto.UINT16)

                # Check scale data types
                orig_scale_init = None
                if test_qparam_sharing:
                    self.assertNotIn("op_0_out_scale", initializers)
                    orig_scale_init = initializers["input_0_scale"]
                else:
                    orig_scale_init = initializers["op_0_out_scale"]

                self.assertEqual(orig_scale_init.data_type, float_type)
                convert_scale_init = initializers["op_0_out_scale_convert"]
                self.assertEqual(convert_scale_init.data_type, float_type)
                output_0_scale_init = initializers["output_0_scale"]
                self.assertEqual(output_0_scale_init.data_type, float_type)
                output_1_scale_init = initializers["output_1_scale"]
                self.assertEqual(output_1_scale_init.data_type, float_type)
                output_2_scale_init = initializers["output_2_scale"]
                self.assertEqual(output_2_scale_init.data_type, float_type)
                output_3_scale_init = initializers["output_3_scale"]
                self.assertEqual(output_3_scale_init.data_type, float_type)

                self.assertIn("op_0_out", graph_outputs)

    def test_add_tensor_qdq_ops_case_5(self):
        """
        Tensor T is a graph output without any consumers.
        <Producer> ---> Q1 ---> DQ1 ---> Q2 ---> DQ2 ---> <Graph output>
        """
        float_model_path = os.path.join(self._tmp_dir_path, "case_5.onnx")
        qdq_model_path = os.path.join(self._tmp_dir_path, "case_5.qdq.onnx")

        # Build model with input_0 -> op_0 -> op_0_out
        # The graph output has no consumers.
        float_model = self.build_test_model_for_add_qdq_ops(0, True)
        onnx.save_model(float_model, float_model_path)

        data_reader = self.input_feeds(3, {"input_0": (1, 2, 3)}, np.float32)

        mixed_prec_overrides = {
            "input_0": [{"quant_type": QuantType.QUInt16}],
            "op_0_out": [
                {
                    "quant_type": QuantType.QUInt16,
                    "convert": {"quant_type": QuantType.QUInt8},
                }
            ],
        }
        quantize_static(
            float_model_path,
            qdq_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            op_types_to_quantize=[node.op_type for node in float_model.graph.node],
            extra_options={
                "TensorQuantOverrides": mixed_prec_overrides,
            },
        )

        # Expect the following QDQ model:
        # input_0 --> Q_16 --> DQ_16 --> op_0 --> Q_16 --> DQ_16 --> Q_8 --> DQ_8 --> output_0
        qdq_node_counts = {"QuantizeLinear": 3, "DequantizeLinear": 3}
        check_op_type_count(self, qdq_model_path, **qdq_node_counts)

        qdq_model = onnx.load_model(qdq_model_path)
        onnx.checker.check_model(qdq_model, True)

        initializers = {init.name: init for init in qdq_model.graph.initializer}

        # Check zero-point data types
        orig_zp_init = initializers["op_0_out_zero_point"]
        self.assertEqual(orig_zp_init.data_type, onnx.TensorProto.UINT16)
        convert_zp_init = initializers["op_0_out_zero_point_convert"]
        self.assertEqual(convert_zp_init.data_type, onnx.TensorProto.UINT8)

    def build_test_model_1(self, shape):
        """
        Returns the following float32 model.

        input_0 --> op1 --> op3 --> op5 --> op6 --> output_0
                                     ^
                                     |
        input_1 --> op2 -+-> op4 ----+
                         |
                         +-> op7 --> output_1
                         |
                         +-> op8 --> output_2
        """
        input_0 = onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, shape)
        input_1 = onnx.helper.make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, shape)
        output_0 = onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, shape)
        output_1 = onnx.helper.make_tensor_value_info("output_1", onnx.TensorProto.FLOAT, shape)
        output_2 = onnx.helper.make_tensor_value_info("output_2", onnx.TensorProto.FLOAT, shape)

        op1_node = onnx.helper.make_node("Sigmoid", ["input_0"], ["op1_out"], name="op1")
        op2_node = onnx.helper.make_node("Cos", ["input_1"], ["op2_out"], name="op2")
        op3_node = onnx.helper.make_node("Sin", ["op1_out"], ["op3_out"], name="op3")
        op4_node = onnx.helper.make_node("Tanh", ["op2_out"], ["op4_out"], name="op4")
        op5_node = onnx.helper.make_node("Mul", ["op3_out", "op4_out"], ["op5_out"], name="op5")
        op6_node = onnx.helper.make_node("Relu", ["op5_out"], ["output_0"], name="op6")
        op7_node = onnx.helper.make_node("Cos", ["op2_out"], ["output_1"], name="op7")
        op8_node = onnx.helper.make_node("Sigmoid", ["op2_out"], ["output_2"], name="op8")

        graph = onnx.helper.make_graph(
            [
                op1_node,
                op2_node,
                op3_node,
                op4_node,
                op5_node,
                op6_node,
                op7_node,
                op8_node,
            ],
            "mixed_prec_test",
            [input_0, input_1],
            [output_0, output_1, output_2],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        return onnx.shape_inference.infer_shapes(model)

    def test_16bit_subgraph(self):
        """
        Test correctness of a qdq model that uses a default 8-bit quantization type and contains
        a subgraph that uses 16-bit activations.
        """
        shape = (1, 2, 3)
        f32_model_path = os.path.join(self._tmp_dir_path, "model.onnx")
        qdq_model_path = os.path.join(self._tmp_dir_path, "model.qdq.onnx")
        qdq_mixed_model_path = os.path.join(self._tmp_dir_path, "model.mixed.qdq.onnx")
        f32_model = self.build_test_model_1(shape)
        onnx.save_model(f32_model, f32_model_path)

        data_reader = self.input_feeds(3, {"input_0": shape, "input_1": shape})

        # Create pure 8-bit qdq model
        quantize_static(
            f32_model_path,
            qdq_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            op_types_to_quantize=[node.op_type for node in f32_model.graph.node],
        )

        # Create mixed precision 8-bit/16-bit qdq model
        mixed_prec_overrides = {
            "op2_out": [
                {"quant_type": QuantType.QUInt8, "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"op4"}}}
            ],
            "op3_out": [
                {"quant_type": QuantType.QUInt8, "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"op5"}}}
            ],
            "op4_out": [{"quant_type": QuantType.QUInt16}],
            "op5_out": [{"quant_type": QuantType.QUInt16}],
            "output_0": [{"quant_type": QuantType.QUInt16}],
        }
        data_reader.rewind()
        quantize_static(
            f32_model_path,
            qdq_mixed_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            op_types_to_quantize=[node.op_type for node in f32_model.graph.node],
            extra_options={"TensorQuantOverrides": mixed_prec_overrides},
        )

        qop_nodes = {"Relu": 0, "QuantizeLinear": 11, "DequantizeLinear": 12}
        check_op_type_count(self, qdq_mixed_model_path, **qop_nodes)
        data_reader.rewind()
        check_model_correctness(self, f32_model_path, qdq_mixed_model_path, data_reader.get_next())
        data_reader.rewind()
        check_model_correctness(self, f32_model_path, qdq_model_path, data_reader.get_next())


class TestQDQ4bit(TestQDQFormat):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.qdq.4bit_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def build_conv_test_model(
        self,
        inp_shape: list[int],
        weight_data: np.ndarray,
        bias_data: np.ndarray,
    ):
        input_0 = onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape)
        output_0 = onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, None)
        weight = onnx.numpy_helper.from_array(weight_data, "weight")
        bias = onnx.numpy_helper.from_array(bias_data, "bias")

        conv_node = onnx.helper.make_node("Conv", ["input_0", "weight", "bias"], ["output_0"], name="Conv0")
        graph = onnx.helper.make_graph(
            [conv_node],
            "Convf32",
            [input_0],
            [output_0],
            initializer=[weight, bias],
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)

        return onnx.shape_inference.infer_shapes(model)

    def test_int4_qdq_conv(self):
        """
        Test quantization of int4 conv weight.
        """
        float_model_path = os.path.join(self._tmp_dir_path, "conv_int4.f32.onnx")
        qdq_model_path = os.path.join(self._tmp_dir_path, "conv_int4.qdq.onnx")

        inp_shape = [1, 2, 100, 100]
        weight_shape = [2, 2, 20, 20]

        # range = 3.0, scale = 3/15, zp = 0
        weight_data = np.linspace(-1.5, 1.5, num=1600, dtype=np.float32).reshape(weight_shape)
        bias_data = np.array([-10.0, 10.0], dtype=np.float32)
        float_model = self.build_conv_test_model(inp_shape, weight_data, bias_data)

        onnx.checker.check_model(float_model, True)
        onnx.save_model(float_model, float_model_path)

        data_reader = self.input_feeds(3, {"input_0": inp_shape}, np.float32)

        tensor_quant_overrides = {
            "weight": [{"quant_type": QuantType.QInt4}],  # Quantize weights to INT4
        }
        quantize_static(
            float_model_path,
            qdq_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt16,
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=[node.op_type for node in float_model.graph.node],
            extra_options={
                "TensorQuantOverrides": tensor_quant_overrides,
            },
        )

        qdq_node_counts = {"QuantizeLinear": 2, "DequantizeLinear": 4}
        check_op_type_count(self, qdq_model_path, **qdq_node_counts)

        qdq_model = onnx.load_model(qdq_model_path)
        onnx.checker.check_model(qdq_model, True)

        initializers = {init.name: init for init in qdq_model.graph.initializer}

        # Check the the weight's zero-point data type is INT4 and has expected value
        zp_val = 0
        weight_zp_init = initializers["weight_zero_point"]
        self.assertEqual(weight_zp_init.data_type, onnx.TensorProto.INT4)
        self.assertEqual(weight_zp_init.int32_data[0], zp_val)

        # Check for the expected scale value
        weight_scale_init = initializers["weight_scale"]
        scale_val = np.float32(3.0 / 15)
        self.assertEqual(weight_scale_init.data_type, onnx.TensorProto.FLOAT)
        self.assertEqual(weight_scale_init.float_data[0], scale_val)

        # Check that INT4 weights take up approximately 50% the size of INT8 weights.
        # Using protobuf's ByteSize() is not exact because it includes other fields in the proto message.
        unpacked_size = 1
        for dim in weight_shape:
            unpacked_size *= dim

        weight_quant_init = initializers["weight_quantized"]
        size_ratio = weight_quant_init.ByteSize() / unpacked_size
        self.assertLess(size_ratio, 0.55)

        # Check that the quantized weight values are correct.
        if weight_quant_init.HasField("raw_data"):
            float_data = weight_data.flatten().tolist()
            for index, float_val in enumerate(float_data):
                expected_int4_val = np.clip(np.float32(float_val / scale_val).round() + zp_val, -8, 7)
                int4_pair = onnx.subbyte.unpack_single_4bitx2(weight_quant_init.raw_data[index >> 1], True)
                int4_val = int4_pair[index & 0x1]

                self.assertEqual(np.float32(int4_val), expected_int4_val)

    def test_int4_qdq_per_channel_conv(self):
        """
        Test per-channel quantization of int4 conv weight.
        """
        float_model_path = os.path.join(self._tmp_dir_path, "conv_int4_per_chan.f32.onnx")
        qdq_model_path = os.path.join(self._tmp_dir_path, "conv_int4_per_chan.qdq.onnx")

        inp_shape = [1, 2, 100, 100]
        weight_shape = [2, 2, 20, 20]

        weight_data = np.linspace(-1.5, 1.5, num=1600, dtype=np.float32).reshape(weight_shape)
        bias_data = np.array([-10.0, 10.0], dtype=np.float32)
        float_model = self.build_conv_test_model(inp_shape, weight_data, bias_data)

        onnx.checker.check_model(float_model, True)
        onnx.save_model(float_model, float_model_path)

        data_reader = self.input_feeds(3, {"input_0": inp_shape}, np.float32)

        per_chan_axis = 0
        tensor_quant_overrides = {
            "weight": [{"quant_type": QuantType.QInt4, "axis": per_chan_axis}],  # Quantize weight to INT4 (per-channel)
        }
        quantize_static(
            float_model_path,
            qdq_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt16,
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=[node.op_type for node in float_model.graph.node],
            extra_options={
                "TensorQuantOverrides": tensor_quant_overrides,
            },
        )

        qdq_node_counts = {"QuantizeLinear": 2, "DequantizeLinear": 4}
        check_op_type_count(self, qdq_model_path, **qdq_node_counts)

        qdq_model = onnx.load_model(qdq_model_path)
        onnx.checker.check_model(qdq_model, True)

        initializers = {init.name: init for init in qdq_model.graph.initializer}

        # Check that the weight's zero-point data type is INT4 and has 2 elems
        weight_zp_init = initializers["weight_zero_point"]
        self.assertEqual(weight_zp_init.data_type, onnx.TensorProto.INT4)
        self.assertEqual(weight_zp_init.dims[0], 2)

        # Check that the weight's scale data type is FLOAT and has 2 elems
        weight_scale_init = initializers["weight_scale"]
        self.assertEqual(weight_scale_init.data_type, onnx.TensorProto.FLOAT)
        self.assertEqual(weight_scale_init.dims[0], 2)

        # Check that INT4 weights take up approximately 50% the size of INT8 weights.
        # Using protobuf's ByteSize() is not exact because it includes other fields in the proto message.
        unpacked_size = 1
        for dim in weight_shape:
            unpacked_size *= dim

        weight_quant_init = initializers["weight_quantized"]
        size_ratio = weight_quant_init.ByteSize() / unpacked_size
        self.assertLess(size_ratio, 0.55)

    def test_json_serialization(self):
        td = TensorData(lowest=np.array([0.1], dtype=np.float32), highest=np.array([1.1], dtype=np.float32))
        new_calibrate_tensors_range = TensorsData(CalibrationMethod.MinMax, {"td": td})
        write_calibration_table(new_calibrate_tensors_range)


class TestAdjustWeightScaleForInt32Bias(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.qdq.adj_int32_bias_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def build_conv_test_model(
        self,
        input0_shape: list[int],
        weight_shape: list[int],
        onnx_float_type: onnx.TensorProto.DataType,
    ):
        np_float_type = onnx.helper.tensor_dtype_to_np_dtype(onnx_float_type)
        input_0 = onnx.helper.make_tensor_value_info("input_0", onnx_float_type, input0_shape)
        output_0 = onnx.helper.make_tensor_value_info("output_0", onnx_float_type, None)

        tiny_value = 1e-7 if np_float_type == np.float32 else 0.007782
        # weight_scale = 2*tiny_value / 255.0 = 7.84313725490196e-10

        weight_data = np.full(weight_shape, tiny_value, dtype=np_float_type)
        with np.nditer(weight_data, op_flags=["readwrite"]) as it:
            for i, x in enumerate(it):
                if i % 2 == 0:
                    x[...] = -x

        weight = onnx.numpy_helper.from_array(weight_data, "weight")

        # if we set input_scale to 0.05, then normally bias_scale would be
        # (input_scale * weight_scale) => (0.05 * 7.84314e-10) => 3.9215686274509805e-11
        #
        # If we quantize the f32 bias with this bias_scale, we get
        # [5.0/bias_scale, 4.0/bias_scale] = [127500000000, 102000000000]. These quantized bias values exceed the
        # range of int32.
        #
        # The ORT quantization tool will clamp these out-of-bounds values to int32::max(),
        # which can be very inaccurate.
        bias_shape = [weight_shape[0]]
        bias_data = np.ones(bias_shape, dtype=np_float_type)
        with np.nditer(bias_data, op_flags=["readwrite"]) as it:
            for i, x in enumerate(it):
                if i % 2 == 0:
                    x[...] = 5.0 if np_float_type == np.float32 else 1400
                else:
                    x[...] = -4.5 if np_float_type == np.float32 else -1200

        bias = onnx.numpy_helper.from_array(bias_data, "bias")

        conv_node = onnx.helper.make_node("Conv", ["input_0", "weight", "bias"], ["output_0"], name="Conv0")
        graph = onnx.helper.make_graph(
            [conv_node],
            "Convfloat",
            [input_0],
            [output_0],
            initializer=[weight, bias],
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_adjust_weight_scale_for_int32_bias(self):
        """
        Test adjustment of weight input's scale to ensure int32 bias's scale is not too small.
        """
        test_configs = [
            (onnx.TensorProto.FLOAT, True),
            (onnx.TensorProto.FLOAT, False),
            (onnx.TensorProto.FLOAT16, True),
            (onnx.TensorProto.FLOAT16, False),
        ]

        for float_type, per_channel in test_configs:
            with self.subTest(float_type=float_type, per_channel=per_channel):
                label = f"_f{float_type}_perchannel{per_channel}"
                float_model_path = os.path.join(self._tmp_dir_path, f"conv{label}.float.onnx")
                qdq_model_path = os.path.join(self._tmp_dir_path, f"conv{label}.qdq.onnx")

                # Create float model with a Conv that has tiny weight values.
                # This tiny weight scale would normally create a very small bias scale that will saturate
                # bias's int32 range. But, the qdq_quantizer adjusts the weight's scale to ensure this doesn't happen.
                input0_shape = [1, 2, 4, 4]
                weight_shape = [2, 2, 2, 2]
                float_model = self.build_conv_test_model(input0_shape, weight_shape, float_type)
                onnx.save_model(float_model, float_model_path)

                # Create a data reader
                np_float_type = onnx.helper.tensor_dtype_to_np_dtype(float_type)
                input0_rmin = 0.0
                input0_scale = 0.05 if float_type == onnx.TensorProto.FLOAT else 0.01
                input0_rmax = (input0_scale * 255.0) + input0_rmin
                input_data_list = [
                    {"input_0": np.full(input0_shape, input0_rmin, dtype=np_float_type)},
                    {"input_0": np.full(input0_shape, (input0_rmax - input0_rmin) / 2.0, dtype=np_float_type)},
                    {"input_0": np.full(input0_shape, input0_rmax, dtype=np_float_type)},
                ]
                data_reader = TestDataFeeds(input_data_list)

                # quantize model to QDQ
                quantize_static(
                    float_model_path,
                    qdq_model_path,
                    data_reader,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=per_channel,
                )

                # Check correctness
                data_reader.rewind()
                check_model_correctness(self, float_model_path, qdq_model_path, data_reader.get_next())

    def build_model_convs_share_bias(
        self,
        input0_shape: list[int],
        weight_shape: list[int],
        onnx_float_type: onnx.TensorProto.DataType,
    ):
        np_float_type = onnx.helper.tensor_dtype_to_np_dtype(onnx_float_type)
        input_0 = onnx.helper.make_tensor_value_info("input_0", onnx_float_type, input0_shape)
        output_0 = onnx.helper.make_tensor_value_info("output_0", onnx_float_type, None)
        output_1 = onnx.helper.make_tensor_value_info("output_1", onnx_float_type, None)

        weight_0_data = np.ones(weight_shape, dtype=np_float_type)
        weight_0 = onnx.numpy_helper.from_array(weight_0_data, "weight_0")

        weight_1_data = np.full(weight_shape, 0.5, dtype=np_float_type)
        weight_1 = onnx.numpy_helper.from_array(weight_1_data, "weight_1")

        bias_shape = [weight_shape[0]]
        bias_data = np.ones(bias_shape, dtype=np_float_type)
        bias_shared = onnx.numpy_helper.from_array(bias_data, "bias_shared")

        conv_0_node = onnx.helper.make_node("Conv", ["input_0", "weight_0", "bias_shared"], ["output_0"], name="Conv0")
        conv_1_node = onnx.helper.make_node("Conv", ["input_0", "weight_1", "bias_shared"], ["output_1"], name="Conv1")
        graph = onnx.helper.make_graph(
            [conv_0_node, conv_1_node],
            "ConvWithSharedBiasToDup",
            [input_0],
            [output_0, output_1],
            initializer=[weight_0, weight_1, bias_shared],
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_dup_shared_bias(self):
        """
        Test duplicating a bias that is shared by two nodes that want to quantize their bias to int32.
        """
        float_model_path = os.path.join(self._tmp_dir_path, "convs_share_bias.float.onnx")
        qdq_model_path = os.path.join(self._tmp_dir_path, "convs_share_bias.qdq.onnx")

        # Create float model with a Convs that share a bias input. The QDQ quantizer should add a
        # duplicate bias so that each node has its own.
        input0_shape = [1, 2, 4, 4]
        weight_shape = [2, 2, 2, 2]
        float_model = self.build_model_convs_share_bias(input0_shape, weight_shape, onnx.TensorProto.FLOAT)
        onnx.save_model(float_model, float_model_path)

        # Create a data reader
        input0_rmin = 0.0
        input0_scale = 0.05
        input0_rmax = (input0_scale * 255.0) + input0_rmin
        input_data_list = [
            {"input_0": np.full(input0_shape, input0_rmin, dtype=np.float32)},
            {"input_0": np.full(input0_shape, (input0_rmax - input0_rmin) / 2.0, dtype=np.float32)},
            {"input_0": np.full(input0_shape, input0_rmax, dtype=np.float32)},
        ]
        data_reader = TestDataFeeds(input_data_list)

        # quantize model to QDQ
        quantize_static(
            float_model_path,
            qdq_model_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )

        qdq_model = onnx.load_model(qdq_model_path)
        bias_names = set()

        for node in qdq_model.graph.node:
            if node.op_type == "DequantizeLinear" and node.input[0].startswith("bias_shared"):
                bias_names.add(node.input[0])

        self.assertEqual(len(bias_names), 2)


class TestQDQPrequantWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.qdq.prequant_weight")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def build_conv_model(
        self,
        inp_shape: list[int],
        weight_quant_data: np.ndarray,
        weight_scale_data: np.ndarray,
        weight_zp_data: np.ndarray,
        bias_data: np.ndarray,
        float_type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT,
    ):
        """
        Builds a model with a Conv that has a pre-quantized constant weight input.
        """
        input_0 = onnx.helper.make_tensor_value_info("input_0", float_type, inp_shape)
        output_0 = onnx.helper.make_tensor_value_info("output_0", float_type, None)
        weight_quant = onnx.numpy_helper.from_array(weight_quant_data, "weight_quant")
        weight_scale = onnx.numpy_helper.from_array(weight_scale_data, "weight_scale")
        weight_zp = onnx.numpy_helper.from_array(weight_zp_data, "weight_zp")
        bias = onnx.numpy_helper.from_array(bias_data, "bias")

        dq_node = onnx.helper.make_node(
            "DequantizeLinear", ["weight_quant", "weight_scale", "weight_zp"], ["weight_dequant"], name="DQ0"
        )
        conv_node = onnx.helper.make_node("Conv", ["input_0", "weight_dequant", "bias"], ["output_0"], name="Conv0")
        graph = onnx.helper.make_graph(
            [dq_node, conv_node],
            "ConvPreQuantWeight",
            [input_0],
            [output_0],
            initializer=[weight_quant, weight_scale, weight_zp, bias],
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)

        return onnx.shape_inference.infer_shapes(model)

    def build_conv_dynamic_weight_model(
        self,
        input_quant_data: np.ndarray,
        input_scale_data: np.ndarray,
        input_zp_data: np.ndarray,
        weight_shape: list[int],
        bias_data: np.ndarray,
        float_type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT,
    ):
        """
        Builds a model with a Conv that has a dynamic float weight input, but a constant
        pre-quantized input[0].
        """
        dyn_weight = onnx.helper.make_tensor_value_info("dyn_weight", float_type, weight_shape)
        output_0 = onnx.helper.make_tensor_value_info("output_0", float_type, None)
        input_quant = onnx.numpy_helper.from_array(input_quant_data, "input_quant")
        input_scale = onnx.numpy_helper.from_array(input_scale_data, "input_scale")
        input_zp = onnx.numpy_helper.from_array(input_zp_data, "input_zp")
        bias = onnx.numpy_helper.from_array(bias_data, "bias")

        dq_node = onnx.helper.make_node(
            "DequantizeLinear", ["input_quant", "input_scale", "input_zp"], ["input_dequant"], name="DQ0"
        )
        conv_node = onnx.helper.make_node("Conv", ["input_dequant", "dyn_weight", "bias"], ["output_0"], name="Conv0")
        graph = onnx.helper.make_graph(
            [dq_node, conv_node],
            "ConvPreQuantInput_DynamicWeight",
            [dyn_weight],
            [output_0],
            initializer=[input_quant, input_scale, input_zp, bias],
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)

        return onnx.shape_inference.infer_shapes(model)

    def test_quantize_with_prequantized_weights(self):
        """
        Test quantization of Conv with pre-quantized weights.
        """
        rng = np.random.default_rng(123)
        test_configs = [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16]

        for float_type in test_configs:
            with self.subTest(float_type=float_type):
                label = f"_{onnx.TensorProto.DataType.Name(float_type)}"
                float_model_path = os.path.join(self._tmp_dir_path, f"conv.f32.prequant_weight{label}.onnx")
                qdq_model_path = os.path.join(self._tmp_dir_path, f"conv.prequant_weight{label}.qdq.onnx")

                inp_shape = [1, 2, 100, 100]
                weight_shape = [2, 2, 20, 20]
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(float_type)

                # range = 2.0, scale = 2/254, zp = 0
                weight_scale_data = np.array(2 / 254, dtype=np_dtype)
                weight_zp_data = np.array(0, dtype=np.int8)
                weight_data = np.linspace(-1.0, 1.0, num=1600, dtype=np_dtype).reshape(weight_shape)
                weight_quant_data = quantize_nparray(
                    onnx.TensorProto.INT8, weight_data, weight_scale_data, weight_zp_data
                )

                bias_data = np.array([-10.0, 10.0], dtype=np_dtype)
                float_model = self.build_conv_model(
                    inp_shape, weight_quant_data, weight_scale_data, weight_zp_data, bias_data, float_type
                )

                onnx.checker.check_model(float_model, True)
                onnx.save_model(float_model, float_model_path)

                # Check that the input model only has a pre-quantized weight and save its scale/zero-point
                # to check that it doesn't change after quantization.
                float_node_counts = {"QuantizeLinear": 0, "DequantizeLinear": 1}
                check_op_type_count(self, float_model_path, **float_node_counts)
                conv_node_original = next((node for node in float_model.graph.node if node.op_type == "Conv"), None)
                self.assertNotEqual(conv_node_original, None)

                _, producers_original = get_tensor_consumers_and_producers(float_model)
                weight_dq_node_original = producers_original.get(conv_node_original.input[1], None)
                initializers_original = {initializer.name: initializer for initializer in float_model.graph.initializer}
                scale_name_original = weight_dq_node_original.input[1]
                scale_val_original = onnx.numpy_helper.to_array(initializers_original[scale_name_original])
                zp_name_original = weight_dq_node_original.input[2]
                zp_val_original = onnx.numpy_helper.to_array(initializers_original[zp_name_original])

                input_data_list = [
                    {"input_0": rng.uniform(-10.0, 10.0, inp_shape).astype(np_dtype)},
                ]
                data_reader = TestDataFeeds(input_data_list)

                quantize_static(
                    float_model_path,
                    qdq_model_path,
                    data_reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv"],
                )

                # The final model should have everything quantized
                qdq_node_counts = {"QuantizeLinear": 2, "DequantizeLinear": 4}
                check_op_type_count(self, qdq_model_path, **qdq_node_counts)

                # Check that the pre-quantized weight still has the same scale/zp after quantization
                qdq_model = onnx.load_model(qdq_model_path)
                conv_node = next((node for node in qdq_model.graph.node if node.op_type == "Conv"), None)
                self.assertNotEqual(conv_node, None)

                _, producers = get_tensor_consumers_and_producers(qdq_model)
                weight_dq_node = producers.get(conv_node.input[1], None)
                initializers = {initializer.name: initializer for initializer in qdq_model.graph.initializer}

                scale_name = weight_dq_node.input[1]
                self.assertEqual(scale_name, scale_name_original)
                scale_val = onnx.numpy_helper.to_array(initializers[scale_name])
                self.assertEqual(scale_val, scale_val_original)

                zp_name = weight_dq_node.input[2]
                self.assertEqual(zp_name, zp_name_original)
                zp_val = onnx.numpy_helper.to_array(initializers[zp_name])
                self.assertEqual(zp_val, zp_val_original)

    def test_quantize_with_prequantized_input(self):
        """
        Test quantization of Conv with pre-quantized input and dynamic weight.
        """
        rng = np.random.default_rng(123)
        test_configs = [
            (onnx.TensorProto.FLOAT, False),
            (onnx.TensorProto.FLOAT16, False),
            (onnx.TensorProto.FLOAT, True),
            (onnx.TensorProto.FLOAT16, True),
        ]

        for float_type, convert_weight_qtype in test_configs:
            with self.subTest(float_type=float_type):
                convert_label = "_convert_qtype" if convert_weight_qtype else ""
                label = f"_{onnx.TensorProto.DataType.Name(float_type)}{convert_label}"
                float_model_path = os.path.join(self._tmp_dir_path, f"conv.f32.prequant_input{label}.onnx")
                qdq_model_path = os.path.join(self._tmp_dir_path, f"conv.prequant_input{label}.qdq.onnx")

                inp_shape = [1, 2, 40, 40]
                weight_shape = [2, 2, 20, 20]
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(float_type)

                # range = 3.0, scale = 3/255, zp = 127
                input_scale_data = np.array(3 / 255, dtype=np_dtype)
                input_zp_data = np.array(127, dtype=np.uint8)
                input_data = np.linspace(-1.5, 1.5, num=3200, dtype=np_dtype).reshape(inp_shape)
                input_quant_data = quantize_nparray(onnx.TensorProto.UINT8, input_data, input_scale_data, input_zp_data)

                bias_data = np.array([-10.0, 10.0], dtype=np_dtype)
                float_model = self.build_conv_dynamic_weight_model(
                    input_quant_data, input_scale_data, input_zp_data, weight_shape, bias_data, float_type
                )

                onnx.checker.check_model(float_model, True)
                onnx.save_model(float_model, float_model_path)

                # Check that the input model only has a pre-quantized input and save its scale/zero-point
                # to check that it doesn't change after quantization.
                float_node_counts = {"QuantizeLinear": 0, "DequantizeLinear": 1}
                check_op_type_count(self, float_model_path, **float_node_counts)
                conv_node_original = next((node for node in float_model.graph.node if node.op_type == "Conv"), None)
                self.assertNotEqual(conv_node_original, None)

                _, producers_original = get_tensor_consumers_and_producers(float_model)
                input_dq_node_original = producers_original.get(conv_node_original.input[0], None)
                initializers_original = {initializer.name: initializer for initializer in float_model.graph.initializer}
                scale_name_original = input_dq_node_original.input[1]
                scale_val_original = onnx.numpy_helper.to_array(initializers_original[scale_name_original])
                zp_name_original = input_dq_node_original.input[2]
                zp_val_original = onnx.numpy_helper.to_array(initializers_original[zp_name_original])

                # Create data reader with random input calibration data.
                dyn_weight_data_list = [
                    {"dyn_weight": rng.uniform(-10.0, 10.0, weight_shape).astype(np_dtype)},
                ]
                data_reader = TestDataFeeds(dyn_weight_data_list)

                extra_options = {}
                if convert_weight_qtype:
                    # Test converting the dynamic weight's quantization type, which results in
                    # dyn_weight -> Q(u16) -> DQ(f32) -> Q(u8) -> DQ(f32) -> Conv
                    extra_options["TensorQuantOverrides"] = {
                        "dyn_weight": [{"quant_type": QuantType.QUInt16, "convert": {"quant_type": QuantType.QUInt8}}],
                    }

                quantize_static(
                    float_model_path,
                    qdq_model_path,
                    data_reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv"],
                    extra_options=extra_options,
                )

                # The final model should have everything quantized
                qdq_node_counts = {"QuantizeLinear": 2, "DequantizeLinear": 4}
                if convert_weight_qtype:
                    qdq_node_counts["QuantizeLinear"] += 1
                    qdq_node_counts["DequantizeLinear"] += 1

                check_op_type_count(self, qdq_model_path, **qdq_node_counts)

                # Check that the pre-quantized input still has the same scale/zp after quantization
                qdq_model = onnx.load_model(qdq_model_path)
                conv_node = next((node for node in qdq_model.graph.node if node.op_type == "Conv"), None)
                self.assertNotEqual(conv_node, None)

                _, producers = get_tensor_consumers_and_producers(qdq_model)
                input_dq_node = producers.get(conv_node.input[0], None)
                initializers = {initializer.name: initializer for initializer in qdq_model.graph.initializer}

                scale_name = input_dq_node.input[1]
                self.assertEqual(scale_name, scale_name_original)
                scale_val = onnx.numpy_helper.to_array(initializers[scale_name])
                self.assertEqual(scale_val, scale_val_original)

                zp_name = input_dq_node.input[2]
                self.assertEqual(zp_name, zp_name_original)
                zp_val = onnx.numpy_helper.to_array(initializers[zp_name])
                self.assertEqual(zp_val, zp_val_original)


if __name__ == "__main__":
    unittest.main()
