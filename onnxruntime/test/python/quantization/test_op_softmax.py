#!/usr/bin/env python
"""
Softmax quantization test case
"""
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestOpSoftmax(unittest.TestCase):
    """_summary_
    unittest (softmax): quantization of QDQ and Qop with u8 and s8
    """

    def input_feeds(self, n_repeat, name2shape):
        input_data_list = []
        for _ in range(n_repeat):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        data_r = TestDataFeeds(input_data_list)
        return data_r

    def construct_model_conv_softmax(
        self,
        output_model_path,
        conv_input_shape,
        conv_weight_shape,
        softmax_input_shape,
        softmax_attributes,
        output_shape,
    ):
        #      (input)
        #          \
        #         Conv
        #        /    \
        #   Identity  Softmax
        #    /            \
        # (identity_out)  (output)
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, conv_input_shape)

        conv_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name="conv1_weight")
        conv_node = onnx.helper.make_node("Conv", ["input", "conv1_weight"], ["conv_output"], name="conv_node")

        identity_out = helper.make_tensor_value_info("identity_out", TensorProto.FLOAT, softmax_input_shape)
        identity_node = helper.make_node("Identity", ["conv_output"], ["identity_out"], name="IdentityNode")

        initializers = [conv_weight_initializer]

        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        softmax_node = helper.make_node(
            "Softmax", ["conv_output"], ["output"], name="softmax_node", **softmax_attributes
        )

        graph = helper.make_graph(
            [conv_node, identity_node, softmax_node],
            "TestOpQuantizersoftmax_test_model",
            [input_tensor],
            [identity_out, output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)])
        model.ir_version = 7  # use stable onnx ir version
        onnx.save(model, output_model_path)

    def quantize_softmax_test_qop(self, activation_type, weight_type, extra_options={}):  # noqa: B006
        np.random.seed(1)
        model_fp32_path = "softmax_fp32.onnx"
        self.construct_model_conv_softmax(
            model_fp32_path,
            [1, 2, 26, 42],
            [3, 2, 3, 3],
            [1, 3, 24, 40],
            {"axis": -2},
            [1, 3, 24, 40],
        )
        data_reader = self.input_feeds(1, {"input": [1, 2, 26, 42]})

        activation_proto_qtype = activation_type.tensor_type
        activation_type_str = str(activation_type)
        weight_type_str = str(weight_type)
        model_q8_path = f"softmax_{activation_type_str}{weight_type_str}.onnx"

        # Verify QOperator mode
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_q8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        qnode_counts = {
            "QLinearConv": 1,
            "QuantizeLinear": 1,
            "DequantizeLinear": 2,
            "QLinearSoftmax": 1,
            "Softmax": 0,
        }
        check_op_type_count(self, model_q8_path, **qnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        qnode_io_qtypes.update(
            {
                "QLinearConv": [
                    ["i", 2, activation_proto_qtype],
                    ["i", 7, activation_proto_qtype],
                    ["o", 0, activation_proto_qtype],
                ]
            }
        )
        qnode_io_qtypes.update(
            {"QLinearSoftmax": [["i", 4, activation_proto_qtype]]}
        )  # shape info note workig on custome ops
        check_qtype_by_node_type(self, model_q8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_q8_path, data_reader.get_next())

    def quantize_softmax_test_qdq(self, activation_type, weight_type, extra_options={}):  # noqa: B006
        np.random.seed(1)
        model_fp32_path = "softmax_fp32.onnx"
        self.construct_model_conv_softmax(
            model_fp32_path,
            [1, 2, 26, 42],
            [3, 2, 3, 3],
            [1, 3, 24, 40],
            {"axis": -2},
            [1, 3, 24, 40],
        )
        data_reader = self.input_feeds(1, {"input": [1, 2, 26, 42]})

        activation_proto_qtype = activation_type.tensor_type
        activation_type_str = str(activation_type)
        weight_type_str = str(weight_type)
        model_qdq_path = f"softmax_qdq_{activation_type_str}{weight_type_str}.onnx"

        # Verify QDQ mode
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        result_model = onnx.load(Path(model_qdq_path))
        qnode_cnt = 0
        dqnode_cnt = 0
        softmax_cnt = 0
        qnode_zeropoints = []
        for node in result_model.graph.node:
            if node.op_type == "QuantizeLinear":
                qnode_cnt += 1
                qnode_zeropoints.append(node.input[2])
            elif node.op_type == "DequantizeLinear":
                dqnode_cnt += 1
            elif node.op_type == "Softmax":
                softmax_cnt += 1
        self.assertEqual(3, qnode_cnt, f"Expected 3 QuantizeLinear nodes, found {qnode_cnt}")
        self.assertEqual(4, dqnode_cnt, f"Expected 4 DequantizeLinear nodes, found {dqnode_cnt}")
        self.assertEqual(1, softmax_cnt, f"Expected 1 Softmax node, found {softmax_cnt}")
        if extra_options.get("ActivationSymmetric", False):
            for tensor in result_model.graph.initializer:
                if tensor.name in qnode_zeropoints:
                    np_value = numpy_helper.to_array(tensor)
                    self.assertEqual(
                        0,
                        np_value,
                        f"QuantizeLinear node zero point value must be 0, found {np_value} instead!",
                    )

        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_qdq_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_qdq_path, data_reader.get_next())

    def test_quantize_softmax(self):
        self.quantize_softmax_test_qop(QuantType.QUInt8, QuantType.QUInt8)
        self.quantize_softmax_test_qdq(QuantType.QUInt8, QuantType.QUInt8)

    def test_quantize_softmax_s8s8(self):
        self.quantize_softmax_test_qop(
            QuantType.QInt8,
            QuantType.QInt8,
        )
        self.quantize_softmax_test_qdq(
            QuantType.QInt8,
            QuantType.QInt8,
        )
        self.quantize_softmax_test_qop(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )
        self.quantize_softmax_test_qdq(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )

    def test_quantize_softmax_qdq_u16u16(self):
        self.quantize_softmax_test_qdq(
            QuantType.QUInt16,
            QuantType.QUInt16,
            extra_options={"UseQDQContribOps": True},
        )

    def test_quantize_softmax_qdq_s16s16(self):
        self.quantize_softmax_test_qdq(
            QuantType.QInt16,
            QuantType.QInt16,
            extra_options={"UseQDQContribOps": True},
        )
        self.quantize_softmax_test_qdq(
            QuantType.QInt16,
            QuantType.QInt16,
            extra_options={"UseQDQContribOps": True,
                "ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()
