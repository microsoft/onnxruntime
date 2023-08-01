# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save
from op_test_utils import (
    check_model_correctness,
    check_op_type_count,
    check_qtype_by_node_type,
    input_feeds_neg_one_zero_one,
)

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestConcatModel(unittest.TestCase):
    def construct_model(self, model_path):
        #          (input)
        #         /    |  \
        #        /     |   \
        #       /      |    \
        #      /       |     \
        #  Conv(1)  Conv(2)  conv(3)
        #       \      |     /
        #         \    |    /
        #           \  |   /
        #            Concat
        #              |
        #             Relu
        #              |
        #           Identity
        #              |
        #           (output)
        initializers = []
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 15, 15])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 13, 13, 13])

        # Conv1 output [1, 2, 13, 13]
        conv1_weight_initializer = numpy_helper.from_array(
            np.random.randint(-1, 2, [2, 3, 3, 3]).astype(np.float32),
            name="conv1_weight",
        )
        conv1_node = helper.make_node("Conv", ["input", "conv1_weight"], ["conv1_output"], name="conv1_node")

        # Conv2 output [1, 5, 13, 13]
        conv2_weight_initializer = numpy_helper.from_array(
            np.random.randint(-1, 2, [5, 3, 3, 3]).astype(np.float32),
            name="conv2_weight",
        )
        conv2_node = helper.make_node("Conv", ["input", "conv2_weight"], ["conv2_output"], name="conv2_node")

        # Conv3 output [1, 6, 13, 13]
        conv3_weight_initializer = numpy_helper.from_array(
            np.random.randint(-1, 2, [6, 3, 3, 3]).astype(np.float32),
            name="conv3_weight",
        )
        conv3_node = helper.make_node("Conv", ["input", "conv3_weight"], ["conv3_output"], name="conv3_node")

        concat_node = helper.make_node(
            "Concat",
            ["conv1_output", "conv2_output", "conv3_output"],
            ["concat_output"],
            name="concat_node",
            axis=1,
        )

        relu_node = helper.make_node("Relu", ["concat_output"], ["relu_output"], name="relu_node")
        identity_node = helper.make_node("Identity", ["relu_output"], ["output"], name="identity_node")

        initializers = [
            conv1_weight_initializer,
            conv2_weight_initializer,
            conv3_weight_initializer,
        ]
        graph = helper.make_graph(
            [conv1_node, conv2_node, conv3_node, concat_node, relu_node, identity_node],
            "qlinear_concat_op_test",
            [input],
            [output],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        save(model, model_path)

    def quantize_concat_test(self, activation_type, weight_type, extra_options={}):  # noqa: B006
        np.random.seed(1)
        model_fp32_path = "concat_fp32.onnx"
        self.construct_model(model_fp32_path)
        data_reader = input_feeds_neg_one_zero_one(1, {"input": [1, 3, 15, 15]})

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_q8_path = f"concat_{activation_type_str}{weight_type_str}.onnx"
        model_q8_qdq_path = f"concat_{activation_type_str}{weight_type_str}_qdq.onnx"

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
            "QLinearConv": 3,
            "QuantizeLinear": 1,
            "DequantizeLinear": 1,
            "QLinearConcat": 1,
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
                "QLinearConcat": [
                    ["i", 1, activation_proto_qtype],
                    ["i", 4, activation_proto_qtype],
                    ["i", 7, activation_proto_qtype],
                ]
            }
        )
        check_qtype_by_node_type(self, model_q8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_q8_path, data_reader.get_next())

        # Verify QDQ mode
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_q8_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        relu_count = 0
        if activation_type == QuantType.QInt8 and extra_options.get("ActivationSymmetric", False):
            relu_count = 1
        qdqnode_counts = {
            "Conv": 3,
            "QuantizeLinear": 5 + relu_count,
            "DequantizeLinear": 8 + relu_count,
            "Concat": 1,
            "Relu": 0 + relu_count,
        }
        check_op_type_count(self, model_q8_qdq_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_q8_qdq_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_q8_qdq_path, data_reader.get_next())

    def test_quantize_concat(self):
        self.quantize_concat_test(QuantType.QUInt8, QuantType.QUInt8, extra_options={})

    def test_quantize_concat_s8s8(self):
        self.quantize_concat_test(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()
