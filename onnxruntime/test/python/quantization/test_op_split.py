# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
from onnx import TensorProto, helper, save
from op_test_utils import (
    InputFeedsNegOneZeroOne,
    check_model_correctness,
    check_op_type_count,
    check_qtype_by_node_type,
)

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestONNXModel(unittest.TestCase):
    def construct_model(self, model_path):
        #             (input)
        #                |
        #                |
        #                |
        #              Split
        #           /    |    \
        #          /     |     \
        #         /      |      \
        #        /       |       \
        #       /        |        \
        #      /         |         \
        # (output_1) (output_2) (output_3)

        initializers = []
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 6])
        output_1 = helper.make_tensor_value_info("output_1", TensorProto.FLOAT, [1, 6])
        output_2 = helper.make_tensor_value_info("output_2", TensorProto.FLOAT, [1, 6])
        output_3 = helper.make_tensor_value_info("output_3", TensorProto.FLOAT, [1, 6])

        split_node = helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output_1", "output_2", "output_3"],
            name="split_node",
            axis=0,
        )
        graph = helper.make_graph(
            [split_node],
            "qlinear_split_op_test",
            [input],
            [output_1, output_2, output_3],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        save(model, model_path)

    def quantize_split_test(self, activation_type, weight_type, extra_options={}):
        np.random.seed(1)
        model_fp32_path = "split_fp32.onnx"
        self.construct_model(model_fp32_path)
        data_reader = InputFeedsNegOneZeroOne(1, {"input": [3, 6]})

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_uint8_path = "split_{}{}.onnx".format(activation_type_str, weight_type_str)
        model_uint8_qdq_path = "split_{}{}_qdq.onnx".format(activation_type_str, weight_type_str)

        # Verify QOperator mode
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_uint8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        qnode_counts = {
            "Split": 1,
            "QuantizeLinear": 1,
            "DequantizeLinear": 3,
        }
        check_op_type_count(self, model_uint8_path, **qnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_uint8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_path, data_reader.get_next())

        # Verify QDQ mode
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_uint8_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        qdqnode_counts = {
            "Split": 1,
            "QuantizeLinear": 4,
            "DequantizeLinear": 4,
        }
        check_op_type_count(self, model_uint8_qdq_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_uint8_qdq_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_qdq_path, data_reader.get_next())

    def test_quantize_split(self):
        self.quantize_split_test(QuantType.QUInt8, QuantType.QUInt8, extra_options={})
        print(__name__)

    def test_quantize_split_s8s8(self):
        self.quantize_split_test(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )
        print(__name__)


if __name__ == "__main__":
    unittest.main()
