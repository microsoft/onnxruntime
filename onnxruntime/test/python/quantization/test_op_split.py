# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, save
from op_test_utils import (
    check_model_correctness,
    check_op_type_count,
    check_qtype_by_node_type,
    input_feeds_neg_one_zero_one,
)

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestONNXModel(unittest.TestCase):
    # input -> conv -> reshape -> split
    def construct_model(self, model_path):
        #             (input)
        #                |
        #               Conv
        #                |
        #             Reshape
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
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [6, 3])

        output_1 = helper.make_tensor_value_info("output_1", TensorProto.FLOAT, [1, 12])
        output_2 = helper.make_tensor_value_info("output_2", TensorProto.FLOAT, [1, 12])
        output_3 = helper.make_tensor_value_info("output_3", TensorProto.FLOAT, [1, 12])

        reshape_shape = "reshape_shape"
        initializers.append(
            onnx.numpy_helper.from_array(
                np.random.randint(-1, 2, [6, 3]).astype(np.float32),
                name="conv_weight",
            )
        )
        conv_node = helper.make_node("Conv", ["input", "conv_weight"], ["conv_output"], name="conv_node")
        initializers.append(onnx.numpy_helper.from_array(np.array([3, 12], dtype=np.int64), name=reshape_shape))
        reshape_node = helper.make_node(
            "Reshape", ["conv_output", reshape_shape], ["reshape_output"], name="reshape_node"
        )
        initializers.append(onnx.numpy_helper.from_array(np.array([1, 1, 1], dtype=np.int64), name="split"))
        split_node = helper.make_node(
            "Split",
            inputs=["reshape_output", "split"],
            outputs=["output_1", "output_2", "output_3"],
            name="split_node",
            axis=0,
        )
        graph = helper.make_graph(
            [conv_node, reshape_node, split_node],
            "qlinear_split_op_test",
            [input],
            [output_1, output_2, output_3],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        save(model, model_path)

    def quantize_split_test(self, activation_type, weight_type, extra_options={}):  # noqa: B006
        np.random.seed(1)
        model_fp32_path = "split_fp32.onnx"
        self.construct_model(model_fp32_path)
        data_reader = input_feeds_neg_one_zero_one(1, {"input": [6, 3]})

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_uint8_path = f"split_{activation_type_str}{weight_type_str}.onnx"
        model_uint8_qdq_path = f"split_{activation_type_str}{weight_type_str}_qdq.onnx"

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
            "QuantizeLinear": 6,
            "DequantizeLinear": 7,
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
