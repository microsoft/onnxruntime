#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestOpSqueezeUnsqueeze(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_conv_squeezes(
        self,
        output_model_path,
        conv_input_shape,
        conv_weight_shape,
        conv_output_shape,
        opset=13,
    ):
        #             (input)
        #            /   |     \
        #         Conv1 conv2    conv3
        #           |     |         |
        #       Squeeze1 Squeeze2   |
        #           \      /        |
        #             add1          |
        #               |           |
        #              Unsqueeze    |
        #                      \    |
        #                       add2
        #                         |
        #                      (output)
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, conv_input_shape)

        conv1_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv1_weight_initializer = onnx.numpy_helper.from_array(conv1_weight_arr, name="conv1_weight")
        conv1_node = onnx.helper.make_node("Conv", ["input", "conv1_weight"], ["conv1_output"], name="conv1_node")

        conv2_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv2_weight_initializer = onnx.numpy_helper.from_array(conv2_weight_arr, name="conv2_weight")
        conv2_node = onnx.helper.make_node("Conv", ["input", "conv2_weight"], ["conv2_output"], name="conv2_node")

        conv3_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv3_weight_initializer = onnx.numpy_helper.from_array(conv3_weight_arr, name="conv3_weight")
        conv3_node = onnx.helper.make_node("Conv", ["input", "conv3_weight"], ["conv3_output"], name="conv3_node")

        if opset >= 13:
            squeeze_axes_initializer = onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name="squeeze_axes")
            squeeze1_node = helper.make_node(
                "Squeeze",
                ["conv1_output", "squeeze_axes"],
                ["squeeze1_output"],
                name="suqeeze1_node",
            )
            squeeze2_node = helper.make_node(
                "Squeeze",
                ["conv2_output", "squeeze_axes"],
                ["squeeze2_output"],
                name="suqeeze2_node",
            )
        else:
            squeeze1_node = helper.make_node(
                "Squeeze",
                ["conv1_output"],
                ["squeeze1_output"],
                name="suqeeze1_node",
                axes=[0],
            )
            squeeze2_node = helper.make_node(
                "Squeeze",
                ["conv2_output"],
                ["squeeze2_output"],
                name="suqeeze2_node",
                axes=[0],
            )

        add1_node = helper.make_node(
            "Add",
            ["squeeze1_output", "squeeze2_output"],
            ["add1_output"],
            name="add1_node",
        )
        if opset >= 13:
            unsqueeze_node = helper.make_node(
                "Unsqueeze",
                ["add1_output", "squeeze_axes"],
                ["unsqueeze_output"],
                name="unsqueeze_node",
            )
        else:
            unsqueeze_node = helper.make_node(
                "Unsqueeze",
                ["add1_output"],
                ["unsqueeze_output"],
                name="unsqueeze_node",
                axes=[0],
            )

        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, conv_output_shape)
        add2_node = helper.make_node("Add", ["unsqueeze_output", "conv3_output"], ["output"], name="add2_node")

        initializers = [
            conv1_weight_initializer,
            conv2_weight_initializer,
            conv3_weight_initializer,
        ]
        if opset >= 13:
            initializers.append(squeeze_axes_initializer)
        graph = helper.make_graph(
            [
                conv1_node,
                conv2_node,
                conv3_node,
                squeeze1_node,
                squeeze2_node,
                add1_node,
                unsqueeze_node,
                add2_node,
            ],
            "TestOpSuqeezes_test_model",
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        model.ir_version = 7  # use stable onnx ir version
        onnx.save(model, output_model_path)

    def run_quantize_squeezes_of_opset(
        self,
        opset=13,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        extra_options={},  # noqa: B006
    ):
        np.random.seed(1)

        model_fp32_path = f"squeezes_opset{opset}_fp32.onnx"
        self.construct_model_conv_squeezes(model_fp32_path, [1, 2, 26, 42], [3, 2, 3, 3], [1, 3, 24, 40], opset=opset)

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_uint8_path = f"squeezes_opset{opset}_{activation_type_str}{weight_type_str}.onnx"
        model_uint8_qdq_path = f"squeezes_opset{opset}_{activation_type_str}{weight_type_str}_qdq.onnx"

        # Verify QOperator mode
        data_reader = self.input_feeds(1, {"input": [1, 2, 26, 42]})
        quantize_static(
            model_fp32_path,
            model_uint8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        # make sure squeezes become xint8 operator, its input name could tell that
        qnode_counts = {"QuantizeLinear": 1, "DequantizeLinear": 1}
        check_op_type_count(self, model_uint8_path, **qnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        qnode_io_qtypes.update({"DequantizeLinear": [["i", 2, activation_proto_qtype]]})
        check_qtype_by_node_type(self, model_uint8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(
            self,
            model_fp32_path,
            model_uint8_path,
            data_reader.get_next(),
            rtol=0.01,
            atol=0.5,
        )

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
        qdqnode_counts = {"Conv": 3, "QuantizeLinear": 9, "DequantizeLinear": 12}
        check_op_type_count(self, model_uint8_qdq_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_uint8_qdq_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(
            self,
            model_fp32_path,
            model_uint8_qdq_path,
            data_reader.get_next(),
            rtol=0.01,
            atol=0.5,
        )

    def test_quantize_squeeze_unsqueeze(self):
        self.run_quantize_squeezes_of_opset(11)
        self.run_quantize_squeezes_of_opset(13)

    def test_quantize_squeeze_unsqueeze_s8s8(self):
        self.run_quantize_squeezes_of_opset(
            11,
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )
        self.run_quantize_squeezes_of_opset(
            13,
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()
