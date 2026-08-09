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
from op_test_utils import (
    TestDataFeeds,
    check_model_correctness,
    check_op_nodes,
    check_op_type_count,
    check_qtype_by_node_type,
)

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestOpArgMax(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_argmax(self, output_model_path, input_shape, output_shape):
        #     (input)
        #        |
        #       Conv
        #        |
        #      ArgMax
        #        |
        #     (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        # make Conv node
        conv_weight_name = "conv_weight"
        conv_weight_arr = np.random.randint(-1, 2, [32, 256, 1, 1]).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name=conv_weight_name)
        conv_output_name = "conv_output"
        conv_inputs = [input_name, conv_weight_name]
        conv_outputs = [conv_output_name]
        conv_name = "conv_node"
        conv_node = onnx.helper.make_node(
            "Conv",
            conv_inputs,
            conv_outputs,
            dilations=[1, 1],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            name=conv_name,
        )

        # make ArgMax node
        argmax_inputs = [conv_output_name]
        argmax_outputs = [output_name]
        argmax_name = "argmax_node"
        argmax_node = onnx.helper.make_node(
            "ArgMax",
            argmax_inputs,
            argmax_outputs,
            axis=3,
            keepdims=0,
            name=argmax_name,
        )

        initializers = [conv_weight_initializer]

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.INT64, output_shape)
        graph_name = "ArgMax_Quant_Test"
        graph = helper.make_graph(
            [conv_node, argmax_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def quantize_argmax_test(self, activation_type, weight_type, extra_options={}):  # noqa: B006
        np.random.seed(1)
        model_fp32_path = "argmax_fp32.onnx"

        self.construct_model_argmax(model_fp32_path, [1, 256, 128, 128], [1, 32, 128])

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_uint8_path = f"argmax_{activation_type_str}{weight_type_str}.onnx"
        model_uint8_qdq_path = f"argmax_{activation_type_str}{weight_type_str}_qdq.onnx"
        model_uint8_qdq_trt_path = f"argmax_{activation_type_str}{weight_type_str}_qdq_trt.onnx"

        # Verify QOperator mode
        data_reader = self.input_feeds(1, {"input": [1, 256, 128, 128]})
        quantize_static(
            model_fp32_path,
            model_uint8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        # make sure argmax become xint8 operator, its input name could tell that
        check_op_nodes(
            self,
            model_uint8_path,
            lambda node: not (node.name == "argmax_node" and node.input[0] == "conv_output"),
        )
        qnode_counts = {"QuantizeLinear": 1, "QLinearConv": 1, "ArgMax": 1}
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
        qdqnode_counts = {"QuantizeLinear": 2, "DequantizeLinear": 3, "ArgMax": 1}
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

        # Verify QDQ mode for TensorRT
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_uint8_qdq_trt_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
            op_types_to_quantize=["ArgMax"],
        )
        qdqnode_counts = {"QuantizeLinear": 1, "DequantizeLinear": 1, "ArgMax": 1}
        check_op_type_count(self, model_uint8_qdq_trt_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_uint8_qdq_trt_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_qdq_trt_path, data_reader.get_next())

    def test_quantize_argmax(self):
        self.quantize_argmax_test(QuantType.QUInt8, QuantType.QUInt8)

    def test_quantize_argmax_s8s8(self):
        self.quantize_argmax_test(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()
