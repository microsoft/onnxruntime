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


class TestOpRelu(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_gemm(self, output_model_path):
        #      (input)
        #         |
        #        Gemm
        #         |
        #        Relu
        #         |
        #        Gemm
        #         |
        #      (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_gemm(input_name, weight_shape, weight_name, bias_shape, bias_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            bias_data = np.random.normal(0, 0.1, bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))

            return onnx.helper.make_node(
                "Gemm",
                [input_name, weight_name, bias_name],
                [output_name],
                alpha=1.0,
                beta=1.0,
                transB=1,
            )

        # make gemm1 node
        gemm1_output_name = "gemm1_output"
        gemm1_node = make_gemm(
            input_name,
            [100, 10],
            "linear1.weight",
            [100],
            "linear1.bias",
            gemm1_output_name,
        )

        # make Relu
        relu_output = "relu_output"
        relu_node = onnx.helper.make_node("Relu", [gemm1_output_name], [relu_output])

        # make gemm2 node
        gemm2_node = make_gemm(
            relu_output,
            [10, 100],
            "linear2.weight",
            [10],
            "linear2.bias",
            output_name,
        )

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [-1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [-1, 10])
        graph_name = "relu_test"
        graph = helper.make_graph(
            [gemm1_node, relu_node, gemm2_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)

    def static_quant_test(
        self,
        model_fp32_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},  # noqa: B006
    ):
        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_int8_path = f"relu_fp32.quant_{activation_type_str}{weight_type_str}.onnx"

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        qdq_count = 1 if activation_type == QuantType.QUInt8 else 2
        relu_count = 0 if activation_type == QuantType.QUInt8 else 1
        quant_nodes = {"QGemm": 2, "QuantizeLinear": qdq_count, "DequantizeLinear": qdq_count, "Relu": relu_count}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        qnode_io_qtypes.update({"DequantizeLinear": [["i", 2, activation_proto_qtype]]})
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next())

    def static_quant_test_qdq(
        self,
        model_fp32_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},  # noqa: B006
    ):
        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_int8_path = f"relu_fp32.quant_dqd_{activation_type_str}{weight_type_str}.onnx"

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        relu_count = 0 if activation_type == QuantType.QUInt8 else 1
        q_count = 3 if activation_type == QuantType.QUInt8 else 4
        dq_count = 7 if activation_type == QuantType.QUInt8 else 8
        quant_nodes = {"Gemm": 2, "QuantizeLinear": q_count, "DequantizeLinear": dq_count, "Relu": relu_count}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next())

    def test_quantize_gemm(self):
        np.random.seed(1)
        model_fp32_path = "relu_fp32.onnx"
        self.construct_model_gemm(model_fp32_path)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )
        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )

    def test_quantize_relu_s8s8(self):
        np.random.seed(1)
        model_fp32_path = "relu_fp32.onnx"
        self.construct_model_gemm(model_fp32_path)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )
        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()
