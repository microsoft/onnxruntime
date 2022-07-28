#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import TestCaseTempDir, TestDataFeeds, check_model_correctness, check_op_type_count
from pathlib import Path
from onnxruntime.tools import symbolic_shape_infer

from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic


class TestOpLSTM(TestCaseTempDir):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.normal(0, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_lstm_and_matmul(self, output_model_path):
        #      (input)
        #         |
        #        LSTM
        #         |
        #       MatMul
        #         |
        #      (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_lstm_node(input_name, weight_shape, weight_name, bias_shape, bias_name, output_name):
            input_size = 2
            hidden_size = 7
            weight_scale = 0.3
            number_of_gates = 4
            layout = 1

            weight_data = np.random.normal(0, 0.3, [1, number_of_gates * hidden_size, input_size]).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            bias_data = np.random.normal(0, 0.3, [1, number_of_gates * hidden_size, input_size]).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))

            return onnx.helper.make_node("LSTM", [input_name, weight_name, bias_name], [output_name],hidden_size=hidden_size)

        # make lstm node
        lstm_node = make_lstm_node(
            input_name, [10, 30], "qkv.weight", [30], "qkv.bias", output_name
        )

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, -1, 2])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, -1, 2])
        graph_name = "lstm_test"
        graph = helper.make_graph(
            [lstm_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)
        model_inferenced = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(model)
        onnx.save(model_inferenced, output_model_path)

    def dynamic_lstm_quant_test(self, model_fp32_path, per_channel, reduce_range):
        model_int8_qop_path = "lstm_int8.qop.onnx"
        model_int8_qop_path = Path(self._tmp_model_dir.name).joinpath(model_int8_qop_path).as_posix()
        model_int8_qdq_path = "lstm_int8.qdq.onnx"
        model_int8_qdq_path = Path(self._tmp_model_dir.name).joinpath(model_int8_qdq_path).as_posix()
        model_uint8_qdq_path = "lstm_uint8.qdq.onnx"
        model_uint8_qdq_path = Path(self._tmp_model_dir.name).joinpath(model_uint8_qdq_path).as_posix()

        inputarr = np.random.rand(1, 3, 2).astype(np.float32)
        
        # Test LSTM QOperator Dynamic
        quantize_dynamic(
            model_fp32_path,
            model_int8_qop_path,
            quant_format=QuantFormat.QOperator,
            per_channel=per_channel,
            reduce_range=reduce_range,
        )

        quant_nodes = {"DynamicQuantizeLSTM": 1}
        check_op_type_count(self, model_int8_qop_path, **quant_nodes)
        check_model_correctness(
            self,
            model_fp32_path,
            model_int8_qop_path,
            {"input": inputarr},
        )

        # Test LSTM QDQ Dynamic QInt8
        quantize_dynamic(
            model_fp32_path,
            model_int8_qdq_path,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=reduce_range,
        )

        quant_nodes = {"LSTM": 1, "QuantizeLinear":1, "DequantizeLinear":2}
        check_op_type_count(self, model_int8_qdq_path, **quant_nodes)
        check_model_correctness(
            self,
            model_fp32_path,
            model_int8_qdq_path,
            {"input": inputarr},
        )

        # Test LSTM QDQ Dynamic QUInt8 
        quantize_dynamic(
            model_fp32_path,
            model_uint8_qdq_path,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QUInt8,
            activation_type=QuantType.QUInt8,
            per_channel=per_channel,
            reduce_range=reduce_range,
        )

        quant_nodes = {"LSTM": 1, "QuantizeLinear":1, "DequantizeLinear":2}
        check_op_type_count(self, model_uint8_qdq_path, **quant_nodes)
        check_model_correctness(
            self,
            model_fp32_path,
            model_uint8_qdq_path,
            {"input": inputarr},
        )

        model = onnx.load(model_uint8_qdq_path)
        intermediate_layer_value_info = helper.ValueInfoProto()
        intermediate_layer_value_info.name = "input_DequantizeLinear"
        model.graph.output.append(intermediate_layer_value_info)
        intermediate_layer_value_info = helper.ValueInfoProto()
        intermediate_layer_value_info.name = "qkv.weight_DequantizeLinear"
        model.graph.output.append(intermediate_layer_value_info)
        onnx.save(model, model_uint8_qdq_path)

        model = onnx.load(model_int8_qdq_path)
        intermediate_layer_value_info = helper.ValueInfoProto()
        intermediate_layer_value_info.name = "input_DequantizeLinear"
        model.graph.output.append(intermediate_layer_value_info)
        intermediate_layer_value_info = helper.ValueInfoProto()
        intermediate_layer_value_info.name = "qkv.weight_DequantizeLinear"
        model.graph.output.append(intermediate_layer_value_info)
        onnx.save(model, model_int8_qdq_path)

        check_model_correctness(
            self,
            model_uint8_qdq_path,
            model_int8_qdq_path,
            {"input": inputarr},
        )

    def test_quantize_lstm(self):
        np.random.seed(1)
        model_fp32_path = "lstm_fp32.onnx"
        model_fp32_path = Path(self._tmp_model_dir.name).joinpath(model_fp32_path).as_posix()
        self.construct_model_lstm_and_matmul(model_fp32_path)

        self.dynamic_lstm_quant_test(model_fp32_path, True, True)
        self.dynamic_lstm_quant_test(model_fp32_path, True, False)
        self.dynamic_lstm_quant_test(model_fp32_path, False, True)
        self.dynamic_lstm_quant_test(model_fp32_path, False, False)


if __name__ == "__main__":
    unittest.main()
