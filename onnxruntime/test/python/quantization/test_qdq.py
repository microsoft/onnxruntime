#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import onnx
import numpy as np
from onnx import helper, TensorProto
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_op_type_order

class TestQDQFormat(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

class TestQDQFormatConv(TestQDQFormat):
    def construct_model_conv(self, output_model_path, input_shape, weight_shape, output_shape, has_bias):
        #    (input)
        #      |
        #     Conv
        #      |
        #    (output)
        input_name = 'input'
        output_name = 'output'
        initializers = []

        # make Conv node
        weight_name = 'conv_weight'
        bias_name = 'conv_bias'
        conv_inputs = [input_name, weight_name]
        conv_outputs = [output_name]
        conv_name = 'conv_node'
        conv_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(conv_weight_data, name=weight_name))
        if has_bias:
            conv_inputs.append(bias_name)
            bias_data = np.random.normal(0, 0.05, (weight_shape[0])).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))
        conv_node = onnx.helper.make_node('Conv', conv_inputs, conv_outputs, name=conv_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = 'QDQ_Test_Conv'
        graph = helper.make_graph([conv_node], graph_name,
                                  [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)

    def verify_quantize_conv(self, has_bias, per_channel):
        np.random.seed(1)
        model_fp32_path = 'conv_fp32.{}.{}.onnx'.format(has_bias, per_channel)
        model_int8_qdq_path = 'conv_quant_qdq.{}.{}.onnx'.format(has_bias, per_channel)
        model_int8_qop_path = 'conv_quant_qop.{}.{}.onnx'.format(has_bias, per_channel)
        data_reader = self.input_feeds(1, {'input': [1, 8, 33, 33]})
        self.construct_model_conv(model_fp32_path,
                                  [1, 8, 33, 33],
                                  [16, 8, 3, 3],
                                  [1, 16, 31, 31],
                                  has_bias)
        quantize_static(model_fp32_path,
                        model_int8_qdq_path,
                        data_reader,
                        quant_format=QuantFormat.QDQ,
                        per_channel = per_channel,
                        reduce_range = per_channel
                        )
        data_reader.rewind()
        qdq_nodes = {'Conv': 1, 'QuantizeLinear': 1, 'DequantizeLinear': 3 if has_bias else 2}
        check_op_type_count(self, model_int8_qdq_path, **qdq_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_qdq_path, data_reader.get_next())

        data_reader.rewind()
        quantize_static(model_fp32_path,
                        model_int8_qop_path,
                        data_reader,
                        quant_format=QuantFormat.QOperator,
                        per_channel = per_channel,
                        reduce_range = per_channel
                        )
        data_reader.rewind()
        qop_nodes = {'QLinearConv': 1, 'QuantizeLinear': 1, 'DequantizeLinear': 1}
        check_op_type_count(self, model_int8_qop_path, **qop_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_qop_path, data_reader.get_next())

    def test_quantize_conv_without_bias(self):
        self.verify_quantize_conv(False, False) # has_bias:False, per_channel:False
        self.verify_quantize_conv(False, True) # has_bias:False, per_channel:True
        self.verify_quantize_conv(True, False) # has_bias:True, per_channel:False
        self.verify_quantize_conv(True, True) # has_bias:True, per_channel:True

class TestQDQFormatConvClip(TestQDQFormat):
    def construct_model_conv_clip(self, output_model_path, input_shape, weight_shape, output_shape):
        #    (input)
        #      |
        #     Conv
        #      |
        #     Clip
        #      |
        #    (output)
        input_name = 'input'
        output_name = 'output'
        initializers = []

        # make Conv node
        weight_name = 'conv_weight'
        conv_inputs = [input_name, weight_name]
        conv_outputs = ['conv_output']
        conv_name = 'conv_node'
        conv_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(conv_weight_data, name=weight_name))
        conv_node = onnx.helper.make_node('Conv', conv_inputs, conv_outputs, name=conv_name)

        # make Clip node
        clip_min_name = 'clip_min'
        clip_max_name = 'clip_max'
        clip_inputs = [conv_outputs[0], clip_min_name, clip_max_name]
        clip_outputs = [output_name]
        clip_name = 'clip_node'
        initializers.append(onnx.numpy_helper.from_array(np.array(-1.0, dtype=np.float32), name=clip_min_name))
        initializers.append(onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=clip_max_name))
        clip_node = onnx.helper.make_node('Clip', clip_inputs, clip_outputs, name=clip_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = 'QDQ_Test_Conv_clip'
        graph = helper.make_graph([conv_node, clip_node], graph_name,
                                  [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)

    def verify(self, per_channel):
        np.random.seed(1)
        model_fp32_path = 'conv_clip_fp32.{}.onnx'.format(per_channel)
        model_int8_qdq_path = 'conv_clip_quant_qdq.{}.onnx'.format(per_channel)
        model_int8_qop_path = 'conv_clip_quant_qop.{}.onnx'.format(per_channel)
        data_reader = self.input_feeds(1, {'input': [1, 8, 33, 33]})
        self.construct_model_conv_clip(model_fp32_path,
                                       [1, 8, 33, 33],
                                       [16, 8, 3, 3],
                                       [1, 16, 31, 31])
        quantize_static(model_fp32_path,
                        model_int8_qdq_path,
                        data_reader,
                        quant_format=QuantFormat.QDQ,
                        per_channel = per_channel,
                        reduce_range = per_channel
                        )
        data_reader.rewind()
        #topo sort check
        check_op_type_order(self, model_int8_qdq_path, ['DequantizeLinear', 'QuantizeLinear', 'DequantizeLinear', 'Conv', 'Clip'])
        check_model_correctness(self, model_fp32_path, model_int8_qdq_path, data_reader.get_next())

        data_reader.rewind()
        quantize_static(model_fp32_path,
                        model_int8_qop_path,
                        data_reader,
                        quant_format=QuantFormat.QOperator,
                        per_channel = per_channel,
                        reduce_range = per_channel
                        )
        data_reader.rewind()
        qop_nodes = {'QLinearConv': 1, 'QuantizeLinear': 1, 'DequantizeLinear': 1}
        check_op_type_count(self, model_int8_qop_path, **qop_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_qop_path, data_reader.get_next())

    def test_quantize_conv_without_bias(self):
        self.verify(False) # per_channel:False
        self.verify(True) # per_channel:True

if __name__ == '__main__':
    unittest.main()
