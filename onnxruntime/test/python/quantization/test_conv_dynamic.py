#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import onnx
import onnxruntime
import numpy as np
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import quantize_dynamic
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_op_type_order


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
  '''
  Helper function to generate initializers for test inputs
  '''
  tensor = np.random.normal(0, 0.3, tensor_shape).astype(tensor_dtype)
  init = numpy_helper.from_array(tensor, input_name)
  return init

class TestONNXModel(unittest.TestCase):
    def construct_model(self, model_path):
        #       input
        #      /    |
        #     /     |
        #  Conv(1)  |
        #     |     |
        #    Relu  Conv(2)
        #     |     |
        #     \     /
        #       Add
        #        |
        #       (output)
        initializers = []
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 2, 8, 8])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 2, 8, 8])

        initializers.append(generate_input_initializer([2, 2, 1, 1], np.float32, 'W1'))
        initializers.append(generate_input_initializer([2, 2, 1, 1], np.float32, 'W2'))
        initializers.append(generate_input_initializer([2], np.float32, 'B'))
        conv_node_1 = onnx.helper.make_node('Conv', ['input', 'W1', 'B'], ['Conv1_O'], name='Conv1')
        conv_node_2 = onnx.helper.make_node('Conv', ['input', 'W2', 'B'], ['Conv2_O'], name='Conv2')
        relu_node = onnx.helper.make_node('Relu', ['Conv1_O'], ['Relu_O'], name='Relu')
        add_node = onnx.helper.make_node('Add', ['Relu_O', 'Conv2_O'], ['output'], name='Add')
        graph = helper.make_graph([conv_node_1, relu_node, conv_node_2, add_node],
                                  'onnx_model_test', [input], [output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, model_path)

    def dynamic_quant_conv(self, model_fp32_path, model_int8_path):
        quantize_dynamic(model_fp32_path, model_int8_path)
        quant_nodes = {'ConvInteger' : 2}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_path, {'input': np.random.rand(4, 2, 8, 8).astype(np.float32)})

    def test_quant_conv(self):
        np.random.seed(1)
        model_fp32_path = 'conv_bias.fp32.onnx'
        model_int8_path = 'conv_bias.quant.onnx'
        self.construct_model(model_fp32_path)

        self.dynamic_quant_conv(model_fp32_path, model_int8_path)

if __name__ == '__main__':
    unittest.main()
