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
from onnxruntime.quantization.onnx_model import ONNXModel
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
        #    (input)
        #       |
        #      GRU
        #      /  \
        #  Conv(1) \
        #     |     \
        #    Relu  Conv(2)
        #     |     |
        #     \     /
        #       Add
        #        |
        #       (output)
        initializers = []
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 8, 12])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 2, 8, 8])

        # make GRU
        initializers.append(generate_input_initializer([2, 24, 12], np.float32, 'W_GRU'))
        initializers.append(generate_input_initializer([2, 24, 8], np.float32, 'R_GRU'))
        initializers.append(generate_input_initializer([2, 8, 8], np.float32, 'H_GRU'))
        gru_node = onnx.helper.make_node(
            'GRU',
            ['input', 'W_GRU', 'R_GRU', '', '', 'H_GRU'],
            ['GRU_O'],
            hidden_size = 8, 
            direction = 'bidirectional')

        initializers.append(generate_input_initializer([2, 2, 1, 1], np.float32, 'W1'))
        initializers.append(generate_input_initializer([2, 2, 1, 1], np.float32, 'W2'))
        initializers.append(generate_input_initializer([2], np.float32, 'B1'))
        initializers.append(generate_input_initializer([2], np.float32, 'B2'))
        conv_node_1 = onnx.helper.make_node('Conv', ['GRU_O', 'W1', 'B1'], ['Conv1_O'], name='Conv1')
        conv_node_2 = onnx.helper.make_node('Conv', ['GRU_O', 'W2', 'B2'], ['Conv2_O'], name='Conv2')
        relu_node = onnx.helper.make_node('Relu', ['Conv1_O'], ['Relu_O'], name='Relu')
        add_node = onnx.helper.make_node('Add', ['Relu_O', 'Conv2_O'], ['output'], name='Add')
        graph = helper.make_graph([conv_node_1, relu_node, conv_node_2, gru_node, add_node],
                                  'onnx_model_test', [input], [output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, model_path)

    def construct_model_Constant(self, model_path):
        #    (input)    Constant
        #       \         /
        #        \       /
        #         \     /
        #          \   /
        #           Add
        #            |
        #         (output)

        initializers = []
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 8, 12])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 8, 12])

        # make nodes
        constant_node = onnx.helper.make_node('Constant', [], ['const_output'], value_float=42.0)
        add_node = onnx.helper.make_node('Add', ['input', 'const_output'], ['output'], name='Add')
        graph = helper.make_graph([add_node, constant_node],
                                  'onnx_model_test', [input], [output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, model_path)

    def test_topo_sort(self):
        test_model_path = 'onnx_model_topo_sort.onnx'
        self.construct_model(test_model_path)
        onnx_model = ONNXModel(onnx.load(test_model_path))
        check_op_type_order(self, onnx_model.model, ['Conv', 'Relu', 'Conv', 'GRU', 'Add'])
        onnx_model.topological_sort()
        check_op_type_order(self, onnx_model.model, ['GRU', 'Conv', 'Conv', 'Relu', 'Add'])

    def test_topo_sort_constant(self):
        test_model_path = 'onnx_model_topo_sort_constant.onnx'
        self.construct_model_Constant(test_model_path)
        onnx_model = ONNXModel(onnx.load(test_model_path))
        check_op_type_order(self, onnx_model.model, ['Add', 'Constant'])
        onnx_model.topological_sort()
        check_op_type_order(self, onnx_model.model, ['Constant', 'Add'])

if __name__ == '__main__':
    unittest.main()
