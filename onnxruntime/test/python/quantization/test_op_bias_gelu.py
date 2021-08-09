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
from onnxruntime.quantization import quantize_dynamic
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count


class TestOpBiasGelu(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr


    def construct_model(self, batch, sequence_length, model_path):
        #
        # TODO(kreeger): Cool ASCII art for this.
        #
        input_shape = [batch, sequence_length]
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)

        bias_shape = [sequence_length]
        bias_weights = np.random.random_sample(bias_shape).astype(dtype='float32')
        bias_initializer = onnx.numpy_helper.from_array(bias_weights, name='bias')

        output_shape = input_shape
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        bias_gelu_node = helper.make_node('BiasGelu', ['input', 'bias'], ['output'],
                                          domain='com.microsoft',
                                          name='BiasGelu_1')

        nodes = [bias_gelu_node]
        graph_name = 'bias_gelu_graph'
        inputs = [input_tensor]
        outputs = [output_tensor]
        initializers = [bias_initializer]

        graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers)

        # TODO(kreeger): is |14| the right opset number?
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = 7  # Use stable onnx IR version
        onnx.save(model, model_path)


    def test_stub(self):
        print('Hi from OpBiasGelu test')
        batch = 1
        sequence_length = 10

        model_f32_path = 'test_bias_gelu_batch1.onnx'
        model_uint8_path = 'test_bias_gelu_batch1_uint8.onnx'

        self.construct_model(batch, sequence_length, model_f32_path)

        data_reader = self.input_feeds(1, {'input': [batch, sequence_length]})

        quantize_dynamic(model_f32_path, model_uint8_path)

        # TODO - now check the nodes that end up in the thing.

        check_model_correctness(self, model_f32_path, model_uint8_path, data_reader.get_next())



if __name__ == '__main__':
    unittest.main()
