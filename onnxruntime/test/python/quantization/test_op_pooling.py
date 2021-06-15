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
from onnxruntime.quantization import quantize_static, QuantFormat
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_op_nodes


class TestOpAveragePool(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_conv_avgpool(self, output_model_path,
                                     conv_input_shape, conv_weight_shape,
                                     avgpool_input_shape, avgpool_attributes,
                                     output_shape,
                                     ):
        #      (input)
        #          \
        #         Conv
        #        /    \
        #   Identity  AveragePool
        #    /            \
        # (identity_out)  (output)
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, conv_input_shape)

        conv_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name='conv1_weight')
        conv_node = onnx.helper.make_node('Conv', ['input', 'conv1_weight'], ['conv_output'], name='conv_node')

        identity_out = helper.make_tensor_value_info('identity_out', TensorProto.FLOAT, avgpool_input_shape)
        identity_node = helper.make_node('Identity', ['conv_output'], ['identity_out'], name='IdentityNode')

        initializers = [conv_weight_initializer]

        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        avgpool_node = helper.make_node('AveragePool', ['conv_output'], ['output'], name='avgpool_node', **avgpool_attributes)

        graph = helper.make_graph([conv_node, identity_node, avgpool_node], 'TestOpQuantizerAveragePool_test_model',
                                  [input_tensor], [identity_out, output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
        model.ir_version = 7 # use stable onnx ir version
        onnx.save(model, output_model_path)

    def test_quantize_avgpool(self):
        np.random.seed(1)

        model_fp32_path = 'avgpool_fp32.onnx'
        model_uint8_path = 'avgpool_uint8.onnx'
        model_uint8_qdq_path = 'avgpool_uint8_qdq.onnx'

        self.construct_model_conv_avgpool(model_fp32_path,
                                          [1, 2, 26, 42], [3, 2, 3, 3],
                                          [1, 3, 24, 40], {'kernel_shape': [3, 3]},
                                          [1, 3, 22, 38])

        # Verify QOperator mode
        data_reader = self.input_feeds(1, {'input': [1, 2, 26, 42]})
        quantize_static(model_fp32_path, model_uint8_path, data_reader)
        qnode_counts = {'QLinearConv': 1, 'QuantizeLinear': 1, 'DequantizeLinear': 2, 'QLinearAveragePool': 1}
        check_op_type_count(self, model_uint8_path, **qnode_counts)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_path, data_reader.get_next())

        # Verify QDQ mode
        data_reader.rewind()
        quantize_static(model_fp32_path, model_uint8_qdq_path, data_reader, quant_format=QuantFormat.QDQ)
        qdqnode_counts = {'Conv': 1, 'QuantizeLinear': 2, 'DequantizeLinear': 3, 'AveragePool': 1}
        check_op_type_count(self, model_uint8_qdq_path, **qdqnode_counts)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_qdq_path, data_reader.get_next())


if __name__ == '__main__':
    unittest.main()
