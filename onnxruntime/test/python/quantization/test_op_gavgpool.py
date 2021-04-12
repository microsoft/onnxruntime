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
from onnxruntime.quantization import quantize_static, quantize_dynamic
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count


class TestOpGlobalAveragePool(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_gavgpool(self, output_model_path, input_shape, weight_shape, output_shape):
        #      (input)
        #         |
        #  GlobalAveragePool
        #         |
        #       Expand
        #         |
        #        Conv
        #         |
        #  GlobalAveragePool
        #         |
        #      (output)
        input_name = 'input'
        expand_input = 'expand_input'
        conv_input = 'conv_input'
        gavgpool_input_2nd = 'gavgpool_input'
        output_name = 'output'
        initializers = []

        #make 1st GlobalAveragePool node
        gavgpool_node_1 = onnx.helper.make_node('GlobalAveragePool', [input_name], [expand_input])

        #make Expand node
        expand_shape_name = 'expand_shape'
        initializers.append(onnx.numpy_helper.from_array(np.array(input_shape, dtype=np.int64), name=expand_shape_name))
        expand_node = onnx.helper.make_node('Expand', [expand_input, expand_shape_name], [conv_input])

        # make Conv node
        weight_name = 'conv_weight'
        conv_name = 'conv_node'
        conv_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(conv_weight_data, name=weight_name))
        conv_node = onnx.helper.make_node('Conv', [conv_input, weight_name], [gavgpool_input_2nd], name=conv_name)

        #make 1st GlobalAveragePool node
        gavgpool_node_2 = onnx.helper.make_node('GlobalAveragePool', [gavgpool_input_2nd], [output_name])

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = 'GAveragePool_test'
        graph = helper.make_graph([gavgpool_node_1, expand_node, conv_node, gavgpool_node_2], graph_name,
                                  [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)

    def test_quantize_reshape(self):
        np.random.seed(1)
        model_fp32_path = 'gavg_pool_fp32.onnx'
        model_int8_path = 'gavg_pool_fp32.quant.onnx'
        data_reader = self.input_feeds(1, {'input': [1, 8, 33, 33]})
        self.construct_model_gavgpool(model_fp32_path,
                                      [1, 8, 33, 33],
                                      [16, 8, 3, 3],
                                      [1, 16, 1, 1])
        quantize_static(model_fp32_path,
                        model_int8_path,
                        data_reader)
        data_reader.rewind()
        quant_nodes = {'QLinearConv' : 1,
                       'GlobalAveragePool' : 1,
                       'QLinearGlobalAveragePool' : 1,
                       'QuantizeLinear' : 1,
                       'DequantizeLinear' : 1}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next())


if __name__ == '__main__':
    unittest.main()
