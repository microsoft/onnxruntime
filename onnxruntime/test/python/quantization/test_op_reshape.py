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


class TestOpReshape(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_matmul_reshape(self, output_model_path, input_shape, weight_shape, output_shape):
        #    (input)
        #      |
        #     MatMul
        #      |
        #    Reshape
        #      |
        #    (output)
        input_name = 'input'
        output_name = 'output'
        initializers = []

        # make MatMul node
        weight_name = 'matmul_weight'
        matmul_output_name = 'matmul_output'
        matmul_inputs = [input_name, weight_name]
        matmul_outputs = [matmul_output_name]
        matmul_name = 'matmul_node'
        matmul_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data, name=weight_name))

        matmul_node = onnx.helper.make_node('MatMul', matmul_inputs, matmul_outputs, name=matmul_name)

        # make Reshape node
        reshape_shape = 'reshape_shape'
        reshape_inputs = [matmul_output_name, reshape_shape]
        reshape_output = [output_name]
        reshape_name = 'reshape_node'
        initializers.append(onnx.numpy_helper.from_array(np.array(output_shape, dtype=np.int64), name=reshape_shape))
        reshape_node = onnx.helper.make_node('Reshape', reshape_inputs, reshape_output, name=reshape_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = 'Reshape_Quant_Test'
        graph = helper.make_graph([matmul_node, reshape_node], graph_name,
                                  [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)

    def test_quantize_reshape(self):
        np.random.seed(1)
        model_fp32_path = 'reshape_fp32.onnx'
        model_uint8_path = 'reshape_uint8.onnx'
        model_uint8_qdq_path = 'reshape_uint8_qdq.onnx'

        self.construct_model_matmul_reshape(model_fp32_path,
                                            [3, 7],
                                            [7, 3],
                                            [1, 9])

        # Verify QOperator mode
        data_reader = self.input_feeds(1, {'input': [3, 7]})
        quantize_static(model_fp32_path, model_uint8_path, data_reader)
        # make sure transpose become xint8 operator, its input name could tell that
        check_op_nodes(self, model_uint8_path, lambda node: (node.name != "reshape_node" or node.input[0] != 'matmul_output'))
        qnode_counts = {'QLinearMatMul': 1, 'QuantizeLinear': 1, 'DequantizeLinear': 1, 'Reshape': 1}
        check_op_type_count(self, model_uint8_path, **qnode_counts)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_path, data_reader.get_next())

        # Verify QDQ mode
        data_reader.rewind()
        quantize_static(model_fp32_path, model_uint8_qdq_path, data_reader, quant_format=QuantFormat.QDQ)
        qdqnode_counts = {'MatMul': 1, 'QuantizeLinear': 2, 'DequantizeLinear': 3, 'Reshape': 1}
        check_op_type_count(self, model_uint8_qdq_path, **qdqnode_counts)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_qdq_path, data_reader.get_next())


if __name__ == '__main__':
    unittest.main()
