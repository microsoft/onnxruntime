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
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantFormat
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count


class TestOpGEMM(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_gemm(self, output_model_path):
        #      (input)
        #         |
        #        GEMM
        #         |
        #        Clip
        #         |
        #        GEMM
        #         |
        #      (output)
        input_name = 'input'
        output_name = 'output'
        initializers = []

        def make_gemm(input_name, weight_shape, weight_name, bias_shape, bias_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            bias_data = np.random.normal(0, 0.1, bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))

            return onnx.helper.make_node('Gemm', [input_name, weight_name, bias_name], [output_name], alpha=1.0, beta=1.0, transB = 1)
        # make gemm1 node
        gemm1_output_name = "gemm1_output"
        gemm1_node = make_gemm(input_name, [100, 10], 'linear1.weight', [100], 'linear1.bias', gemm1_output_name)

        #make Clip
        clip_min_name = 'clip_min'
        clip_max_name = 'clip_max'
        clip_output_name = 'clip_output'
        clip_inputs = [gemm1_output_name, clip_min_name, clip_max_name]
        clip_outputs = [clip_output_name]
        initializers.append(onnx.numpy_helper.from_array(np.array(-1.0, dtype=np.float32), name=clip_min_name))
        initializers.append(onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=clip_max_name))
        clip_node = onnx.helper.make_node('Clip', clip_inputs, clip_outputs)

        # make gemm2 node
        gemm2_node = make_gemm(clip_output_name, [10, 100], 'linear2.weight', [10], 'linear2.bias', output_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [-1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [-1, 10])
        graph_name = 'gemm_test'
        graph = helper.make_graph([gemm1_node, clip_node, gemm2_node], graph_name,
                                  [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)

    def construct_model_attention_and_matmul(self, output_model_path):
        #      (input)
        #         |
        #     Attention
        #         |
        #       MatMul
        #         |
        #      (output)
        input_name = 'input'
        output_name = 'output'
        initializers = []

        def make_attention_node(input_name, weight_shape, weight_name, bias_shape, bias_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            bias_data = np.random.normal(0, 0.1, bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))

            return onnx.helper.make_node('Attention', [input_name, weight_name, bias_name], [output_name])

        def make_matmul_node(input_name, weight_shape, weight_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            return onnx.helper.make_node('MatMul', [input_name, weight_name], [output_name])
        # make attention node
        attention_output_name = "attention_output"
        attention_node = make_attention_node(input_name, [10, 30], 'qkv.weight', [30], 'qkv.bias', attention_output_name)
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", 5)])

        # make matmul node
        matmul_node = make_matmul_node(attention_output_name, [10, 10], 'matmul.weight', output_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, -1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, -1, 10])
        graph_name = 'attention_test'
        graph = helper.make_graph([attention_node, matmul_node], graph_name,
                                  [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)

    def static_quant_test(self, model_fp32_path, model_int8_path):
        data_reader = self.input_feeds(1, {'input': [5, 10]})
        quantize_static(model_fp32_path,
                        model_int8_path,
                        data_reader)
        data_reader.rewind()
        quant_nodes = {'QLinearMatMul' : 2,
                       'QLinearAdd' : 2,
                       'QuantizeLinear' : 1,
                       'DequantizeLinear' : 1}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next())

    def static_quant_test_qdq(self, model_fp32_path, model_int8_path):
        data_reader = self.input_feeds(1, {'input': [5, 10]})
        quantize_static(model_fp32_path,
                        model_int8_path,
                        data_reader,
                        quant_format=QuantFormat.QDQ)
        data_reader.rewind()
        quant_nodes = {'MatMul' : 2,
                       'Add' : 2,
                       'QuantizeLinear' : 4,
                       'DequantizeLinear' : 8}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next())


    def dynamic_quant_test(self, model_fp32_path, model_int8_path):
        quantize_dynamic(model_fp32_path, model_int8_path)
        quant_nodes = {'MatMulInteger' : 2}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_path, {'input': np.random.rand(5,10).astype(np.float32)})

    def dynamic_attention_quant_test(self, model_fp32_path, model_int8_path, per_channel, reduce_range):
        quantize_dynamic(model_fp32_path, model_int8_path, per_channel=per_channel, reduce_range=reduce_range)
        quant_nodes = {'QAttention' : 1, 'MatMulInteger' : 1}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        check_model_correctness(self, model_fp32_path, model_int8_path, {'input': np.random.rand(1,5,10).astype(np.float32)})

    def test_quantize_reshape(self):
        np.random.seed(1)
        model_fp32_path = 'gemm_fp32.onnx'
        model_int8_path = 'gemm_fp32.quant.onnx'
        self.construct_model_gemm(model_fp32_path)

        self.static_quant_test(model_fp32_path, model_int8_path)
        self.static_quant_test_qdq(model_fp32_path, model_int8_path)
        self.dynamic_quant_test(model_fp32_path, model_int8_path)

    def test_quantize_attention(self):
        np.random.seed(1)
        model_fp32_path = 'attention_fp32.onnx'
        model_int8_path = 'attention_fp32.quant.onnx'
        self.construct_model_attention_and_matmul(model_fp32_path)

        self.dynamic_attention_quant_test(model_fp32_path, model_int8_path, True, True)
        self.dynamic_attention_quant_test(model_fp32_path, model_int8_path, True, False)
        self.dynamic_attention_quant_test(model_fp32_path, model_int8_path, False, True)
        self.dynamic_attention_quant_test(model_fp32_path, model_int8_path, False, False)

if __name__ == '__main__':
    unittest.main()
