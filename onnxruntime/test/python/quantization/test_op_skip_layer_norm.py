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


class TestOpSkipLayerNormalization(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.ones(shape).astype(np.float32)})
            input_data_list.extend([inputs])

        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_subgraph_attention_matmul(self, batch_size, hidden_size, sequence_length, model_path):
        nodes = []
        inputs = []
        outputs = []
        initializers = []

        #
        # SkipLayerNorm #1
        #
        sln_1_input_shape = [batch_size, sequence_length, hidden_size]
        sln_1_input_tensor = helper.make_tensor_value_info('sln_1_input', TensorProto.FLOAT, sln_1_input_shape)

        sln_1_skip_shape = [batch_size, sequence_length, hidden_size]
        sln_1_skip_tensor = helper.make_tensor_value_info('sln_1_skip', TensorProto.FLOAT, sln_1_skip_shape)

        sln_1_gamma_shape = [hidden_size]
        sln_1_gamma_weights = np.random.random_sample(sln_1_gamma_shape).astype(dtype='float32')
        sln_1_gamma_initializer = onnx.numpy_helper.from_array(sln_1_gamma_weights, name='sln_1_gamma')

        # TODO(kreeger): Optionally include beta input!
        sln_1_beta_shape = [hidden_size]
        sln_1_beta_weights = np.random.random_sample(sln_1_beta_shape).astype(dtype='float32')
        sln_1_beta_initializer = onnx.numpy_helper.from_array(sln_1_beta_weights, name='sln_1_beta')

        # TODO(kreeger): Optionally include bias input!
        sln_1_bias_shape = [hidden_size]
        sln_1_bias_weights = np.random.random_sample(sln_1_bias_shape).astype(dtype='float32')
        sln_1_bias_initializer = onnx.numpy_helper.from_array(sln_1_bias_weights, name='sln_1_bias')

        sln_1_output_shape = [batch_size, sequence_length, hidden_size]
        sln_1_output_tensor = helper.make_tensor_value_info('sln_1_output', TensorProto.FLOAT, sln_1_output_shape)

        sln_1_inputs = ['sln_1_input', 'sln_1_skip', 'sln_1_gamma', 'sln_1_beta', 'sln_1_bias']
        sln_1_outputs = ['sln_1_output']
        sln_1_node = helper.make_node('SkipLayerNormalization',
                                      sln_1_inputs,
                                      sln_1_outputs,
                                      domain='com.microsoft',
                                      name='SLN_1')

        inputs.append(sln_1_input_tensor)
        inputs.append(sln_1_skip_tensor)

        initializers.append(sln_1_gamma_initializer)
        initializers.append(sln_1_beta_initializer)
        initializers.append(sln_1_bias_initializer)

        nodes.append(sln_1_node)

        #
        # Attention
        #
        attention_weights_shape = [hidden_size, 3 * hidden_size]
        attention_weights = np.random.random_sample(attention_weights_shape).astype(dtype='float32')
        attention_weights_initializer = onnx.numpy_helper.from_array(attention_weights, name='attention_weights')

        attention_bias_shape = [3 * hidden_size]
        attention_bias_weights = np.random.random_sample(attention_bias_shape).astype(dtype='float32')
        attention_bias_initializer = onnx.numpy_helper.from_array(attention_bias_weights, name='attention_bias')

        # Assume simple batch size for mask:
        attention_mask_shape = [batch_size]
        attention_mask_tensor = helper.make_tensor_value_info('attention_mask', TensorProto.INT32, attention_mask_shape)

        attention_output_shape = [batch_size, sequence_length, hidden_size]
        attention_output_tensor = helper.make_tensor_value_info('attention_output', TensorProto.FLOAT,
                                                                attention_output_shape)

        # TODO - need to set the 'num_heads' attribute!

        attention_inputs = ['sln_1_output', 'attention_weights', 'attention_bias', 'attention_mask']
        attention_outputs = ['attention_output']

        attention_node = helper.make_node('Attention',
                                          attention_inputs,
                                          attention_outputs,
                                          domain='com.microsoft',
                                          name='Attention_1')
        attention_node.attribute.extend([helper.make_attribute("num_heads", 4)])

        inputs.append(attention_mask_tensor)

        initializers.append(attention_weights_initializer)
        initializers.append(attention_bias_initializer)

        nodes.append(attention_node)

        #
        # MatMul
        #
        matmul_b_shape = [hidden_size, hidden_size]
        matmul_b_weights = np.random.random_sample(matmul_b_shape).astype(dtype='float32')
        matmul_b_initializer = onnx.numpy_helper.from_array(matmul_b_weights, name='matmul_b')

        matmul_output_shape = [batch_size, sequence_length, hidden_size]
        matmul_output_tensor = helper.make_tensor_value_info('matmul_output', TensorProto.FLOAT, matmul_output_shape)

        matmul_inputs = ['attention_output', 'matmul_b']
        matmul_outputs = ['matmul_output']

        matmul_node = helper.make_node('MatMul', matmul_inputs, matmul_outputs, name='MatMul_1')

        initializers.append(matmul_b_initializer)

        nodes.append(matmul_node)

        #
        # SkipLayerNorm #2
        #
        sln_2_gamma_shape = [hidden_size]
        sln_2_gamma_weights = np.random.random_sample(sln_2_gamma_shape).astype(dtype='float32')
        sln_2_gamma_initializer = onnx.numpy_helper.from_array(sln_2_gamma_weights, name='sln_2_gamma')

        # TODO(kreeger): Optionally include beta input!
        sln_2_beta_shape = [hidden_size]
        sln_2_beta_weights = np.random.random_sample(sln_2_beta_shape).astype(dtype='float32')
        sln_2_beta_initializer = onnx.numpy_helper.from_array(sln_2_beta_weights, name='sln_2_beta')

        # TODO(kreeger): Optionally include bias input!
        sln_2_bias_shape = [hidden_size]
        sln_2_bias_weights = np.random.random_sample(sln_2_bias_shape).astype(dtype='float32')
        sln_2_bias_initializer = onnx.numpy_helper.from_array(sln_2_bias_weights, name='sln_2_bias')

        sln_2_output_shape = [batch_size, sequence_length, hidden_size]
        sln_2_output_tensor = helper.make_tensor_value_info('sln_2_output', TensorProto.FLOAT, sln_2_output_shape)

        sln_2_inputs = ['sln_1_output', 'matmul_output', 'sln_2_gamma', 'sln_2_beta', 'sln_2_bias']
        sln_2_outputs = ['sln_2_output']
        sln_2_node = helper.make_node('SkipLayerNormalization',
                                      sln_2_inputs,
                                      sln_2_outputs,
                                      domain='com.microsoft',
                                      name='SLN_2')

        initializers.append(sln_2_gamma_initializer)
        initializers.append(sln_2_beta_initializer)
        initializers.append(sln_2_bias_initializer)

        outputs.append(sln_2_output_tensor)

        nodes.append(sln_2_node)

        # Finally, construct the graph:
        graph = helper.make_graph(nodes, 'SLN_Attention_MatMul_Subgraph', inputs, outputs, initializer=initializers)

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = 7  # use stable onnx ir version
        onnx.save(model, model_path)

    def construct_subgraph_matmul_biasgelu_matmul(self, batch, hidden_size, sequence_length, model_path):

        # SLN #1 input
        # SLN #1 skip
        # SLN #1 gamma
        # SLN #1 beta
        # SLN #1 bias

        #
        # MatMul #1
        #

        #
        # BiasGelu
        #

        #
        # MatMul #2
        #

        # SLN #2 input
        # SLN #2 skip
        # SLN #2 gamma
        # SLN #2 beta
        # SLN #2 bias

        pass

    def test(self):
        batch_size = 1
        hidden_size = 4
        sequence_length = 8

        model_f32_path = 'test_skip_layer_norm_attention_matmul_batch1.onnx'
        model_uint8_path = 'test_skip_layer_norm_attention_matmul_batch1_uint8.onnx' 

        self.construct_subgraph_attention_matmul(batch_size, hidden_size, sequence_length, model_f32_path)

        data_reader = self.input_feeds(1, {
            'sln_1_input': [batch_size, sequence_length, hidden_size],
            'sln_1_skip': [batch_size, sequence_length, hidden_size]
        })

        quantize_dynamic(model_f32_path, model_uint8_path)


if __name__ == '__main__':
    unittest.main()
