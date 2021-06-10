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
from onnxruntime.quantization import quantize_dynamic, QuantFormat
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_op_nodes


class TestOpEmbedLayerNormalization(unittest.TestCase):

    def construct_model(self):
        #         <segment_ids>    <input_ids>
        #                   \        /
        #            (EmbedLayerNormalization)
        #                    /       \
        #      <layernorm_output>  <mask_index_output>

        # TODO(batch needs to be an attribute in the graph somewhere. I don't know where to put it)
        batch = 1  # XXX is this the right value?

        # Inputs to EmbedLayerNormalizationNode
        input_ids_shape = ['batch', 3]
        input_ids_tensor = helper.make_tensor_value_info('input_ids', TensorProto.INT32, input_ids_shape)

        segment_ids_shape = ['batch', 3]
        segment_ids_tensor = helper.make_tensor_value_info('segment_ids', TensorProto.INT32, segment_ids_shape)

        # EmbedLayerNormalization Node Constants and Weights:
        word_embed_shape = [2, 4]
        word_embed_weights = np.random.random_sample(word_embed_shape).astype(dtype='float32')
        word_embed_initializer = onnx.numpy_helper.from_array(word_embed_weights, name='word_embed')

        pos_embed_shape = [4, 4]
        pos_embed_weights = np.random.random_sample(pos_embed_shape).astype(dtype='float32')
        pos_embed_initializer = onnx.numpy_helper.from_array(pos_embed_weights, name='pos_embed')

        seg_embed_shape = [2, 4]
        seg_embed_weights = np.random.random_sample(seg_embed_shape).astype(dtype='float32')
        seg_embed_initializer = onnx.numpy_helper.from_array(seg_embed_weights, name='seg_embed')

        layer_norm_weight_shape = [4]
        layer_norm_weights = np.random.random_sample(layer_norm_weight_shape).astype(dtype='float32')
        layer_norm_weights_initializer = onnx.numpy_helper.from_array(layer_norm_weights, name='layer_norm_weight')

        layer_norm_bias_shape = [4]
        layer_norm_bias_weights = np.random.random_sample(layer_norm_bias_shape).astype(dtype='float32')
        layer_norm_bias_initializer = onnx.numpy_helper.from_array(layer_norm_bias_weights, name='layer_norm_bias')

        # EmbedLayerNormalization Outputs:
        layernorm_out_shape = []
        layernorm_out_tensor = helper.make_tensor_value_info('layernorm_out', TensorProto.FLOAT, layernorm_out_shape)

        mask_index_out_shape = []
        mask_index_out_tensor = helper.make_tensor_value_info('mask_index_out', TensorProto.FLOAT, mask_index_out_shape)

        # EmbedLayerNormalization Node:
        embed_layer_norm_inputs = ['input_ids', 'segment_ids', 'word_embed', 'pos_embed', 'seg_embed', 'layer_norm_weight', 'layer_norm_bias']
        embed_layer_norm_outputs = ['layernorm_out', 'mask_index_out']
        embed_layer_norm_node = helper.make_node('EmbedLayerNormalization', embed_layer_norm_inputs, embed_layer_norm_outputs, domain='com.microsoft')

        # Construct the Graph and Model:
        nodes = [embed_layer_norm_node]
        graph_name = 'embed_layernorm_graph'
        inputs = [input_ids_tensor, segment_ids_tensor]
        outputs = [layernorm_out_tensor, mask_index_out_tensor]
        initializers = [word_embed_initializer, pos_embed_initializer, seg_embed_initializer, layer_norm_weights_initializer, layer_norm_bias_initializer]

        # import pdb; pdb.set_trace()
        graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers)
        model = helper.make_model(graph)

        onnx.save(model, "/mnt/c/Users/nickkreeger/models/test_embed_layer_norm_unit_test.onnx")
        onnx.save(model, "test_embed_layer_norm_unit_test.onnx")


    def test_quantize(self):
        self.construct_model()

        model_f32_path = 'test_embed_layer_norm_unit_test.onnx'
        model_uint8_path = 'test_embed_layer_norm_unit_test_uint8.onnx'

        quantize_dynamic(model_f32_path, model_uint8_path)


if __name__ == '__main__':
    unittest.main()

