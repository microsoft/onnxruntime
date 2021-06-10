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


class TestOpEmbedLayerNormalization(unittest.TestCase):

    def construct_model(self):
        # Inputs:
        # 0: input_ids
        # 1: segment_ids
        # 2: word_embed
        # 3: pos_embed
        # 4: seg_embed
        # 5: layer_norm_weight
        # 6: layer_norm_bias

        #         (segment_ids)    (input_ids)
        #            (Cast)             (Cast)
        #       
        #            (EmbedLayerNormalization) 

        batch = 1  # XXX is this the right value?

        ### Inputs to the EmbedLayerNorm subgraph
        input_ids_shape = [batch, 3]
        input_ids_tensor = helper.make_tensor_value_info('input_ids', TensorProto.INT32, input_ids_shape)

        segment_ids_shape = [batch, 3]
        segment_ids_tensor = helper.make_tensor_value_info('segment_ids', TensorProto.INT32, segment_ids_shape)

        word_embed_shape = [2, 4]
        word_embed_weights = np.random.random_sample(word_embed_shape)
        word_embed_tensor = helper.make_tensor_value_info('word_embed', TensorProto.FLOAT, word_embed_shape)

        pos_embed_shape = [4, 4]
        pos_embed_weights = np.random.random_sample(pos_embed_shape)
        pos_embed_tensor = helper.make_tensor_value_info('pos_embed', TensorProto.FLOAT, pos_embed_shape)

        seg_embed_shape = [2, 4]
        seg_embed_weights = np.random.random_sample(seg_embed_shape)
        seg_embed_tensor = helper.make_tensor_value_info('seg_embed', TensorProto.FLOAT, seg_embed_shape)

        layer_norm_weight_shape = [4]
        layer_norm_weights = np.random.random_sample(layer_norm_weight_shape)
        layer_norm_weight_tensor = helper.make_tensor_value_info('layer_norm_weight', TensorProto.FLOAT, layer_norm_weight_shape)

        layer_norm_bias_shape = [4]
        layer_norm_bias_weights = np.random.random_sample(layer_norm_bias_shape)
        layer_norm_bias_tensor = helper.make_tensor_value_info('layer_norm_bias', TensorProto.FLOAT, layer_norm_bias_shape)

        # TODO - implement these right here.
        layernorm_output_shape = []
        layernorm_output_tensor = {}

        mask_index_output_shape = []
        mask_index_output_tensor = {}

        inputs = ['input_ids', 'segment_ids']


        #
        #
        # TODO(kreeger): LEFT OFF RIGHT HERE. WRITE THE REST OF THIS TEST
        #
        #


    def test_quantize(self):
        print("this is a unit test")
        model = self.construct_model()

if __name__ == '__main__':
    unittest.main()

