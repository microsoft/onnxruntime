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

class TestOpEmbedLayerNormBiasGelu(unittest.TestCase):

    def construct_model(self, batch_size, sequence_length, hidden_size, bias_size):
        input_dims = [batch_size, sequence_length, hidden_size]
        skip_dims = [batch_size, sequence_length, hidden_size]
        gamma_dims = [hidden_size]
        beta_dims = [hidden_size]
        bias_dims = [hidden_size]  # TODO(kreeger): Create a model without bias.

        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_dims)
        skip_tensor = helper.make_tensor_value_info('skip', TensorProto.FLOAT, skip_dims)
        gamma_tensor = helper.make_tensor_value_info('gamma', TensorProto.FLOAT, gamma_dims)
        beta_tensor = helper.make_tensor_value_info('beta', TensorProto.FLOAT, beta_dims)
        bias_tensor = helper.make_tensor_value_info('bias', TensorProto.FLOAT, bias_dims)


    def test_quantize(self):
        pass


if __name__ == '__main__':
    unittest.main()

