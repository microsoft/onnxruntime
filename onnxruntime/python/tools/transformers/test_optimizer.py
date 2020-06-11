#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# For live logging, use the command: pytest -o log_cli=true --log-cli-level=DEBUG

import unittest
import os
import onnx
import onnxruntime
from onnx import helper, TensorProto, ModelProto
from onnx.helper import make_node, make_tensor_value_info
import numpy as np
from onnx import numpy_helper
from optimizer import optimize_model, optimize_by_onnxruntime
from OnnxModel import OnnxModel

BERT_TEST_MODELS = {
    "bert_pytorch_0": 'test_data\\bert_squad_pytorch1.4_opset11\\BertForQuestionAnswering_0.onnx',
    "bert_pytorch_1": 'test_data\\bert_squad_pytorch1.4_opset11\\BertForQuestionAnswering_1.onnx',
    "bert_squad_pytorch1.4_opset10_fp32":
    'test_data\\bert_squad_pytorch1.4_opset10_fp32\\BertForQuestionAnswering.onnx',
    "bert_keras_0": 'test_data\\bert_mrpc_tensorflow2.1_opset10\\TFBertForSequenceClassification_1.onnx',
    "bert_keras_squad": 'test_data\\bert_squad_tensorflow2.1_keras2onnx_opset11\\TFBertForQuestionAnswering.onnx',
    "gpt2": 'test_data\\gpt2_pytorch1.4_opset11_no_past\\GPT2Model.onnx'
}


class TestBertOptimization(unittest.TestCase):
    def verify_node_count(self, bert_model, expected_node_count, test_name):
        for op_type, count in expected_node_count.items():
            if len(bert_model.get_nodes_by_op_type(op_type)) != count:
                print(f"Counters is not expected in test: {test_name}")
                for op, counter in expected_node_count.items():
                    print("{}: {} expected={}".format(op, len(bert_model.get_nodes_by_op_type(op)), counter))
            self.assertEqual(len(bert_model.get_nodes_by_op_type(op_type)), count)

    def test_pytorch_model_0_cpu_onnxruntime(self):
        input = BERT_TEST_MODELS['bert_pytorch_0']
        output = 'temp.onnx'
        optimize_by_onnxruntime(input, use_gpu=False, optimized_model_path=output)
        model = ModelProto()
        with open(output, "rb") as f:
            model.ParseFromString(f.read())
        os.remove(output)
        bert_model = OnnxModel(model)
        expected_node_count = {
            'EmbedLayerNormalization': 1,
            'Attention': 12,
            'SkipLayerNormalization': 24,
            'Gelu': 0,
            'FastGelu': 0,
            'BiasGelu': 12
        }
        self.verify_node_count(bert_model, expected_node_count, 'test_pytorch_model_0_cpu_onnxruntime')

    def test_pytorch_model_0_gpu_onnxruntime(self):
        if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
            print("skip test_pytorch_model_0_gpu_onnxruntime since no gpu found")
            return

        input = BERT_TEST_MODELS['bert_pytorch_0']
        output = 'temp.onnx'
        optimize_by_onnxruntime(input, use_gpu=True, optimized_model_path=output)
        model = ModelProto()
        with open(output, "rb") as f:
            model.ParseFromString(f.read())
        os.remove(output)
        bert_model = OnnxModel(model)
        expected_node_count = {
            'EmbedLayerNormalization': 1,
            'Attention': 12,
            'SkipLayerNormalization': 24,
            'Gelu': 0,
            'FastGelu': 12,
            'BiasGelu': 0
        }
        self.verify_node_count(bert_model, expected_node_count, 'test_pytorch_model_0_gpu_onnxruntime')

    def test_pytorch_model_1_cpu_onnxruntime(self):
        input = BERT_TEST_MODELS['bert_pytorch_1']
        output = 'temp.onnx'
        optimize_by_onnxruntime(input, use_gpu=False, optimized_model_path=output)
        model = ModelProto()
        with open(output, "rb") as f:
            model.ParseFromString(f.read())
        os.remove(output)
        bert_model = OnnxModel(model)
        expected_node_count = {
            'EmbedLayerNormalization': 1,
            'Attention': 12,
            'LayerNormalization': 24,
            'SkipLayerNormalization': 0,
            'Gelu': 0,
            'FastGelu': 0,
            'BiasGelu': 12
        }
        self.verify_node_count(bert_model, expected_node_count, 'test_pytorch_model_1_cpu_onnxruntime')

    def test_pytorch_model_1_gpu_onnxruntime(self):
        if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
            print("skip test_pytorch_model_1_gpu_onnxruntime since no gpu found")
            return

        input = BERT_TEST_MODELS['bert_pytorch_1']
        output = 'temp.onnx'
        optimize_by_onnxruntime(input, use_gpu=True, optimized_model_path=output)
        model = ModelProto()
        with open(output, "rb") as f:
            model.ParseFromString(f.read())
        os.remove(output)
        bert_model = OnnxModel(model)
        expected_node_count = {
            'EmbedLayerNormalization': 1,
            'Attention': 12,
            'LayerNormalization': 24,
            'SkipLayerNormalization': 0,
            'Gelu': 0,
            'FastGelu': 12,
            'BiasGelu': 0
        }
        self.verify_node_count(bert_model, expected_node_count, 'test_pytorch_model_1_gpu_onnxruntime')

    def test_pytorch_model_0(self):
        input = BERT_TEST_MODELS['bert_pytorch_0']
        bert_model = optimize_model(input, 'bert', num_heads=2, hidden_size=8)

        expected_node_count = {
            'EmbedLayerNormalization': 1,
            'Attention': 12,
            'SkipLayerNormalization': 24,
            'Gelu': 0,
            'FastGelu': 0,
            'BiasGelu': 12
        }
        self.verify_node_count(bert_model, expected_node_count, 'test_pytorch_model_0')

    def test_pytorch_model_2(self):
        input = BERT_TEST_MODELS['bert_squad_pytorch1.4_opset10_fp32']
        bert_model = optimize_model(input, 'bert', num_heads=2, hidden_size=8)
        print("fused_operator_statistics for test_pytorch_model_2", bert_model.get_fused_operator_statistics())
        self.assertTrue(bert_model.is_fully_optimized())

    def test_keras_model_1(self):
        input = BERT_TEST_MODELS['bert_keras_0']

        bert_model = optimize_model(input, 'bert_keras', num_heads=2, hidden_size=8)

        expected_node_count = {
            'EmbedLayerNormalization': 1,
            'Attention': 12,
            'LayerNormalization': 0,
            'SkipLayerNormalization': 24,
            'BiasGelu': 12,
            'Gelu': 0,
            'FastGelu': 0
        }
        self.verify_node_count(bert_model, expected_node_count, 'test_keras_model_1')

    def test_keras_squad_model(self):
        input = BERT_TEST_MODELS['bert_keras_squad']

        bert_model = optimize_model(input, 'bert_keras', num_heads=2, hidden_size=8)

        print("fused_operator_statistics for test_keras_squad_model", bert_model.get_fused_operator_statistics())

        self.assertTrue(bert_model.is_fully_optimized())

    def test_gpt2(self):
        input = BERT_TEST_MODELS['gpt2']
        bert_model = optimize_model(input, 'gpt2', num_heads=2, hidden_size=4)

        expected_node_count = {
            'EmbedLayerNormalization': 0,
            'Attention': 12,
            'Gelu': 0,
            'FastGelu': 12,
            'BiasGelu': 0,
            'LayerNormalization': 25,
            'SkipLayerNormalization': 0
        }
        self.verify_node_count(bert_model, expected_node_count, 'test_gpt2')


if __name__ == '__main__':
    unittest.main()
