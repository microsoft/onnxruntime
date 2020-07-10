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
import pytest
from onnx import helper, TensorProto, ModelProto, load_model
from onnx.helper import make_node, make_tensor_value_info
import numpy as np
from onnx import numpy_helper
from optimizer import optimize_model, optimize_by_onnxruntime
from onnx_model import OnnxModel

BERT_TEST_MODELS = {
    "bert_pytorch_0": ('bert_squad_pytorch1.4_opset11', 'BertForQuestionAnswering_0.onnx'),
    "bert_pytorch_1": ('bert_squad_pytorch1.4_opset11', 'BertForQuestionAnswering_1.onnx'),
    "bert_squad_pytorch1.4_opset10_fp32": ('bert_squad_pytorch1.4_opset10_fp32', 'BertForQuestionAnswering.onnx'),
    "bert_keras_0": ('bert_mrpc_tensorflow2.1_opset10', 'TFBertForSequenceClassification_1.onnx'),
    "bert_keras_squad": ('bert_squad_tensorflow2.1_keras2onnx_opset11', 'TFBertForQuestionAnswering.onnx'),
    "gpt2": ('gpt2_pytorch1.4_opset11_no_past', 'GPT2Model.onnx'),
    "gpt2_past": ('gpt2_pytorch1.5_opset11', 'gpt2_past.onnx'),
    "gpt2_past_mask": ('FUSION', 'gpt2_past_mask_one_layer.onnx'),
    "multiple_embed": ('FUSION', 'embed_layer_norm_multiple.onnx'),
}

skip_on_ort_version = pytest.mark.skipif(onnxruntime.__version__ == ('1.3.0'),
                                         reason="skip failed tests. TODO: fix them in 1.4.0.")


def _get_test_model_path(name):
    sub_dir, file = BERT_TEST_MODELS[name]
    if sub_dir == "FUSION":
        return os.path.join('..', '..', '..', 'test', 'testdata', 'transform', 'fusion', file)
    else:
        return os.path.join('test_data', sub_dir, file)


class TestBertOptimization(unittest.TestCase):
    def verify_node_count(self, bert_model, expected_node_count, test_name):
        for op_type, count in expected_node_count.items():
            if len(bert_model.get_nodes_by_op_type(op_type)) != count:
                print(f"Counters is not expected in test: {test_name}")
                for op, counter in expected_node_count.items():
                    print("{}: {} expected={}".format(op, len(bert_model.get_nodes_by_op_type(op)), counter))
            self.assertEqual(len(bert_model.get_nodes_by_op_type(op_type)), count)

    @skip_on_ort_version
    def test_pytorch_model_0_cpu_onnxruntime(self):
        input = _get_test_model_path('bert_pytorch_0')
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

    @skip_on_ort_version
    def test_pytorch_model_0_gpu_onnxruntime(self):
        if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
            print("skip test_pytorch_model_0_gpu_onnxruntime since no gpu found")
            return

        input = _get_test_model_path('bert_pytorch_0')
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

    @skip_on_ort_version
    def test_pytorch_model_1_cpu_onnxruntime(self):
        input = _get_test_model_path('bert_pytorch_1')
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

    @skip_on_ort_version
    def test_pytorch_model_1_gpu_onnxruntime(self):
        if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
            print("skip test_pytorch_model_1_gpu_onnxruntime since no gpu found")
            return

        input = _get_test_model_path('bert_pytorch_1')
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

    @skip_on_ort_version
    def test_pytorch_model_0(self):
        input = _get_test_model_path('bert_pytorch_0')
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
        input = _get_test_model_path('bert_squad_pytorch1.4_opset10_fp32')
        bert_model = optimize_model(input, 'bert', num_heads=2, hidden_size=8)
        print("fused_operator_statistics for test_pytorch_model_2", bert_model.get_fused_operator_statistics())
        self.assertTrue(bert_model.is_fully_optimized())

        # Test change input to int32
        bert_model.change_input_to_int32()
        embed_nodes = bert_model.get_nodes_by_op_type('EmbedLayerNormalization')
        for embed_node in embed_nodes:
            bert_inputs = embed_node.input[:2] + embed_node.input[7:]
            for bert_input in bert_inputs:
                self.assertIsNotNone(bert_model.find_graph_input(bert_input))
        for input in bert_model.graph().input:
            self.assertEqual(input.type.tensor_type.elem_type, TensorProto.INT32)

    def test_keras_model_1(self):
        input = _get_test_model_path('bert_keras_0')

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
        input = _get_test_model_path('bert_keras_squad')

        bert_model = optimize_model(input, 'bert_keras', num_heads=2, hidden_size=8)

        print("fused_operator_statistics for test_keras_squad_model", bert_model.get_fused_operator_statistics())

        self.assertTrue(bert_model.is_fully_optimized())

    def test_gpt2(self):
        input = _get_test_model_path('gpt2')
        model = optimize_model(input, 'gpt2', num_heads=2, hidden_size=4)

        expected_node_count = {
            'EmbedLayerNormalization': 0,
            'Attention': 12,
            'Gelu': 0,
            'FastGelu': 12,
            'BiasGelu': 0,
            'LayerNormalization': 25,
            'SkipLayerNormalization': 0
        }
        self.verify_node_count(model, expected_node_count, 'test_gpt2')

    def test_gpt2_past(self):
        input = _get_test_model_path('gpt2_past')
        model = optimize_model(input, 'gpt2', num_heads=2, hidden_size=4)

        expected_node_count = {
            'EmbedLayerNormalization': 0,
            'Attention': 12,
            'Gelu': 0,
            'FastGelu': 12,
            'BiasGelu': 0,
            'LayerNormalization': 25,
            'SkipLayerNormalization': 0
        }
        self.verify_node_count(model, expected_node_count, 'test_gpt2_past')

    def test_gpt2_past_fp16(self):
        input_model_path = _get_test_model_path('gpt2_past')
        model = OnnxModel(load_model(input_model_path, format=None, load_external_data=True))
        model.convert_model_float32_to_float16(cast_input_output=False)
        for input in model.graph().input[1:]:
            self.assertEqual(input.type.tensor_type.elem_type, TensorProto.FLOAT16)
        for output in model.graph().output:
            self.assertEqual(output.type.tensor_type.elem_type, TensorProto.FLOAT16)

    def test_gpt2_past_mask(self):
        input = _get_test_model_path('gpt2_past_mask')
        model = optimize_model(input, 'gpt2', num_heads=2, hidden_size=4)
        expected_node_count = {
            'EmbedLayerNormalization': 0,
            'Attention': 1,
            'Gelu': 0,
            'FastGelu': 1,
            'BiasGelu': 0,
            'LayerNormalization': 2,
            'SkipLayerNormalization': 0
        }
        self.verify_node_count(model, expected_node_count, 'test_gpt2_past_mask')

    def test_multiple_embed(self):
        input_model_path = _get_test_model_path('multiple_embed')
        model = optimize_model(input_model_path, 'bert', num_heads=2, hidden_size=4)
        expected_node_count = {
            'EmbedLayerNormalization': 2,
            'Attention': 2,
            'Gelu': 0,
            'FastGelu': 0,
            'BiasGelu': 0,
            'LayerNormalization': 0,
            'SkipLayerNormalization': 0
        }
        self.verify_node_count(model, expected_node_count, 'test_multiple_embed')


if __name__ == '__main__':
    unittest.main()
