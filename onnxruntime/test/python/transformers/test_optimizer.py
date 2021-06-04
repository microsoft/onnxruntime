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
import sys

# set path so that we could import from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from onnxruntime.transformers.optimizer import optimize_model, optimize_by_onnxruntime
from onnxruntime.transformers.onnx_model import OnnxModel

BERT_TEST_MODELS = {
    "bert_pytorch_1": ('bert_squad_pytorch1.4_opset11', 'BertForQuestionAnswering_1.onnx'),
    "bert_squad_pytorch1.4_opset10_fp32": ('bert_squad_pytorch1.4_opset10_fp32', 'BertForQuestionAnswering.onnx'),
    "bert_keras_0": ('bert_mrpc_tensorflow2.1_opset10', 'TFBertForSequenceClassification_1.onnx'),
    "bert_keras_squad": ('bert_squad_tensorflow2.1_keras2onnx_opset11', 'TFBertForQuestionAnswering.onnx'),
    "gpt2": ('gpt2_pytorch1.4_opset11_no_past', 'GPT2Model.onnx'),
    "gpt2_past": ('gpt2_pytorch1.5_opset11', 'gpt2_past.onnx'),
    "gpt2_past_mask": ('FUSION', 'gpt2_past_mask_one_layer.onnx'),
    "multiple_embed": ('FUSION', 'embed_layer_norm_multiple.onnx'),
    "bert_tf2onnx_0": ('other_models', 'bert_tf2onnx_0.onnx')
}


def _get_test_model_path(name):
    sub_dir, file = BERT_TEST_MODELS[name]
    if sub_dir == "FUSION":
        return os.path.join('..', '..', '..', '..', 'test', 'testdata', 'transform', 'fusion', file)
    else:
        return os.path.join('test_data', sub_dir, file)


# class TestBertOptimization(unittest.TestCase):
#     def verify_node_count(self, bert_model, expected_node_count, test_name):
#         for op_type, count in expected_node_count.items():
#             if len(bert_model.get_nodes_by_op_type(op_type)) != count:
#                 print(f"Counters is not expected in test: {test_name}")
#                 for op, counter in expected_node_count.items():
#                     print("{}: {} expected={}".format(op, len(bert_model.get_nodes_by_op_type(op)), counter))
#             self.assertEqual(len(bert_model.get_nodes_by_op_type(op_type)), count)

#     # add test function for huggingface pytorch model
#     def _test_optimizer_on_huggingface_model(self,
#                                              model_name,
#                                              expected_fusion_result_list,
#                                              inputs_count=1,
#                                              validate_model=True):
#         # expect fusion result list have the following keys
#         # EmbedLayerNormalization, Attention, Gelu, FastGelu, BiasGelu, LayerNormalization, SkipLayerNormalization
#         model_fusion_statistics = {}
#         from onnx_exporter import export_onnx_model_from_pt
#         from huggingface_models import MODELS
#         from benchmark_helper import Precision

#         input_names = MODELS[model_name][0]

#         import torch
#         with torch.no_grad():
#             _, is_valid_onnx_model, _, _ = export_onnx_model_from_pt(model_name, MODELS[model_name][1],
#                                                                      MODELS[model_name][2], MODELS[model_name][3], None,
#                                                                      './cache_models', './onnx_models',
#                                                                      input_names[:inputs_count], False,
#                                                                      Precision.FLOAT32, True, True, True, True,
#                                                                      model_fusion_statistics)

#         onnx_model = list(model_fusion_statistics.keys())[0]
#         fusion_result_list = list(model_fusion_statistics[onnx_model].values())

#         if validate_model:
#             self.assertEqual(is_valid_onnx_model, True)
#         self.assertEqual(fusion_result_list, expected_fusion_result_list)

#     def _test_optimizer_on_tf_model(self, model_name, expected_fusion_result_list, inputs_count, validate_model=True):
#         # expect fusion result list have the following keys
#         # EmbedLayerNormalization, Attention, Gelu, FastGelu, BiasGelu, LayerNormalization, SkipLayerNormalization
#         model_fusion_statistics = {}
#         from onnx_exporter import export_onnx_model_from_tf
#         from huggingface_models import MODELS
#         from benchmark_helper import Precision
#         print("testing mode ", model_name)
#         print("testing input number = ", inputs_count)
#         input_names = MODELS[model_name][0]

#         import torch
#         with torch.no_grad():
#             _, is_valid_onnx_model, _, _ = export_onnx_model_from_tf(model_name, MODELS[model_name][1],
#                                                                      MODELS[model_name][2], MODELS[model_name][3], None,
#                                                                      './cache_models', './onnx_models',
#                                                                      input_names[:inputs_count], False,
#                                                                      Precision.FLOAT32, True, True, True, True,
#                                                                      model_fusion_statistics)

#         onnx_model = list(model_fusion_statistics.keys())[0]
#         fusion_result_list = list(model_fusion_statistics[onnx_model].values())

#         if validate_model:
#             self.assertEqual(is_valid_onnx_model, True)
#         self.assertEqual(fusion_result_list, expected_fusion_result_list)

#     def test_pytorch_model_1_cpu_onnxruntime(self):
#         input = _get_test_model_path('bert_pytorch_1')
#         output = 'temp.onnx'
#         optimize_by_onnxruntime(input, use_gpu=False, optimized_model_path=output)
#         model = ModelProto()
#         with open(output, "rb") as f:
#             model.ParseFromString(f.read())
#         os.remove(output)
#         bert_model = OnnxModel(model)
#         expected_node_count = {
#             'EmbedLayerNormalization': 1,
#             'Attention': 12,
#             'LayerNormalization': 24,
#             'SkipLayerNormalization': 0,
#             'Gelu': 0,
#             'FastGelu': 0,
#             'BiasGelu': 12
#         }
#         self.verify_node_count(bert_model, expected_node_count, 'test_pytorch_model_1_cpu_onnxruntime')

#     def test_pytorch_model_1_gpu_onnxruntime(self):
#         if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
#             print("skip test_pytorch_model_1_gpu_onnxruntime since no gpu found")
#             return

#         input = _get_test_model_path('bert_pytorch_1')
#         output = 'temp.onnx'
#         optimize_by_onnxruntime(input, use_gpu=True, optimized_model_path=output)
#         model = ModelProto()
#         with open(output, "rb") as f:
#             model.ParseFromString(f.read())
#         os.remove(output)
#         bert_model = OnnxModel(model)
#         expected_node_count = {
#             'EmbedLayerNormalization': 1,
#             'Attention': 12,
#             'LayerNormalization': 24,
#             'SkipLayerNormalization': 0,
#             'Gelu': 0,
#             'FastGelu': 0,
#             'BiasGelu': 12
#         }
#         self.verify_node_count(bert_model, expected_node_count, 'test_pytorch_model_1_gpu_onnxruntime')

#     def test_pytorch_model_2(self):
#         input = _get_test_model_path('bert_squad_pytorch1.4_opset10_fp32')
#         bert_model = optimize_model(input, 'bert', num_heads=2, hidden_size=8)
#         print("fused_operator_statistics for test_pytorch_model_2", bert_model.get_fused_operator_statistics())
#         self.assertTrue(bert_model.is_fully_optimized())

#         # Test change input to int32
#         bert_model.change_input_to_int32()
#         embed_nodes = bert_model.get_nodes_by_op_type('EmbedLayerNormalization')
#         for embed_node in embed_nodes:
#             bert_inputs = embed_node.input[:2] + embed_node.input[7:]
#             for bert_input in bert_inputs:
#                 self.assertIsNotNone(bert_model.find_graph_input(bert_input))
#         for input in bert_model.graph().input:
#             self.assertEqual(input.type.tensor_type.elem_type, TensorProto.INT32)

#     def test_keras_model_1(self):
#         input = _get_test_model_path('bert_keras_0')

#         bert_model = optimize_model(input, 'bert_keras', num_heads=2, hidden_size=8)

#         expected_node_count = {
#             'EmbedLayerNormalization': 1,
#             'Attention': 12,
#             'LayerNormalization': 0,
#             'SkipLayerNormalization': 24,
#             'BiasGelu': 12,
#             'Gelu': 0,
#             'FastGelu': 0
#         }
#         self.verify_node_count(bert_model, expected_node_count, 'test_keras_model_1')

#     def test_keras_squad_model(self):
#         input = _get_test_model_path('bert_keras_squad')

#         bert_model = optimize_model(input, 'bert_keras', num_heads=2, hidden_size=8)

#         print("fused_operator_statistics for test_keras_squad_model", bert_model.get_fused_operator_statistics())

#         self.assertTrue(bert_model.is_fully_optimized())

#     def test_gpt2(self):
#         input = _get_test_model_path('gpt2')
#         model = optimize_model(input, 'gpt2', num_heads=2, hidden_size=4)

#         expected_node_count = {
#             'EmbedLayerNormalization': 0,
#             'Attention': 12,
#             'Gelu': 0,
#             'FastGelu': 12,
#             'BiasGelu': 0,
#             'LayerNormalization': 25,
#             'SkipLayerNormalization': 0
#         }
#         self.verify_node_count(model, expected_node_count, 'test_gpt2')

#     def test_gpt2_past(self):
#         input = _get_test_model_path('gpt2_past')
#         model = optimize_model(input, 'gpt2', num_heads=2, hidden_size=4)

#         expected_node_count = {
#             'EmbedLayerNormalization': 0,
#             'Attention': 12,
#             'Gelu': 0,
#             'FastGelu': 12,
#             'BiasGelu': 0,
#             'LayerNormalization': 25,
#             'SkipLayerNormalization': 0
#         }
#         self.verify_node_count(model, expected_node_count, 'test_gpt2_past')

#     def test_gpt2_past_fp16(self):
#         input_model_path = _get_test_model_path('gpt2_past')
#         model = OnnxModel(load_model(input_model_path, format=None, load_external_data=True))
#         model.convert_model_float32_to_float16(cast_input_output=False)
#         for input in model.graph().input[1:]:
#             self.assertEqual(input.type.tensor_type.elem_type, TensorProto.FLOAT16)
#         for output in model.graph().output:
#             self.assertEqual(output.type.tensor_type.elem_type, TensorProto.FLOAT16)

#     def test_gpt2_past_mask(self):
#         input = _get_test_model_path('gpt2_past_mask')
#         model = optimize_model(input, 'gpt2', num_heads=2, hidden_size=4)
#         expected_node_count = {
#             'EmbedLayerNormalization': 0,
#             'Attention': 1,
#             'Gelu': 0,
#             'FastGelu': 1,
#             'BiasGelu': 0,
#             'LayerNormalization': 2,
#             'SkipLayerNormalization': 0
#         }
#         self.verify_node_count(model, expected_node_count, 'test_gpt2_past_mask')

#     def test_multiple_embed(self):
#         input_model_path = _get_test_model_path('multiple_embed')
#         model = optimize_model(input_model_path, 'bert', num_heads=2, hidden_size=4)
#         expected_node_count = {
#             'EmbedLayerNormalization': 2,
#             'Attention': 2,
#             'Gelu': 0,
#             'FastGelu': 0,
#             'BiasGelu': 0,
#             'LayerNormalization': 0,
#             'SkipLayerNormalization': 0
#         }
#         self.verify_node_count(model, expected_node_count, 'test_multiple_embed')

#     def test_bert_tf2onnx_0(self):
#         input = _get_test_model_path('bert_tf2onnx_0')
#         model = optimize_model(input, 'bert_tf', num_heads=2, hidden_size=8)
#         expected_node_count = {
#             'EmbedLayerNormalization': 0,
#             'Attention': 6,
#             'Gelu': 0,
#             'FastGelu': 6,
#             'BiasGelu': 0,
#             'LayerNormalization': 0,
#             'SkipLayerNormalization': 13
#         }
#         self.verify_node_count(model, expected_node_count, 'test_bert_tf2onnx_0')

#     @pytest.mark.slow
#     def test_huggingface_bert_fusion(self):
#         self._test_optimizer_on_huggingface_model("bert-base-uncased", [1, 12, 0, 0, 12, 0, 24], inputs_count=1)
#         self._test_optimizer_on_huggingface_model("bert-base-uncased", [1, 12, 0, 0, 12, 0, 24], inputs_count=2)
#         self._test_optimizer_on_huggingface_model("bert-base-uncased", [1, 12, 0, 0, 12, 0, 24], inputs_count=3)

#     @pytest.mark.slow
#     def test_huggingface_openaigpt_fusion(self):
#         self._test_optimizer_on_huggingface_model("openai-gpt", [0, 12, 0, 12, 0, 24, 0])

#     @pytest.mark.slow
#     def test_huggingface_gpt2_fusion(self):
#         self._test_optimizer_on_huggingface_model("gpt2", [0, 12, 0, 12, 0, 25, 0])

#     @pytest.mark.slow
#     def test_huggingface_xlm_fusion(self):
#         self._test_optimizer_on_huggingface_model("xlm-mlm-ende-1024", [0, 6, 0, 0, 6, 0, 13])

#     @pytest.mark.slow
#     def test_huggingface_roberta_fusion(self):
#         self._test_optimizer_on_huggingface_model("roberta-base", [0, 12, 0, 0, 12, 1, 24])

#     @pytest.mark.slow
#     def test_huggingface_distillbert_fusion(self):
#         self._test_optimizer_on_huggingface_model("distilbert-base-uncased", [1, 6, 0, 0, 6, 0, 12], inputs_count=1)
#         self._test_optimizer_on_huggingface_model("distilbert-base-uncased", [1, 6, 0, 0, 6, 0, 12], inputs_count=2)

#     @pytest.mark.slow
#     def test_huggingface_camembert_fusion(self):
#         # output not close issue
#         self._test_optimizer_on_huggingface_model("camembert-base", [0, 12, 0, 0, 12, 1, 24], validate_model=False)

#     @pytest.mark.slow
#     def test_huggingface_albert_fusion(self):
#         self._test_optimizer_on_huggingface_model("albert-base-v1", [0, 12, 0, 0, 12, 1, 24])

#     @pytest.mark.slow
#     def test_huggingface_t5_fusion(self):
#         self._test_optimizer_on_huggingface_model("t5-small", [0, 0, 0, 0, 0, 0, 0])

#     @pytest.mark.slow
#     def test_huggingface_xlmroberta_fusion(self):
#         self._test_optimizer_on_huggingface_model("xlm-roberta-base", [0, 12, 0, 0, 12, 1, 24])

#     @pytest.mark.slow
#     def test_huggingface_flaubert_fusion(self):
#         # output not close issue
#         self._test_optimizer_on_huggingface_model("flaubert/flaubert_base_cased", [0, 12, 0, 0, 12, 0, 25],
#                                                   validate_model=False)
#         self._test_optimizer_on_huggingface_model("flaubert/flaubert_small_cased", [0, 6, 0, 0, 6, 12, 1],
#                                                   validate_model=False)

#     @pytest.mark.slow
#     def test_huggingface_dialogpt_fusion(self):
#         self._test_optimizer_on_huggingface_model("microsoft/DialoGPT-small", [0, 12, 0, 12, 0, 25, 0])

#     @pytest.mark.slow
#     def test_huggingface_bart_fusion(self):
#         self._test_optimizer_on_huggingface_model("facebook/bart-base", [0, 0, 0, 0, 12, 2, 30])

      #tensorflow 2.5.0 cause numpy version conficts
#     @pytest.mark.slow
#     def test_huggingface_bert_base_cased_from_tf2onnx(self):
#         self._test_optimizer_on_tf_model("bert-base-cased", [0, 12, 0, 0, 0, 0, 25], 1)
#         self._test_optimizer_on_tf_model("bert-base-cased", [0, 12, 0, 0, 0, 0, 25], 2)
#         self._test_optimizer_on_tf_model("bert-base-cased", [0, 12, 0, 0, 0, 0, 25], 3)

#     @pytest.mark.slow
#     def test_huggingface_distilgpt2_from_tf2onnx(self):
#         self._test_optimizer_on_tf_model("distilgpt2", [0, 0, 0, 0, 0, 12, 1], 1)

#     @pytest.mark.slow
#     def test_huggingface_albert_from_tf2onnx(self):
#         self._test_optimizer_on_tf_model("albert-base-v1", [0, 0, 0, 0, 0, 0, 25], 1)

#     @pytest.mark.slow
#     def test_huggingface_gpt2_from_tf2onnx(self):
#         self._test_optimizer_on_tf_model("gpt2", [0, 0, 0, 0, 0, 24, 1], 1, validate_model=False)

#     @pytest.mark.slow
#     def test_huggingface_roberta_from_tf2onnx(self):
#         self._test_optimizer_on_tf_model("roberta-base", [0, 12, 0, 0, 0, 0, 25], 1, validate_model=False)

#     @pytest.mark.slow
#     def test_huggingface_distilbert_from_tf2onnx(self):
#         self._test_optimizer_on_tf_model("distilbert-base-uncased", [0, 0, 0, 0, 0, 0, 13], 1, validate_model=False)

#     @pytest.mark.slow
#     def test_huggingface_xlm_from_tf2onnx(self):
#         self._test_optimizer_on_tf_model("xlm-mlm-ende-1024", [0, 0, 0, 0, 0, 1, 12], 1, validate_model=False)

#if __name__ == '__main__':
#    unittest.main()
