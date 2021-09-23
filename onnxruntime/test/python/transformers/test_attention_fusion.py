# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import os
import onnx
from bert_model_generator import create_bert_attention, create_tf2onnx_attention_3d
from gpt2_model_generator import create_gpt2_attention
from model_loader import get_test_data_path

from parity_utilities import find_transformers_source
if find_transformers_source():
    from optimizer import optimize_model, optimize_by_fusion
    from onnx_model import OnnxModel
else:
    from onnxruntime.transformers.optimizer import optimize_model, optimize_by_fusion
    from onnxruntime.transformers.onnx_model import OnnxModel


class TestFusion(unittest.TestCase):
    def verify_fusion(self, optimized_model, expected_model_filename):
        optimized_model.topological_sort()

        expected_model_path = os.path.join(os.path.dirname(__file__), 'test_data', 'models', expected_model_filename)
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort()

        self.assertEqual(str(optimized_model.model.graph), str(expected_model.model.graph))

    def test_attention_fusion(self):
        model = create_bert_attention()
        dir = '.'
        model_path = os.path.join(dir, "attention.onnx")
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path)
        os.remove(model_path)

        self.verify_fusion(optimized_model, 'attention_opt.onnx')

    def test_attention_fusion_pruned_model(self):
        model = create_bert_attention(input_hidden_size=16,
                                      num_heads=2,
                                      pruned_qk_hidden_size=8,
                                      pruned_v_hidden_size=8)
        dir = '.'
        model_path = os.path.join(dir, "pruned_attention.onnx")
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path)
        os.remove(model_path)

        self.verify_fusion(optimized_model, 'pruned_attention_opt.onnx')

    def test_attention_fusion_reverse_add_order(self):
        model = create_bert_attention(input_hidden_size=16,
                                      num_heads=2,
                                      pruned_qk_hidden_size=8,
                                      pruned_v_hidden_size=8,
                                      switch_add_inputs=True)
        dir = '.'
        model_path = os.path.join(dir, "bert_attention_reverse_add_order.onnx")
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path)
        os.remove(model_path)

        # reverse add input order will get same optimized model
        self.verify_fusion(optimized_model, 'pruned_attention_opt.onnx')

    def test_attention_fusion_for_varied_qkv_dimensions(self):
        model = create_bert_attention(input_hidden_size=16,
                                      num_heads=2,
                                      pruned_qk_hidden_size=24,
                                      pruned_v_hidden_size=16)
        dir = '.'
        model_path = os.path.join(dir, "attention_with_varied_qkv.onnx")
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path)
        os.remove(model_path)

        self.verify_fusion(optimized_model, 'attention_with_varied_qkv_opt.onnx')

    def test_3d_attention_fusion_tf2onnx_model(self):
        model = create_tf2onnx_attention_3d()
        dir = '.'
        model_path = os.path.join(dir, 'bert_3d_attention.onnx')
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path, model_type='bert_tf', num_heads=4, hidden_size=16)
        os.remove(model_path)

        self.verify_fusion(optimized_model, 'bert_3d_attention_opt.onnx')

    def test_gpt2_attention_fusion(self):
        hidden_size = 64
        num_heads = 4
        for add_order in [False, True]:
            model = create_gpt2_attention(hidden_size=hidden_size, num_heads=num_heads, switch_add_inputs=add_order)
            dir = '.'
            model_path = os.path.join(dir, "gpt2_attention.onnx")
            onnx.save(model, model_path)
            optimized_model = optimize_model(model_path,
                                             model_type='gpt2',
                                             num_heads=num_heads,
                                             hidden_size=hidden_size)
            optimized_model.topological_sort()
            os.remove(model_path)

            model_name = "gpt2_attention_{}.onnx".format("add_opt" if add_order else "opt")
            self.verify_fusion(optimized_model, model_name)

    def test_megatron_gpt2_attention_fusion(self):
        path = get_test_data_path("models", "gpt2_megatron.onnx")
        model = onnx.load(path)
        optimized_model = optimize_by_fusion(model, model_type='gpt2')

        self.verify_fusion(optimized_model, "gpt2_megatron_opt.onnx")


if __name__ == '__main__':
    unittest.main()
