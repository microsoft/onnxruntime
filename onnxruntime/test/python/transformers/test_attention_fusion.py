# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import os
import sys
import onnx
from bert_model_generator import create_bert_attention, create_tf2onnx_attention_3d
from gpt2_model_generator import create_gpt2_attention

# set path so that we could import from parent directory
transformers_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python', 'tools', 'transformers')
if os.path.exists(transformers_dir):
    sys.path.append(transformers_dir)
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.optimizer import optimize_model


class TestFusion(unittest.TestCase):
    def test_attention_fusion(self):
        model = create_bert_attention()
        dir = '.'
        model_path = os.path.join(dir, "attention.onnx")
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path)
        os.remove(model_path)

        expected_model_path = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'attention_opt.onnx')
        expected = onnx.load(expected_model_path)
        self.assertEqual(str(optimized_model.model.graph), str(expected.graph))

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

        expected_model_path = os.path.join(os.path.dirname(__file__), 'test_data', 'models',
                                           'pruned_attention_opt.onnx')
        expected = onnx.load(expected_model_path)
        self.assertEqual(str(optimized_model.model.graph), str(expected.graph))

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
        expected_model_path = os.path.join(os.path.dirname(__file__), 'test_data', 'models',
                                           'pruned_attention_opt.onnx')
        expected = onnx.load(expected_model_path)
        self.assertEqual(str(optimized_model.model.graph), str(expected.graph))

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

        expected_model_path = os.path.join(os.path.dirname(__file__), 'test_data', 'models',
                                           'attention_with_varied_qkv_opt.onnx')
        expected = onnx.load(expected_model_path)
        self.assertEqual(str(optimized_model.model.graph), str(expected.graph))

    def test_3d_attention_fusion_tf2onnx_model(self):
        model = create_tf2onnx_attention_3d()
        dir = '.'
        model_path = os.path.join(dir, 'bert_3d_attention.onnx')
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path, model_type='bert_tf', num_heads=4, hidden_size=16)
        os.remove(model_path)

        expected_model_path = os.path.join(os.path.dirname(__file__), 'test_data', 'models',
                                           'bert_3d_attention_opt.onnx')
        expected = onnx.load(expected_model_path)
        self.assertEqual(str(optimized_model.model.graph), str(expected.graph))

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
                                             hidden_size=hidden_size,
                                             disable_onnxruntime=True)
            os.remove(model_path)

            model_name = "gpt2_attention_{}.onnx".format("add_opt" if add_order else "opt")
            expected_model_path = os.path.join(os.path.dirname(__file__), 'test_data', 'models', model_name)
            expected = onnx.load(expected_model_path)
            self.assertEqual(str(optimized_model.model.graph), str(expected.graph))


if __name__ == '__main__':
    unittest.main()
