# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import onnx
from parity_utilities import find_transformers_source
from whisper_model_generator import (
    create_whisper_decoder_attention,
    create_whisper_decoder_multihead_attention,
    create_whisper_decoder_with_past_multihead_cross_attention,
    create_whisper_decoder_with_past_multihead_self_attention,
    create_whisper_encoder_attention,
)

if find_transformers_source():
    from fusion_options import FusionOptions
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model


class TestFusion(unittest.TestCase):
    def verify_fusion(self, optimized_model, expected_model_filename):
        optimized_model.topological_sort(is_deterministic=True)

        expected_model_path = os.path.join(
            os.path.dirname(__file__), "test_data", "models", "whisper", expected_model_filename
        )
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort(is_deterministic=True)

        nodes = optimized_model.model.graph.node
        self.assertEqual(len(nodes), len(expected_model.model.graph.node))

        for i in range(len(nodes)):
            self.assertEqual(nodes[i], expected_model.model.graph.node[i])

        for expected_initializer in expected_model.model.graph.initializer:
            self.assertTrue(
                OnnxModel.has_same_value(
                    optimized_model.get_initializer(expected_initializer.name), expected_initializer
                )
            )

    # Attention type #1 in fusion_bart_attention.py
    def test_encoder_attention_fusion_with_skiplayernorm(self):
        num_heads = 4
        hidden_size = 64
        model = create_whisper_encoder_attention(
            num_heads=num_heads, hidden_size=hidden_size, add_before_layernorm=False
        )
        dir = "."
        model_path = os.path.join(dir, "whisper_encoder_attention_sln.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bart")
        optimized_model = optimize_model(
            model_path, model_type="bart", num_heads=num_heads, hidden_size=hidden_size, optimization_options=options
        )
        os.remove(model_path)
        self.verify_fusion(optimized_model, "encoder_attention_with_sln_fused.onnx")

    # Attention type #2 in fusion_bart_attention.py
    def test_decoder_attention_fusion_with_skiplayernorm(self):
        num_heads = 4
        hidden_size = 64
        model = create_whisper_decoder_attention(
            num_heads=num_heads, hidden_size=hidden_size, add_before_layernorm=False
        )
        dir = "."
        model_path = os.path.join(dir, "whisper_decoder_attention_sln.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bart")
        optimized_model = optimize_model(
            model_path, model_type="bart", num_heads=num_heads, hidden_size=hidden_size, optimization_options=options
        )
        os.remove(model_path)
        self.verify_fusion(optimized_model, "decoder_attention_with_sln_fused.onnx")

    # Attention type #4 in fusion_bart_attention.py
    def test_decoder_multihead_attention_fusion(self):
        num_heads = 4
        hidden_size = 64
        model = create_whisper_decoder_multihead_attention(num_heads=num_heads, hidden_size=hidden_size)
        dir = "."
        model_path = os.path.join(dir, "whisper_decoder_mha.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bart")
        options.use_multi_head_attention = True
        optimized_model = optimize_model(
            model_path, model_type="bart", num_heads=num_heads, hidden_size=hidden_size, optimization_options=options
        )
        os.remove(model_path)
        self.verify_fusion(optimized_model, "decoder_mha_fused.onnx")

    # Attention type #3 in fusion_bart_attention.py
    def test_decoder_with_past_multihead_self_attention_fusion_with_skiplayernorm(self):
        num_heads = 4
        hidden_size = 64
        model = create_whisper_decoder_with_past_multihead_self_attention(
            num_heads=num_heads, hidden_size=hidden_size, add_before_layernorm=False
        )
        dir = "."
        model_path = os.path.join(dir, "whisper_decoder_with_past_self_mha.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bart")
        options.use_multi_head_attention = True
        optimized_model = optimize_model(
            model_path, model_type="bart", num_heads=num_heads, hidden_size=hidden_size, optimization_options=options
        )
        os.remove(model_path)
        self.verify_fusion(optimized_model, "decoder_with_past_self_mha_fused.onnx")

    # Attention type #5 in fusion_bart_attention.py
    def test_decoder_with_past_multihead_cross_attention_fusion(self):
        num_heads = 4
        hidden_size = 64
        model = create_whisper_decoder_with_past_multihead_cross_attention(num_heads=num_heads, hidden_size=hidden_size)
        dir = "."
        model_path = os.path.join(dir, "whisper_decoder_with_past_cross_mha.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bart")
        options.use_multi_head_attention = True
        optimized_model = optimize_model(
            model_path, model_type="bart", num_heads=num_heads, hidden_size=hidden_size, optimization_options=options
        )
        os.remove(model_path)
        self.verify_fusion(optimized_model, "decoder_with_past_cross_mha_fused.onnx")

    # Attention type #4 in fusion_bart_attention.py
    def test_decoder_multihead_attention_split_bias_fusion(self):
        num_heads = 4
        hidden_size = 64
        model = create_whisper_decoder_multihead_attention(num_heads=num_heads, hidden_size=hidden_size)
        dir = "."
        model_path = os.path.join(dir, "whisper_decoder_mha.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bart")
        options.use_multi_head_attention = True
        options.disable_multi_head_attention_bias = True
        optimized_model = optimize_model(
            model_path, model_type="bart", num_heads=num_heads, hidden_size=hidden_size, optimization_options=options
        )
        os.remove(model_path)
        self.verify_fusion(optimized_model, "decoder_mha_split_bias_fused.onnx")

    # Attention type #3 in fusion_bart_attention.py
    def test_decoder_with_past_multihead_self_attention_split_bias_fusion_with_skiplayernorm(self):
        num_heads = 4
        hidden_size = 64
        model = create_whisper_decoder_with_past_multihead_self_attention(
            num_heads=num_heads, hidden_size=hidden_size, add_before_layernorm=False
        )
        dir = "."
        model_path = os.path.join(dir, "whisper_decoder_with_past_self_mha.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bart")
        options.use_multi_head_attention = True
        options.disable_multi_head_attention_bias = True

        optimized_model = optimize_model(
            model_path, model_type="bart", num_heads=num_heads, hidden_size=hidden_size, optimization_options=options
        )
        os.remove(model_path)
        self.verify_fusion(optimized_model, "decoder_with_past_self_mha_split_bias_fused.onnx")

    # Attention type #5 in fusion_bart_attention.py
    def test_decoder_with_past_multihead_cross_attention_split_bias_fusion(self):
        num_heads = 4
        hidden_size = 64
        model = create_whisper_decoder_with_past_multihead_cross_attention(num_heads=num_heads, hidden_size=hidden_size)
        dir = "."
        model_path = os.path.join(dir, "whisper_decoder_with_past_cross_mha.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bart")
        options.use_multi_head_attention = True
        options.disable_multi_head_attention_bias = True

        optimized_model = optimize_model(
            model_path, model_type="bart", num_heads=num_heads, hidden_size=hidden_size, optimization_options=options
        )
        os.remove(model_path)
        self.verify_fusion(optimized_model, "decoder_with_past_cross_mha_split_bias_fused.onnx")


if __name__ == "__main__":
    unittest.main()
