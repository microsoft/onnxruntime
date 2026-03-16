# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import tempfile
import unittest

import numpy as np
import onnx
from bart_model_generator import create_bart_attention_sdpa
from bert_model_generator import create_bert_attention, create_bert_attention_pre_ln, create_tf2onnx_attention_3d
from gpt2_model_generator import create_gpt2_attention, create_gpt2_attention_no_past
from model_loader import get_test_data_path
from onnx import numpy_helper
from parity_utilities import find_transformers_source
from qwen3_model_generator import create_qwen3_decoder_layer

if find_transformers_source():
    from fusion_options import FusionOptions
    from onnx_model import OnnxModel
    from optimizer import optimize_by_fusion, optimize_model
else:
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_by_fusion, optimize_model


class TestFusion(unittest.TestCase):
    def verify_fusion(self, optimized_model, expected_model_filename):
        optimized_model.topological_sort(is_deterministic=True)

        expected_model_path = os.path.join(os.path.dirname(__file__), "test_data", "models", expected_model_filename)
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

    def test_multi_head_attention_fusion(self):
        model = create_bert_attention()
        dir = "."
        model_path = os.path.join(dir, "attention.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bert")
        options.use_multi_head_attention = True
        options.use_raw_attention_mask(True)
        optimized_model = optimize_model(model_path, optimization_options=options)
        os.remove(model_path)
        self.verify_fusion(optimized_model, "attention_mha.onnx")

    def test_attention_fusion(self):
        model = create_bert_attention()
        dir = "."
        model_path = os.path.join(dir, "attention.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bert")
        options.use_raw_attention_mask(True)
        optimized_model = optimize_model(model_path, optimization_options=options)
        os.remove(model_path)

        self.verify_fusion(optimized_model, "attention_opt.onnx")

    def test_attention_fusion_pruned_model(self):
        model = create_bert_attention(
            input_hidden_size=16,
            num_heads=2,
            pruned_qk_hidden_size=8,
            pruned_v_hidden_size=8,
        )
        dir = "."
        model_path = os.path.join(dir, "pruned_attention.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bert")
        options.use_raw_attention_mask(True)
        optimized_model = optimize_model(model_path, optimization_options=options)
        os.remove(model_path)

        self.verify_fusion(optimized_model, "pruned_attention_opt.onnx")

    def test_attention_fusion_reverse_add_order(self):
        model = create_bert_attention(
            input_hidden_size=16,
            num_heads=2,
            pruned_qk_hidden_size=8,
            pruned_v_hidden_size=8,
            switch_add_inputs=True,
        )
        dir = "."
        model_path = os.path.join(dir, "bert_attention_reverse_add_order.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bert")
        options.use_raw_attention_mask(True)
        optimized_model = optimize_model(model_path, optimization_options=options)
        os.remove(model_path)

        # reverse add input order will get same optimized model
        self.verify_fusion(optimized_model, "pruned_attention_opt.onnx")

    def test_attention_fusion_for_varied_qkv_dimensions(self):
        model = create_bert_attention(
            input_hidden_size=16,
            num_heads=2,
            pruned_qk_hidden_size=24,
            pruned_v_hidden_size=16,
        )
        dir = "."
        model_path = os.path.join(dir, "attention_with_varied_qkv.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bert")
        options.use_raw_attention_mask(True)
        optimized_model = optimize_model(model_path, optimization_options=options)
        os.remove(model_path)

        self.verify_fusion(optimized_model, "attention_with_varied_qkv_opt.onnx")

    def test_attention_fusion_for_varied_qkv_dimensions_with_wrong_opt_parameters(self):
        model = create_bert_attention(
            input_hidden_size=16,
            num_heads=2,
            pruned_qk_hidden_size=24,
            pruned_v_hidden_size=16,
        )
        dir = "."
        model_path = os.path.join(dir, "attention_with_varied_qkv.onnx")
        onnx.save(model, model_path)

        # wrong num_heads and hidden_size
        options = FusionOptions("bert")
        options.use_raw_attention_mask(True)
        optimized_model = optimize_model(model_path, "bert", num_heads=8, hidden_size=8, optimization_options=options)

        os.remove(model_path)

        self.verify_fusion(optimized_model, "attention_with_varied_qkv_opt.onnx")

    def test_3d_attention_fusion_tf2onnx_model(self):
        model = create_tf2onnx_attention_3d()
        dir = "."
        model_path = os.path.join(dir, "bert_3d_attention.onnx")
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path, model_type="bert_tf", num_heads=4, hidden_size=16)
        os.remove(model_path)

        self.verify_fusion(optimized_model, "bert_3d_attention_opt.onnx")

    def test_attention_fusion_pre_ln(self):
        """Test attention fusion for pre-layer-norm first block.

        In a pre-LN model the first block has no Add before the first
        LayerNormalization — the graph input feeds LN directly.
        """
        model = create_bert_attention_pre_ln()
        dir = tempfile.mkdtemp()
        model_path = os.path.join(dir, "pre_ln_attention.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bert")
        options.use_raw_attention_mask(True)
        optimized_model = optimize_model(model_path, opt_level=0, optimization_options=options)
        os.remove(model_path)

        attention_nodes = [n for n in optimized_model.model.graph.node if n.op_type == "Attention"]
        self.assertEqual(len(attention_nodes), 1, "Expected exactly 1 fused Attention node")
        num_heads_attr = next((a for a in attention_nodes[0].attribute if a.name == "num_heads"), None)
        self.assertIsNotNone(num_heads_attr)
        self.assertEqual(num_heads_attr.i, 2)

    def test_attention_fusion_pre_ln_reverse_add_order(self):
        """Pre-LN fusion with reversed Add input ordering."""
        model = create_bert_attention_pre_ln(switch_add_inputs=True)
        dir = tempfile.mkdtemp()
        model_path = os.path.join(dir, "pre_ln_attention_reverse.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bert")
        options.use_raw_attention_mask(True)
        optimized_model = optimize_model(model_path, opt_level=0, optimization_options=options)
        os.remove(model_path)

        attention_nodes = [n for n in optimized_model.model.graph.node if n.op_type == "Attention"]
        self.assertEqual(len(attention_nodes), 1, "Expected exactly 1 fused Attention node")
        num_heads_attr = next((a for a in attention_nodes[0].attribute if a.name == "num_heads"), None)
        self.assertIsNotNone(num_heads_attr)
        self.assertEqual(num_heads_attr.i, 2)

    def test_attention_fusion_pre_ln_with_skiplayernorm(self):
        """Pre-LN fusion when SkipLayerNorm fusion runs first (exercises Change 3).

        The optimizer runs fuse_skip_layer_norm before fuse_attention.  When enabled,
        the Add + LayerNorm after the residual becomes a SkipLayerNormalization node,
        and attention fusion must handle that anchor type.
        """
        model = create_bert_attention_pre_ln()
        dir = tempfile.mkdtemp()
        model_path = os.path.join(dir, "pre_ln_attention_skiplayernorm.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("bert")
        options.use_raw_attention_mask(True)
        options.enable_skip_layer_norm = True
        optimized_model = optimize_model(model_path, opt_level=0, optimization_options=options)
        os.remove(model_path)

        attention_nodes = [n for n in optimized_model.model.graph.node if n.op_type == "Attention"]
        self.assertEqual(len(attention_nodes), 1, "Expected exactly 1 fused Attention node with SkipLN anchor")
        num_heads_attr = next((a for a in attention_nodes[0].attribute if a.name == "num_heads"), None)
        self.assertIsNotNone(num_heads_attr)
        self.assertEqual(num_heads_attr.i, 2)

    def test_gpt2_attention_fusion(self):
        hidden_size = 64
        num_heads = 4
        for add_order in [False, True]:
            for enable_skip_layer_norm_fusion in [False, True]:
                model = create_gpt2_attention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    switch_add_inputs=add_order,
                )
                dir = "."
                model_path = os.path.join(dir, "gpt2_attention.onnx")
                onnx.save(model, model_path)

                options = FusionOptions("gpt2")
                options.enable_skip_layer_norm = enable_skip_layer_norm_fusion

                optimized_model = optimize_model(
                    model_path,
                    model_type="gpt2",
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    optimization_options=options,
                )

                optimized_model.topological_sort()
                os.remove(model_path)

                model_suffix = ""
                if add_order and enable_skip_layer_norm_fusion:
                    model_suffix = "add_opt_skiplayernorm"
                elif add_order and not enable_skip_layer_norm_fusion:
                    model_suffix = "add_opt_no_skiplayernorm"
                elif not add_order and enable_skip_layer_norm_fusion:
                    model_suffix = "opt_skiplayernorm"
                else:
                    model_suffix = "opt_no_skiplayernorm"

                model_name = f"gpt2_attention_{model_suffix}.onnx"
                self.verify_fusion(optimized_model, model_name)

    def test_bart_attention_sdpa_fusion(self):
        hidden_size = 16
        num_heads = 4
        for with_mask in [True, False]:
            model = create_bart_attention_sdpa(
                hidden_size=hidden_size,
                num_heads=num_heads,
                with_mask=with_mask,
            )

            options = FusionOptions("bart")
            # Disable SkipLayerNorm fusion to match real SDPA BART behaviour,
            # where symbolic shape inference fails and the attention fusion
            # anchor is a plain LayerNormalization node.
            options.enable_skip_layer_norm = False

            optimized_model = optimize_by_fusion(
                model,
                model_type="bart",
                num_heads=num_heads,
                hidden_size=hidden_size,
                optimization_options=options,
            )

            attn_nodes = [n for n in optimized_model.model.graph.node if n.op_type == "Attention"]
            self.assertEqual(
                len(attn_nodes),
                1,
                f"Expected 1 Attention node for with_mask={with_mask}, got {len(attn_nodes)}",
            )

            attn = attn_nodes[0]
            num_heads_attr = next((a for a in attn.attribute if a.name == "num_heads"), None)
            self.assertIsNotNone(num_heads_attr)
            self.assertEqual(num_heads_attr.i, num_heads)

            unidirectional_attr = next((a for a in attn.attribute if a.name == "unidirectional"), None)
            if with_mask:
                # With mask → decoder self-attention → unidirectional=1
                self.assertIsNotNone(unidirectional_attr)
                self.assertEqual(unidirectional_attr.i, 1)
            else:
                # No mask → encoder attention → unidirectional=0
                if unidirectional_attr is not None:
                    self.assertEqual(unidirectional_attr.i, 0)

    def test_gpt2_attention_no_past_fusion(self):
        hidden_size = 64
        num_heads = 4
        for add_cast in [True, False]:
            for switch_add_inputs in [False, True]:
                model = create_gpt2_attention_no_past(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    switch_add_inputs=switch_add_inputs,
                    add_cast=add_cast,
                )
                dir = "."
                model_path = os.path.join(dir, "gpt2_attention_no_past.onnx")
                onnx.save(model, model_path)

                options = FusionOptions("gpt2")

                optimized_model = optimize_model(
                    model_path,
                    model_type="gpt2",
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    optimization_options=options,
                )

                os.remove(model_path)

                model_suffix = "add_opt" if switch_add_inputs else "opt"
                model_name = f"gpt2_attention_no_past_{model_suffix}.onnx"
                self.verify_fusion(optimized_model, model_name)

    def test_megatron_gpt2_attention_fusion(self):
        for enable_skip_layer_norm_fusion in [False, True]:
            path = get_test_data_path("models", "gpt2_megatron.onnx")
            model = onnx.load(path)

            options = FusionOptions("gpt2")
            options.enable_skip_layer_norm = enable_skip_layer_norm_fusion

            optimized_model = optimize_by_fusion(
                model,
                model_type="gpt2",
                num_heads=0,
                hidden_size=0,
                optimization_options=options,
            )

            model_suffix = ""
            if enable_skip_layer_norm_fusion:
                model_suffix = "opt_skiplayernorm"
            else:
                model_suffix = "opt_no_skiplayernorm"

            model_name = f"gpt2_megatron_{model_suffix}.onnx"
            self.verify_fusion(optimized_model, model_name)

    def test_qwen3_normalization_fusion(self):
        """Test Qwen3 decoder layer optimization.

        Verifies that the optimizer fuses RMSNorm patterns into SimplifiedLayerNormalization.
        """
        hidden_size = 64
        num_heads = 8
        num_kv_heads = 2

        model = create_qwen3_decoder_layer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        dir = tempfile.mkdtemp()
        model_path = os.path.join(dir, "qwen3_decoder.onnx")
        onnx.save(model, model_path)

        options = FusionOptions("qwen3")
        optimized_model = optimize_model(
            model_path,
            model_type="qwen3",
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=options,
        )

        os.remove(model_path)

        nodes = optimized_model.model.graph.node
        sln_count = sum(1 for n in nodes if n.op_type == "SimplifiedLayerNormalization")

        # 4 RMSNorm patterns all fuse into SimplifiedLayerNormalization:
        # pre-attn, Q-norm, K-norm, post-attn.
        self.assertEqual(
            sln_count,
            4,
            f"Expected 4 SimplifiedLayerNormalization, got {sln_count}",
        )

    def test_qwen3_rotary_embedding_fusion(self):
        """Test Qwen3 RotaryEmbedding fusion for on-the-fly RoPE with dynamic Slice indices.

        Verifies that the optimizer fuses:
          - On-the-fly RoPE (MatMul → Cos/Sin → Mul(scaling)) into RotaryEmbedding nodes
          - Both Q and K paths get RotaryEmbedding fusion (2 total)
        """
        hidden_size = 64
        num_heads = 8
        num_kv_heads = 2

        model = create_qwen3_decoder_layer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            include_rope=True,
        )

        dir = tempfile.mkdtemp()
        model_path = os.path.join(dir, "qwen3_decoder_rope.onnx")
        onnx.save(model, model_path)

        options = FusionOptions("qwen3")
        optimized_model = optimize_model(
            model_path,
            model_type="qwen3",
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=options,
        )

        os.remove(model_path)

        nodes = optimized_model.model.graph.node
        rope_count = sum(1 for n in nodes if n.op_type == "RotaryEmbedding")
        sln_count = sum(1 for n in nodes if n.op_type == "SimplifiedLayerNormalization")

        self.assertEqual(
            rope_count,
            2,
            f"Expected 2 RotaryEmbedding (Q + K), got {rope_count}",
        )
        self.assertEqual(
            sln_count,
            4,
            f"Expected 4 SimplifiedLayerNormalization, got {sln_count}",
        )

    def test_qwen3_rotary_embedding_fusion_with_expand(self):
        """Test RotaryEmbedding fusion when inv_freq path includes Cast → Expand → Where nodes."""
        hidden_size = 64
        num_heads = 8
        num_kv_heads = 2

        model = create_qwen3_decoder_layer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            include_rope=True,
            include_expand_in_inv_freq=True,
        )

        dir = tempfile.mkdtemp()
        model_path = os.path.join(dir, "qwen3_decoder_rope_expand.onnx")
        onnx.save(model, model_path)

        options = FusionOptions("qwen3")
        optimized_model = optimize_model(
            model_path,
            model_type="qwen3",
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=options,
        )

        os.remove(model_path)

        nodes = optimized_model.model.graph.node
        rope_count = sum(1 for n in nodes if n.op_type == "RotaryEmbedding")

        self.assertEqual(
            rope_count,
            2,
            f"Expected 2 RotaryEmbedding (Q + K) with Expand in inv_freq path, got {rope_count}",
        )

    def test_qwen3_rotary_embedding_fusion_negative_dynamic_inv_freq(self):
        """Test that RotaryEmbedding fusion gracefully falls back when inv_freq is a dynamic graph input.

        When inv_freq is not a constant initializer (e.g., computed dynamically), the fusion cannot
        extract the values at optimization time and should skip fusion without crashing.
        """
        hidden_size = 64
        num_heads = 8
        num_kv_heads = 2

        model = create_qwen3_decoder_layer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            include_rope=True,
            inv_freq_as_graph_input=True,
        )

        dir = tempfile.mkdtemp()
        model_path = os.path.join(dir, "qwen3_decoder_rope_dynamic_inv_freq.onnx")
        onnx.save(model, model_path)

        options = FusionOptions("qwen3")
        optimized_model = optimize_model(
            model_path,
            model_type="qwen3",
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=options,
        )

        os.remove(model_path)

        nodes = optimized_model.model.graph.node
        rope_count = sum(1 for n in nodes if n.op_type == "RotaryEmbedding")

        # Fusion should gracefully skip — 0 RotaryEmbedding nodes, no crash
        self.assertEqual(
            rope_count,
            0,
            f"Expected 0 RotaryEmbedding when inv_freq is dynamic, got {rope_count}",
        )

    def test_qwen3_rotary_embedding_fusion_cache_numerical_validation(self):
        """Test that the generated cos/sin caches have correct values.

        Verifies the mathematical correctness of the precomputed caches:
            freqs[pos, i] = pos * inv_freq[i]
            cos_cache[pos, i] = cos(freqs[pos, i]) * scaling
            sin_cache[pos, i] = sin(freqs[pos, i]) * scaling
        """
        hidden_size = 64
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden_size // num_heads  # 8
        half_dim = head_dim // 2  # 4

        model = create_qwen3_decoder_layer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            include_rope=True,
        )

        dir = tempfile.mkdtemp()
        model_path = os.path.join(dir, "qwen3_decoder_rope_numerical.onnx")
        onnx.save(model, model_path)

        options = FusionOptions("qwen3")
        optimized_model = optimize_model(
            model_path,
            model_type="qwen3",
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=options,
        )

        os.remove(model_path)

        # Verify cos/sin cache initializers exist
        cos_init = optimized_model.get_initializer("cos_cache")
        sin_init = optimized_model.get_initializer("sin_cache")
        self.assertIsNotNone(cos_init, "cos_cache initializer not found after fusion")
        self.assertIsNotNone(sin_init, "sin_cache initializer not found after fusion")

        cos_data = numpy_helper.to_array(cos_init)
        sin_data = numpy_helper.to_array(sin_init)

        # Verify shape: (max_seq_len, head_dim // 2)
        self.assertEqual(cos_data.shape[1], half_dim, f"cos_cache dim 1 should be {half_dim}")
        self.assertEqual(sin_data.shape[1], half_dim, f"sin_cache dim 1 should be {half_dim}")
        self.assertEqual(cos_data.shape, sin_data.shape, "cos_cache and sin_cache shapes should match")

        # Recompute expected values from inv_freq (must match the generator's formula)
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        scaling = 1.0  # attention_scaling in the test generator

        # Spot-check at several positions
        for pos in [0, 1, 7, 100, 1000]:
            expected_freqs = pos * inv_freq
            expected_cos = np.cos(expected_freqs) * scaling
            expected_sin = np.sin(expected_freqs) * scaling
            np.testing.assert_allclose(
                cos_data[pos],
                expected_cos,
                rtol=1e-6,
                err_msg=f"cos_cache mismatch at position {pos}",
            )
            np.testing.assert_allclose(
                sin_data[pos],
                expected_sin,
                rtol=1e-6,
                err_msg=f"sin_cache mismatch at position {pos}",
            )


if __name__ == "__main__":
    unittest.main()
