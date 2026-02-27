# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import onnx
from bart_model_generator import create_bart_attention_sdpa
from bert_model_generator import create_bert_attention, create_tf2onnx_attention_3d
from gpt2_model_generator import create_gpt2_attention
from model_loader import get_test_data_path
from parity_utilities import find_transformers_source

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


if __name__ == "__main__":
    unittest.main()
