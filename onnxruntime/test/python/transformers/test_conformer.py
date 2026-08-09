# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import tempfile
import unittest

import onnx
from conformer_model_generator import (
    create_conformer_attention,
    create_conformer_attention_no_add_kv,
    create_conformer_attention_qk_div_masking,
    create_conformer_attention_simple_bias,
)
from parity_utilities import find_transformers_source

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
            os.path.dirname(__file__), "test_data", "models", "conformer", expected_model_filename
        )
        print("Expected model path = ", expected_model_path)
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort(is_deterministic=True)

        nodes = optimized_model.model.graph.node
        self.assertEqual(len(nodes), len(expected_model.model.graph.node))

        for i in range(len(nodes)):
            self.assertEqual(nodes[i], expected_model.model.graph.node[i])

        for expected_initializer in expected_model.model.graph.initializer:
            print("Expected initializer initial = ", expected_initializer.name)
            self.assertTrue(
                OnnxModel.has_same_value(
                    optimized_model.get_initializer(expected_initializer.name), expected_initializer
                )
            )

    def count_fused_attention_nodes(self, optimized_model):
        """Return the number of Attention and MultiHeadAttention nodes in the optimized graph."""
        return sum(
            1
            for node in optimized_model.model.graph.node
            if node.op_type in ("Attention", "MultiHeadAttention") and node.domain == "com.microsoft"
        )

    def _run_conformer_optimization(self, model, num_heads, hidden_size):
        """Save the model to a temp file, run the conformer optimizer, and return the result."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            model_path = f.name
        try:
            onnx.save(model, model_path)
            options = FusionOptions("conformer")
            optimized = optimize_model(
                model_path,
                model_type="conformer",
                num_heads=num_heads,
                hidden_size=hidden_size,
                optimization_options=options,
            )
        finally:
            os.remove(model_path)
        return optimized

    def test_ct_mha_fusion(self):
        num_heads = 8
        hidden_size = 512
        model = create_conformer_attention(num_heads=num_heads, hidden_size=hidden_size, add_before_layernorm=False)
        dir = "."
        model_path = os.path.join(dir, "conformer_self_mha.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("conformer")
        optimized_model = optimize_model(
            model_path,
            model_type="conformer",
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=options,
        )
        os.remove(model_path)
        self.verify_fusion(optimized_model, "conformer_self_mha_fused.onnx")

    def test_conformer_no_extra_q_nodes(self):
        """Regression test: standard conformer without positional embedding extra-Q path.

        Before the fix, the extra_q_nodes block required one of the two branch patterns to match.
        When neither matched (simple QK bias, no CT or Nemotron positional embed), fusion would
        incorrectly return early. This test verifies that fusion still produces a fused attention
        node when extra_q_nodes is None throughout.
        """
        num_heads = 4
        hidden_size = 64
        model = create_conformer_attention_simple_bias(num_heads=num_heads, hidden_size=hidden_size)
        optimized = self._run_conformer_optimization(model, num_heads, hidden_size)
        fused_count = self.count_fused_attention_nodes(optimized)
        self.assertEqual(fused_count, 1, f"Expected 1 fused attention node, got {fused_count}")

    def test_nemotron_conformer_no_bias_kv(self):
        """Nemotron-like model with no Add-bias in K/V paths and no leading Add in QKV output.

        Exercises the new fallback matchers introduced for Nemotron graph shapes:
          - QKV output path without leading Add: ["MatMul", "Reshape", "Transpose", "MatMul"]
          - Q path (Transpose→Add→Reshape→MatMul, no leading Div/Mul):
              ["Transpose", "Add", "Reshape", "MatMul"] with [0, 0, 0, 0]
          - K/V paths without bias Add:
              ["Transpose", "Reshape", "MatMul"] with [1, 0, 0]
        Because add_k and add_v are None, the fused node must be MultiHeadAttention.
        """
        num_heads = 4
        hidden_size = 64
        model = create_conformer_attention_no_add_kv(num_heads=num_heads, hidden_size=hidden_size)
        optimized = self._run_conformer_optimization(model, num_heads, hidden_size)
        fused_count = self.count_fused_attention_nodes(optimized)
        self.assertEqual(fused_count, 1, f"Expected 1 fused attention node, got {fused_count}")
        # add_k / add_v are None → use_packed_attention_op is False → MultiHeadAttention
        mha_count = sum(
            1
            for node in optimized.model.graph.node
            if node.op_type == "MultiHeadAttention" and node.domain == "com.microsoft"
        )
        self.assertEqual(mha_count, 1, f"Expected MultiHeadAttention node, got {mha_count}")

    def test_conformer_qk_div_masking(self):
        """Conformer with a Where→Softmax→Where→Div→Add→MatMul QK masking path.

        Exercises the new QK fallback:
          ["Where", "Softmax", "Where", "Div", "Add", "MatMul"] with [0, 2, 0, 2, 0, 0]
        which handles graphs where the QK logits are scaled by Div before the Where mask is applied.
        """
        num_heads = 4
        hidden_size = 64
        model = create_conformer_attention_qk_div_masking(num_heads=num_heads, hidden_size=hidden_size)
        optimized = self._run_conformer_optimization(model, num_heads, hidden_size)
        fused_count = self.count_fused_attention_nodes(optimized)
        self.assertEqual(fused_count, 1, f"Expected 1 fused attention node, got {fused_count}")


if __name__ == "__main__":
    unittest.main()
