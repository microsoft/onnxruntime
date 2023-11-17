# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import onnx
from ct_model_generator import create_conformer_attention
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
            os.path.dirname(__file__), "test_data", "models", "ct", expected_model_filename
        )
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
            print("Expected initializer done = ", expected_initializer.name)

    def test_ct_mha_fusion(self):
        num_heads = 8
        hidden_size = 512
        model = create_conformer_attention(
            num_heads=num_heads, hidden_size=hidden_size, add_before_layernorm=False, fused=True
        )
        dir = "."
        model_path = os.path.join(dir, "conformer_self_mha.onnx")
        onnx.save(model, model_path)
        options = FusionOptions("ct")
        optimized_model = optimize_model(
            model_path, model_type="ct", num_heads=num_heads, hidden_size=hidden_size, optimization_options=options
        )
        os.remove(model_path)
        self.verify_fusion(optimized_model, "conformer_self_mha_fused.onnx")


if __name__ == "__main__":
    unittest.main()
