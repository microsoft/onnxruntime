# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import onnx
from gpt2_model_generator import create_gpt2_embedlayer
from parity_utilities import find_transformers_source

if find_transformers_source():
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model


class TestFusion(unittest.TestCase):
    def verify_fusion(self, optimized_model, expected_model_filename):
        optimized_model.topological_sort()

        expected_model_path = os.path.join(expected_model_filename)
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort()

        self.assertEqual(str(optimized_model.model.graph), str(expected_model.model.graph))

    def test_embedlayer_fusion(self):
        model = create_gpt2_embedlayer(one_attention_node=False)
        path = "."
        model_path = os.path.join(path, "gpt2_embedlayer.onnx")
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path, model_type="gpt2")
        os.remove(model_path)

        self.verify_fusion(optimized_model, "gpt2_embedlayer_opt.onnx")

    def test_embedlayer_fusion_one_attn_node(self):
        model = create_gpt2_embedlayer(one_attention_node=True)
        path = "."
        model_path = os.path.join(path, "gpt2_embedlayer_one_attn.onnx")
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path, model_type="gpt2")
        os.remove(model_path)

        self.verify_fusion(optimized_model, "gpt2_embedlayer_one_attn_opt.onnx")

if __name__ == "__main__":
    unittest.main()
