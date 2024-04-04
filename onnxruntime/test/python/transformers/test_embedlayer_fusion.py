# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import numpy as np
import onnx
from gpt2_model_generator import create_gpt2_embedlayer
from parity_utilities import find_transformers_source

from onnxruntime import InferenceSession

if find_transformers_source():
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model


class TestFusion(unittest.TestCase):
    def verify_fusion(self, optimized_model, expected_model_filename):
        optimized_model.topological_sort()

        expected_model_path = os.path.join(os.path.dirname(__file__), "test_data", "models", expected_model_filename)
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort()

        self.assertEqual(str(optimized_model.model.graph), str(expected_model.model.graph))

    def verify_parity(self, optimized_model_path, expected_model_filename):
        expected_model_path = os.path.join(os.path.dirname(__file__), "test_data", "models", expected_model_filename)
        sess_optimized = InferenceSession(optimized_model_path, providers=["CPUExecutionProvider"])
        sess_expected = InferenceSession(expected_model_path, providers=["CPUExecutionProvider"])
        inputs = np.random.randint(low=0, high=6, size=(4, 8), dtype=np.int32) + 1

        outputs_optimized = sess_optimized.run(None, {"ids": inputs})
        outputs_expected = sess_expected.run(None, {"ids": inputs})
        self.assertTrue(np.allclose(outputs_optimized[0], outputs_expected[0]))

    def test_embedlayer_fusion(self):
        model = create_gpt2_embedlayer(one_attention_node=False)
        path = "."
        original_model_path = os.path.join(path, "gpt2_embedlayer.onnx")
        optimized_model_path = os.path.join(path, "gpt2_embedlayer_opt.onnx")
        expected_model_filename = "gpt2_embedlayer_exp.onnx"

        onnx.save(model, original_model_path)
        optimized_model = optimize_model(original_model_path, model_type="gpt2")
        optimized_model.save_model_to_file(optimized_model_path, use_external_data_format=True)

        self.verify_fusion(optimized_model, expected_model_filename)
        self.verify_parity(optimized_model_path, expected_model_filename)
        os.remove(original_model_path)
        os.remove(optimized_model_path)

    def test_embedlayer_fusion_one_attn_node(self):
        model = create_gpt2_embedlayer(one_attention_node=True)
        path = "."
        original_model_path = os.path.join(path, "gpt2_embedlayer_one_attn.onnx")
        optimized_model_path = os.path.join(path, "gpt2_embedlayer_one_attn_opt.onnx")
        expected_model_filename = "gpt2_embedlayer_one_attn_exp.onnx"

        onnx.save(model, original_model_path)
        optimized_model = optimize_model(original_model_path, model_type="gpt2")
        optimized_model.save_model_to_file(optimized_model_path, use_external_data_format=True)

        self.verify_fusion(optimized_model, expected_model_filename)
        self.verify_parity(optimized_model_path, expected_model_filename)
        os.remove(original_model_path)
        os.remove(optimized_model_path)

    def test_embedlayer_fusion_with_embedding_sum_output(self):
        model = create_gpt2_embedlayer(one_attention_node=True, output_embedding_sum=True)
        path = "."
        original_model_path = os.path.join(path, "gpt2_embedlayer_one_attn_output_sum.onnx")
        optimized_model_path = os.path.join(path, "gpt2_embedlayer_one_attn_output_sum_opt.onnx")
        expected_model_filename = "gpt2_embedlayer_one_attn_output_sum_exp.onnx"

        onnx.save(model, original_model_path)
        optimized_model = optimize_model(original_model_path, model_type="gpt2")
        optimized_model.save_model_to_file(optimized_model_path, use_external_data_format=True)

        self.verify_fusion(optimized_model, expected_model_filename)
        self.verify_parity(optimized_model_path, expected_model_filename)
        os.remove(original_model_path)
        os.remove(optimized_model_path)

    def test_embedlayer_fusion_with_embedding_sum_output_no_sln(self):
        model = create_gpt2_embedlayer(one_attention_node=True, has_skip_layer_norm=False, output_embedding_sum=True)
        path = "."
        original_model_path = os.path.join(path, "gpt2_embedlayer_one_attn_output_sum_no_sln.onnx")
        optimized_model_path = os.path.join(path, "gpt2_embedlayer_one_attn_output_sum_no_sln_opt.onnx")
        expected_model_filename = "gpt2_embedlayer_one_attn_output_sum_exp.onnx"

        onnx.save(model, original_model_path)
        optimized_model = optimize_model(original_model_path, model_type="gpt2")
        optimized_model.save_model_to_file(optimized_model_path, use_external_data_format=True)

        self.verify_fusion(optimized_model, expected_model_filename)
        self.verify_parity(optimized_model_path, expected_model_filename)
        os.remove(original_model_path)
        os.remove(optimized_model_path)


if __name__ == "__main__":
    unittest.main()
