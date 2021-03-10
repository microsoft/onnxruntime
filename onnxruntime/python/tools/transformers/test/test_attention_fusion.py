# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import os
import sys
import onnx
from bert_model_generator import create_bert_attention

# set path so that we could import from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from optimizer import optimize_model


class TestFusion(unittest.TestCase):
    def test_attention_fusion_pruned_model(self):
        model = create_bert_attention()
        dir = '.'
        model_path = os.path.join(dir, "pruned_attention.onnx")
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path)
        os.remove(model_path)

        expected_model_path = os.path.join(os.path.dirname(__file__), 'test_data', 'fusion',
                                           'pruned_attention_opt.onnx')
        expected = onnx.load(expected_model_path)
        self.assertEqual(str(optimized_model.model.graph), str(expected.graph))


if __name__ == '__main__':
    unittest.main()
