import os
import sys
import unittest
import math
import onnx
import torch

from bert_model_generator import create_bert_skip_layer_norm

# set path so that we could import from parent directory
transformers_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python', 'tools', 'transformers')
if os.path.exists(transformers_dir):
    sys.path.append(transformers_dir)
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.optimizer import optimize_model


class TestEmbedLayerNormBiasGeluFusions(unittest.TestCase):
    def test_fusions(self):
        model = create_bert_skip_layer_norm()
        dir = '.'
        model_path = os.path.join(dir, 'embedlayernorm_bias_gelu.onnx')
        onnx.save(model, model_path)
        optimized_model = optimize_model(model_path)

        #onnx.save(optimized_model.model, os.path.join(dir, 'embedlayernorm_bias_gelu_opt.onnx'))
        os.remove(model_path)

        expected_model_path = os.path.join(os.path.dirname(__file__), 'test_data', 'models',
                                           'embedlayernorm_bias_gelu_opt.onnx')
        expected = onnx.load(expected_model_path)
        self.assertEqual(str(optimized_model.model.graph), str(expected.graph))


if __name__ == '__main__':
    unittest.main()
