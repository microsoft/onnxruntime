import os
import unittest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from onnx_exporter import export_onnx_model_from_pt
from huggingface_models import MODELS
from benchmark_helper import Precision
from shape_infer_helper import *


class SymbolicShapeInferenceHelperTest(unittest.TestCase):
    def _load_onnx(self, model_name):
        input_names = MODELS[model_name][0]
        base_path = "../onnx_models/"
        import torch
        with torch.no_grad():
            export_onnx_model_from_pt(model_name, MODELS[model_name][1], MODELS[model_name][2], MODELS[model_name][3],
                                      None, '../cache_models', base_path, input_names[:1], False, Precision.FLOAT32,
                                      True, True, True, False, {})
        model_path = base_path + model_name.replace('-', '_') + "_1.onnx"
        import onnx
        return onnx.load_model(model_path)

    def test_bert_shape_infer_helper(self):
        model = self._load_onnx("bert-base-cased")
        shape_infer_helper = SymbolicShapeInferenceHelper(model)
        self.assertEqual(shape_infer_helper.infer({"batch_size": 4, "seq_len": 16}), True)
        self.assertEqual(shape_infer_helper.get_edge_shape("802"), [4, 16, 768])
        self.assertEqual(shape_infer_helper.get_edge_shape("804"), [4, 16, 1])
        self.assertEqual(shape_infer_helper.get_edge_shape("1748"), [])
        self.assertEqual(shape_infer_helper.get_edge_shape("encoder.layer.4.attention.output.LayerNorm.weight"), [768])
        self.assertEqual(shape_infer_helper.get_edge_shape("1749"), [768, 3072])
        self.assertEqual(shape_infer_helper.get_edge_shape("817"), [4, 16, 3072])
        self.assertEqual(shape_infer_helper.get_edge_shape("encoder.layer.4.intermediate.dense.bias"), [3072])
        self.assertEqual(shape_infer_helper.get_edge_shape("1750"), [3072, 768])
        self.assertEqual(shape_infer_helper.get_edge_shape("853"), [3])
        self.assertEqual(shape_infer_helper.get_edge_shape("858"), [1])
        self.assertEqual(shape_infer_helper.get_edge_shape("880"), [4, 16, 12, 64])

        self.assertEqual(shape_infer_helper.compare_shape("329", "253"), True)
        self.assertEqual(shape_infer_helper.compare_shape("447", "371"), True)
        self.assertEqual(shape_infer_helper.compare_shape("329", "817"), False)
        self.assertEqual(shape_infer_helper.compare_shape("447", "853"), False)


if __name__ == '__main__':
    unittest.main()
