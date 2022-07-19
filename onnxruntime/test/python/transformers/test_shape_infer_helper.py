import unittest

import onnx
import pytest
import torch
from parity_utilities import find_transformers_source

if find_transformers_source():
    from benchmark_helper import ConfigModifier, OptimizerInfo, Precision
    from huggingface_models import MODELS
    from onnx_exporter import export_onnx_model_from_pt
    from shape_infer_helper import SymbolicShapeInferenceHelper
else:
    from onnxruntime.transformers.benchmark_helper import ConfigModifier, OptimizerInfo, Precision
    from onnxruntime.transformers.huggingface_models import MODELS
    from onnxruntime.transformers.onnx_exporter import export_onnx_model_from_pt
    from onnxruntime.transformers.shape_infer_helper import SymbolicShapeInferenceHelper


class SymbolicShapeInferenceHelperTest(unittest.TestCase):
    def _load_onnx(self, model_name):
        input_names = MODELS[model_name][0]
        base_path = "../onnx_models/"

        config_modifier = ConfigModifier(None)
        fusion_options = None
        model_class = "AutoModel"
        with torch.no_grad():
            export_onnx_model_from_pt(
                model_name,
                MODELS[model_name][1],
                MODELS[model_name][2],
                MODELS[model_name][3],
                model_class,
                config_modifier,
                "../cache_models",
                base_path,
                input_names[:1],
                False,
                Precision.FLOAT32,
                OptimizerInfo.BYSCRIPT,
                True,
                True,
                False,
                {},
                fusion_options,
            )
        model_path = base_path + model_name.replace("-", "_") + "_1.onnx"
        return onnx.load_model(model_path)

    # TODO: use a static lightweight model for test
    @pytest.mark.slow
    def test_bert_shape_infer_helper(self):
        model = self._load_onnx("bert-base-cased")
        shape_infer_helper = SymbolicShapeInferenceHelper(model)
        self.assertEqual(shape_infer_helper.infer({"batch_size": 4, "seq_len": 16}), True)
        self.assertEqual(shape_infer_helper.get_edge_shape("802"), [])
        self.assertEqual(shape_infer_helper.get_edge_shape("804"), [4, 16, 3072])
        self.assertEqual(shape_infer_helper.get_edge_shape("1748"), [1])
        self.assertEqual(
            shape_infer_helper.get_edge_shape("encoder.layer.4.attention.output.LayerNorm.weight"),
            [768],
        )
        self.assertEqual(shape_infer_helper.get_edge_shape("817"), [4, 16, 1])
        self.assertEqual(
            shape_infer_helper.get_edge_shape("encoder.layer.4.intermediate.dense.bias"),
            [3072],
        )
        self.assertEqual(shape_infer_helper.get_edge_shape("880"), [4, 12, 16, 16])

        self.assertEqual(shape_infer_helper.compare_shape("329", "253"), False)
        self.assertEqual(shape_infer_helper.compare_shape("447", "371"), False)
        self.assertEqual(shape_infer_helper.compare_shape("329", "817"), True)
        self.assertEqual(shape_infer_helper.compare_shape("447", "853"), False)


if __name__ == "__main__":
    unittest.main()
