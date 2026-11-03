import ast
from pathlib import Path
import unittest

from prepare_dense_columns8_actual_model import actual_source, model, invalid


class BatchActualModelTests(unittest.TestCase):
    def test_actual_weight_source_and_all_phases(self):
        source = actual_source()
        ast.parse(source)
        self.assertIn("qwen38-int4-dense-paired-matrix-reset/actual-fixtures-v1", source)
        self.assertIn("for k, n in CASES if n != 1", source)
        self.assertIn("actual_model_weights_used=True", source)
        self.assertIn("columns8_batched_actual_and_default_supported", source)
        self.assertIn("model_eligible=False", source)

    def test_model_prerequisites_and_narrow_final_projection_fallback(self):
        source = model()
        ast.parse(source)
        self.assertIn('["capture", "default", "candidate"]', source)
        self.assertIn('row["selected_dispatches"] == (64 if row["n"] >= 2560 else 0)', source)
        self.assertIn('(158 if mode == "columns8" else 159)', source)
        self.assertIn('(1 if mode == "columns8" else 0)', source)
        self.assertIn('native_path = ROOT / "dense-columns8-actual-batch-v1-outcome.json"', source)
        self.assertIn("qwen38-int4-qmoe-prefill-rows4/build-v2/", source)
        self.assertIn('choices=(2048, 8192, 32768)', source)

    def test_fresh_invalid_prerequisite_and_model_final_projection_count(self):
        source = invalid()
        ast.parse(source)
        self.assertIn("columns8_batched_actual_and_default_supported", source)
        self.assertIn("fresh_invalid_columns8_selector_rejected", source)
        self.assertIn("qwen38-int4-dense-paired-matrix-reset/actual-fixtures-v1", source)
        import onnx
        graph = onnx.load("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model/model.onnx",
                          load_external_data=False).graph
        count = 0
        for node in graph.node:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                count += (attrs.get("K"), attrs.get("N")) == (10240, 320)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
