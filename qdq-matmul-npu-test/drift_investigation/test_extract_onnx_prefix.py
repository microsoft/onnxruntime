"""Regression tests for slicing a graph at a QDQ output."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

from drift_investigation.extract_onnx_prefix import make_prefix
from split_gemma_vision_pooler import save_component
from drift_investigation.test_qdq_ablation import make_model


class ExtractOnnxPrefixTest(unittest.TestCase):
    def test_prefix_preserves_cut_value_and_required_input(self):
        model = ir.serde.deserialize_model(make_model())
        prefix = make_prefix(model, "a_dq", "cut_output")
        self.assertEqual([value.name for value in prefix.graph.inputs], ["x"])
        self.assertEqual([value.name for value in prefix.graph.outputs], ["cut_output"])
        self.assertEqual(prefix.graph.num_nodes(), 2)
        result = ort.InferenceSession(ir.serde.serialize_model(prefix).SerializeToString()).run(
            None, {"x": np.asarray([[-0.4, 0.25, 1.1, 4.0]], dtype=np.float32)}
        )[0]
        np.testing.assert_array_equal(result, [[-0.5, 0, 1, 4]])
        with tempfile.TemporaryDirectory() as temp:
            portable = Path(temp) / "subfolder" / "prefix.onnx"
            save_component(prefix, portable, False)
            np.testing.assert_array_equal(
                ort.InferenceSession(str(portable)).run(
                    None, {"x": np.asarray([[-0.4, 0.25, 1.1, 4.0]], dtype=np.float32)}
                )[0],
                result,
            )


if __name__ == "__main__":
    unittest.main()
