"""Regression tests for exposing QDQ graph values without changing CPU outputs."""

import unittest

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

from drift_investigation.add_onnx_taps import add_taps
from drift_investigation.test_qdq_ablation import make_model


class AddOnnxTapsTest(unittest.TestCase):
    def test_exposes_qdq_value_without_changing_original_output(self):
        proto = make_model()
        model = ir.serde.deserialize_model(proto)
        add_taps(model, ["a_dq"])
        tapped = ir.serde.serialize_model(model)
        self.assertEqual([value.name for value in tapped.graph.output], ["y", "a_dq"])
        inputs = {"x": np.asarray([[-0.4, 0.25, 1.1, 4.0]], dtype=np.float32)}
        original = ort.InferenceSession(proto.SerializeToString()).run(None, inputs)[0]
        actual, quantized = ort.InferenceSession(tapped.SerializeToString()).run(None, inputs)
        np.testing.assert_array_equal(actual, original)
        self.assertEqual(quantized.shape, (1, 4))
        with self.assertRaisesRegex(ValueError, "already a graph output"):
            add_taps(model, ["a_dq"])


if __name__ == "__main__":
    unittest.main()
