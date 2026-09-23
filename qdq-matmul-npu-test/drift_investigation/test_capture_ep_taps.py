"""Regression tests for trustworthy NPU intermediate capture."""

import unittest

import numpy as np

from drift_investigation.capture_ep_taps import parse_input_values, select_taps


class CaptureEpTapsTest(unittest.TestCase):
    def test_requires_same_unprobed_ep_output(self):
        outputs = {
            "full": np.asarray([[1.0]], dtype=np.float32),
            "q": np.asarray([[2.0]], dtype=np.float32),
            "context": np.asarray([[3.0]], dtype=np.float32),
        }
        inputs, context = select_taps(outputs, outputs["full"], "full", {"query": "q"}, "context")
        np.testing.assert_array_equal(inputs["query"], outputs["q"])
        np.testing.assert_array_equal(context, outputs["context"])
        with self.assertRaisesRegex(RuntimeError, "tapping changed EP output"):
            select_taps(outputs, np.asarray([[2.0]], dtype=np.float32), "full", {"query": "q"}, "context")
        with self.assertRaisesRegex(ValueError, "does not expose"):
            select_taps(outputs, outputs["full"], "full", {"query": "missing"}, "context")

    def test_rejects_duplicate_or_incomplete_input_mapping(self):
        self.assertEqual(parse_input_values(["query=attn_q", "key=attn_k"]),
                         {"query": "attn_q", "key": "attn_k"})
        with self.assertRaisesRegex(ValueError, "invalid or repeated"):
            parse_input_values(["query=attn_q", "query=other"])


if __name__ == "__main__":
    unittest.main()
