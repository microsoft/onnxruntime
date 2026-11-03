import unittest

from prepare_dense_columns8_double_buffer_proofs import batch_source


class DoubleBufferProofTests(unittest.TestCase):
    def test_independent_original_default_candidate_batches(self):
        for actual in (False, True):
            with self.subTest(actual=actual):
                source = batch_source(actual)
                self.assertIn('("capture", ORIGINAL, "capture", 0)', source)
                self.assertIn('("default", provider, "compare", 0)', source)
                self.assertIn('("candidate", provider, "compare", 1)', source)
                self.assertIn('"; ".join(commands)', source)
                self.assertIn("DenseColumns8DoubleBuffer", source)
                self.assertIn('FLAG = "ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER"', source)
                self.assertIn('ROOT / "build_dense_columns8_double_buffer.py"', source)
                self.assertIn('ROOT / "prepare_dense_columns8_double_buffer_proofs.py"', source)
                self.assertIn("qwen38-int4-dense-columns8-double-buffer/build-v1", source)
                self.assertIn("model_eligible=False", source)
                self.assertIn("natural_model_timing_performed=False", source)
                self.assertIn('reference.exists()', source)
                self.assertNotIn("double_buffer_double_buffer", source)
                self.assertNotIn("double-buffer-double-buffer", source)
                parent = ("dense-paired-matrix-reset-actual-v1-outcome.json" if actual
                          else "dense-columns16-v1-outcome.json")
                self.assertIn(parent, source)


if __name__ == "__main__":
    unittest.main()
