import ast
import unittest

from prepare_dense_paired_matrix_actual_proofs import make_runner


class ActualWeightTests(unittest.TestCase):
    def test_default_and_native_and_original_processes_are_distinct(self):
        code = make_runner()
        ast.parse(code)
        self.assertIn('("capture", 0, ORIGINAL), ("default", 0, provider), ("compare", 1, provider)', code)
        self.assertIn('"compare" if mode == "default" else mode', code)
        self.assertIn('if mode in ("capture", "default") or n == 1:', code)
        self.assertIn("Compiled default-off provider differs from original", code)
        self.assertIn('phases[-1]["mismatches"]', code)
        self.assertNotIn("(2560, 1)", code)

    def test_actual_payload_and_synthetic_prerequisite_are_gated(self):
        code = make_runner()
        self.assertIn("load_external_data=False", code)
        self.assertIn("load_external_data_for_tensor", code)
        self.assertIn('hashlib.sha256(array.tobytes()).hexdigest()', code)
        self.assertIn('attrs.get("block_size") != 32', code)
        self.assertIn("actual_model_weights_used=True", code)
        self.assertIn("Synthetic original-provider gate not supported", code)
        self.assertIn('DIRECTORY / "actual-fixtures-v1"', code)
        self.assertIn("model_eligible=False", code)


if __name__ == "__main__":
    unittest.main()
