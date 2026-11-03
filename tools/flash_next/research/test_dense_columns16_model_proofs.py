import ast
import unittest

from prepare_dense_columns16_model_proofs import model_source, screen_source


class ModelGateTests(unittest.TestCase):
    def test_complete_actual_and_default_and_invalid_prerequisites(self):
        source = model_source()
        ast.parse(source)
        self.assertIn('["capture", "default", "compare"]', source)
        self.assertIn('actual["status"] == "dense_columns16_original_provider_actual_weights_supported"', source)
        self.assertIn('invalid["status"] == "fresh_invalid_columns16_selector_rejected"', source)
        self.assertNotIn('fp32_path', source)
        self.assertIn("build[\"baseline_provider_sha256\"] == parent[\"provider_sha256\"]", source)
        self.assertIn("qwen38-int4-qmoe-prefill-rows4/build-v2/", source)
        self.assertIn('selected = "DenseColumns16"', source)
        self.assertIn('counts[selected] == 159', source)

    def test_all_contexts_and_bracketed_full_retained_screen_preserved(self):
        source = screen_source()
        ast.parse(source)
        self.assertIn("compare_samples", source)
        self.assertIn("(2048, 8192, 32768)", source)
        self.assertIn("for full in (True, False)", source)
        self.assertIn('("control", trial), ("columns16", trial), ("control", trial + 1)', source)
        self.assertIn('assert initial["status"] == "initial_columns16_decode_gain_both_modes"', source)


if __name__ == "__main__":
    unittest.main()
