from pathlib import Path
import unittest

from prepare_dense_two_long_k_model_v1 import controller_source


ROOT = Path(__file__).parent


class TwoWorkerModelTests(unittest.TestCase):
    def test_original_controls_and_unchanged_compute_linkage(self):
        source = controller_source((ROOT / "run_dense_long_k_model_v1.py").read_text())
        self.assertIn("COMPUTE as BUILD, BUILD as PRECISE_BUILD", source)
        self.assertIn('precise["unmodified_candidate_byte_identical"]', source)
        self.assertIn('precise["unmodified_candidate_provider_sha256"] if flag else build["baseline_provider_sha256"]', source)
        self.assertIn('raw["plugin_sha256"] != expected_provider_sha', source)
        self.assertIn('":256,1,1:"', source)
        self.assertIn("benchmark_dense_two_long_k_model_v1.py", source)
        self.assertIn("--tokens 256 --repetitions 3", source)
        self.assertIn("decode_tps=3 * 255 /", source)
        self.assertIn('if result["status"] != "dense_long_k_gain_both_modes"', source)
        self.assertIn("for repeat in (False, True)", source)
        self.assertNotIn("parallel3", source)
        compile(source, "two-worker-natural-model", "exec")


if __name__ == "__main__":
    unittest.main()
