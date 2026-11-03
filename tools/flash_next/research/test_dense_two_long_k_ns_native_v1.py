from pathlib import Path
import unittest

from prepare_dense_two_long_k_ns_native_v1 import controller_source


ROOT = Path(__file__).parent


class FullNativePreciseScopeTests(unittest.TestCase):
    def test_original_full_native_coverage_and_precise_control(self):
        actual = controller_source((ROOT / "run_dense_long_k_native_v3.py").read_text())
        self.assertIn("TOTAL = 116582784", actual)
        self.assertIn("len(fixtures) != 27", actual)
        self.assertIn('"replay-fp32-selected"', actual)
        self.assertIn("if len(rows) != 9", actual)
        self.assertIn("precise_duration_us(event)", actual)
        self.assertIn('":256,1,1:"', actual)
        self.assertIn('manifest["supported_n"] != [2560]', actual)
        self.assertIn('control["provider_sha256"]', actual)
        self.assertIn('"dense-tile-parallel2-long-k-ns-v2-outcome.json"', actual)
        self.assertIn('(COMPUTE / "matmul_nbits.cc")', actual)
        self.assertNotIn("parallel3", actual)
        compile(actual, "dense-two-long-k-full-native", "exec")


if __name__ == "__main__":
    unittest.main()
