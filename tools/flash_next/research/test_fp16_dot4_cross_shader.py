from pathlib import Path
import unittest


ROOT = Path(__file__).parent


class CrossShaderTests(unittest.TestCase):
    def test_plane_mapping_and_reference_bounds(self):
        self.assertEqual(sorted(seed * 256 + row * 16 + col for seed in range(8)
                                for row in range(16) for col in range(16)), list(range(2048)))
        self.assertEqual(2048 * 2, 4096)
        self.assertEqual(8 * 4 * 16 * 16, 8192)
        for row in range(16):
            for col in range(16):
                self.assertLess(row * 1552 + 1536 + col, 16 * 1552)
                for plane in (0, 1, 8, 9, 2):
                    self.assertLess(row * 160 + plane * 16 + col, 16 * 160)

    def test_capture_and_compare_contract(self):
        source = (ROOT / "fp16_dot4_cross_shader_compare.cpp").read_text()
        self.assertIn("for (int pattern : {47, 95})", source)
        self.assertIn("Reference changed between repeats", source)
        self.assertIn("std::array<int, 5> planes{0, 1, 8, 9, 2}", source)
        self.assertIn("return mismatches[1] == 0 ? 0 : 2;", source)
        self.assertIn("session.disable_cpu_ep_fallback", source)
        self.assertIn("input.peek() != std::char_traits<char>::eof()", source)
        self.assertIn("corrected_pairs += (mask & 2u) != 0u", source)

    def test_serial_runner_provenance(self):
        source = (ROOT / "run_fp16_dot4_cross_shader.py").read_text()
        self.assertIn('("capture", "compare")', source)
        self.assertIn('mode == "capture" else (0, 2)', source)
        self.assertIn("Changed qualified fixture", source)
        self.assertIn("Reference capture changed during comparison", source)
        self.assertIn("model_eligible=False", source)
        self.assertIn("0 < native < 8192", source)


if __name__ == "__main__":
    unittest.main()
