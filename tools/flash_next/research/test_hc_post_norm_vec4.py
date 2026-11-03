from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr
from io import StringIO

from build_hc_post_norm_vec4 import FLAG, transform, vector_program
from run_hc_up_parallel2_model import main


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class HCPostNormVec4Tests(unittest.TestCase):
    def test_invalid_attempt_is_rejected_before_loading_evidence(self):
        error = StringIO()
        with patch("sys.argv", ["runner", "--attempt", "0"]), redirect_stderr(error):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--attempt must be a positive integer", error.getvalue())

    def test_normalization_and_dot_arithmetic_preserved(self):
        source = SOURCE.read_text()
        original = source[source.index("class HCPostNormProjectionProgram"):
                          source.index("class HCPostNormProjectionF16")]
        program = vector_program(source)
        for expression in (
                "value", "sum", "inv_rms", "dot", "total", "partial", "sigmoid"):
            original_lines = [line for line in original.splitlines()
                              if expression in line and "post_values" not in line]
            actual_lines = [line for line in program.splitlines()
                            if expression in line and not any(
                                word in line for word in ("post_values", "load_post", "store_post"))]
            self.assertEqual(original_lines, actual_lines, expression)

    def test_scalar_store_locations_and_vector_load_lanes_match(self):
        scalar_addresses = [(logical // 32) * 36 + logical % 32 for logical in range(10240)]
        components = {(address // 4, address % 4) for address in scalar_addresses}
        self.assertEqual(len(components), 10240)
        self.assertLess(max(address // 4 for address in scalar_addresses), 2880)
        self.assertEqual(2880 * 8 + 1024 * 4 + 256 * 2, 27648)
        for block in range(320):
            for i in range(4):
                base = block * 36 + i * 8
                self.assertEqual(base % 4, 0)
                self.assertEqual([4 * (base // 4 + half) + lane
                                  for half in range(2) for lane in range(4)],
                                 scalar_addresses[block * 32 + i * 8:block * 32 + i * 8 + 8])

    def test_original_shader_fallback_limits_and_replay_preserved(self):
        source = SOURCE.read_text()
        actual = transform(source)
        self.assertIn(source[source.index("class HCPostNormProjectionProgram"):
                             source.index("class HCPostNormProjectionF16")], actual)
        self.assertIn(source[source.index("    HCPostNormProjectionBaselineNormProgram norm"):], actual)
        self.assertIn(f'"{FLAG}", 0', actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("}, 0, vec4_mode);", actual)
        self.assertEqual(actual.count("limits.maxComputeWorkgroupStorageSize >= 27648"), 1)
        self.assertEqual(actual.count(".SetWorkgroupSize(256).SetDispatchGroupSize(41)"), 1)


if __name__ == "__main__":
    unittest.main()
