from pathlib import Path
import inspect
import unittest

from build_hc_post_norm_wide16 import FLAG, STORAGE_BYTES, transform, wide_program
from qsa_rank_decode_windows import HC_POST_NORM_WIDE16_FLAG, run_qsa_windows


class Wide16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/"
                          "build-v2/hc_post_norm_projection.cc").read_text()

    def test_norm_dot_and_serial_float_addition_order_preserved(self):
        original = self.source[self.source.index("class HCPostNormProjectionProgram final"):
                               self.source.index("class HCPostNormProjectionF16")]
        result = wide_program(self.source)
        original_norm = original[:original.index('<< "let lane =')]
        actual_norm = result[:result.index('<< "let lane =')]
        restored = actual_norm.replace("HCPostNormProjectionWide16", "HCPostNormProjection").replace(
            "partials: array<f16, 512>", "partials: array<f16, 256>")
        self.assertEqual(restored, original_norm)
        for marker in ("sum += dot(", "total += sum;", "let low =", "let high ="):
            self.assertEqual([line for line in original.splitlines() if marker in line],
                             [line for line in result.splitlines() if marker in line])
        self.assertIn("for (var i = 0u; i < 32u; i++) { value += partials[slot * 256u + local_idx + i]; }", result)
        self.assertEqual(result.count('workgroupBarrier();'), original.count('workgroupBarrier();'))

    def test_all_columns_have_unique_writers_and_partial_slots_are_in_bounds(self):
        owners = []
        for group in range(21):
            for slot in range(2):
                for local_idx in range(256):
                    col = group * 16 + slot * 8 + local_idx // 32
                    offset = slot * 256 + local_idx
                    self.assertLess(offset, 512)
                    if local_idx % 32 == 0 and col < 324:
                        owners.append(col)
                        self.assertLess(offset + 31, 512)
        self.assertEqual(sorted(owners), list(range(324)))
        self.assertEqual(len(set(owners)), 324)
        self.assertEqual(STORAGE_BYTES, 28160)

    def test_parent_program_fallback_and_distinct_replay_key_preserved(self):
        result = transform(self.source)
        start = self.source.index("class HCPostNormProjectionProgram final")
        stop = self.source.index("class HCPostNormProjectionF16", start)
        self.assertIn(self.source[start:stop], result)
        self.assertIn(f'"{FLAG}", 0', result)
        self.assertIn(f'"{FLAG} must be 0 or 1"', result)
        self.assertIn("wide_mode == 0 || wide_mode == 1", result)
        self.assertIn("wide_mode == 1 && limits.maxComputeWorkgroupStorageSize >= 28160", result)
        self.assertIn("SetDispatchGroupSize(wide16 ? 21 : 41)", result)
        self.assertIn("}, 0, wide16);", result)
        self.assertIn("!context.EnableMatmulFp32Accumulation()", result)

    def test_window_flag_prefix_and_cli_are_wired(self):
        self.assertEqual(HC_POST_NORM_WIDE16_FLAG, FLAG)
        self.assertIn('HC_POST_NORM_WIDE16_FLAG: "hc_post_norm_wide16"', inspect.getsource(run_qsa_windows))
        benchmark = Path(__file__).with_name("benchmark_webgpu_int4.py").read_text()
        self.assertIn('add_argument("--compare-hc-post-norm-wide16-windows", action="store_true")', benchmark)
        self.assertIn("args.compare_hc_post_norm_wide16_windows or", benchmark)
        self.assertIn('"hc_post_norm_wide16_decode_windows" if args.compare_hc_post_norm_wide16_windows else', benchmark)
        self.assertIn("HC_POST_NORM_WIDE16_FLAG if args.compare_hc_post_norm_wide16_windows else", benchmark)
        self.assertIn(f'"{FLAG}",', benchmark)


if __name__ == "__main__":
    unittest.main()
