from pathlib import Path
import unittest

from build_hc_narrow4 import FLAG, narrow_program, transform


class Narrow4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/"
                          "build-v2/hc_post_norm_projection.cc").read_text()

    def test_virtual_partials_and_original_order_are_bijective(self):
        partials = [thread + part * 128 for thread in range(128) for part in range(2)]
        self.assertEqual(sorted(partials), list(range(256)))
        visited = []
        for partial in partials:
            indices = list(range(partial, 2560, 256))
            self.assertEqual(indices, [partial + tile * 256 for tile in range(10)])
            visited.extend(indices)
        self.assertEqual(sorted(visited), list(range(2560)))
        for width in (128, 64, 32, 16, 8, 4, 2, 1):
            self.assertEqual([(t, t + width) for t in range(128) if t < width],
                             [(t, t + width) for t in range(256) if t < width])

    def test_projection_and_post_stores_have_unique_complete_ownership(self):
        columns = [group * 4 + thread // 32 for group in range(81)
                   for thread in range(128) if thread % 32 == 0]
        self.assertEqual(columns, list(range(324)))
        self.assertEqual(sorted(h for thread in range(128) for h in range(thread, 2560, 128)),
                         list(range(2560)))
        for thread in range(0, 128, 32):
            self.assertLess(thread + 31, 128)

    def test_arithmetic_tree_and_single_command_original_route_preserved(self):
        program = narrow_program(self.source)
        start = self.source.index("class HCPostNormProjectionProgram final")
        stop = self.source.index("class HCPostNormProjectionF16", start)
        original = self.source[start:stop]
        for marker in ("sum +=", "total += sum;", "workgroupBarrier();", "inverseSqrt",
                       "sum += dot", "stable_sigmoid(f32(z_t))", "value += partials"):
            self.assertEqual([line for line in original.splitlines() if marker in line],
                             [line for line in program.splitlines() if marker in line])
        self.assertIn("virtual_idx += 128u", program)
        self.assertIn("var h = virtual_idx; h < uniforms.hidden; h += 256u", program)
        self.assertIn("partials: array<f16, 128>", program)
        result = transform(self.source)
        self.assertIn(f"{FLAG} must be 0 or 1", result)
        self.assertIn("}, 0, narrow4);", result)
        self.assertIn(".SetWorkgroupSize(narrow4 == 1 ? 128 : 256)", result)
        self.assertIn("HCPostNormProjectionProgram program{shared_, shared_scale}", result)
        self.assertIn(original, result)

    def test_benchmark_and_native_admission_are_wired(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        for marker in ("--compare-hc-narrow4-windows", "args.compare_hc_narrow4_windows or",
                       '"hc_narrow4_decode_windows" if args.compare_hc_narrow4_windows',
                       "HC_NARROW4_FLAG if args.compare_hc_narrow4_windows", f'"{FLAG}",'):
            self.assertIn(marker, benchmark)
        self.assertIn('HC_NARROW4_FLAG: "hc_narrow4"', (root / "qsa_rank_decode_windows.py").read_text())
        self.assertIn('"hc-narrow4-v1"', (root / "run_hc_up_parallel2_model.py").read_text())


if __name__ == "__main__":
    unittest.main()
