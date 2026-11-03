from pathlib import Path
import unittest

from build_hc_post_norm_vec4 import transform as vector_transform
from build_hc_quad_fold import FLAG, quad_program, transform


class QuadFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/"
                          "build-v2/hc_post_norm_projection.cc").read_text()

    def test_shared_factory_preserves_existing_built_vector_source(self):
        previous = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-vec4/"
                        "build-v1/hc_post_norm_projection.cc").read_text()
        self.assertEqual(vector_transform(self.source), previous)

    def test_per_column_partial_order_and_unique_output_ownership(self):
        columns = []
        for group in range(41):
            for thread in range(2):
                for component in range(4):
                    column = group * 8 + thread * 4 + component
                    indices = [thread * 128 + lane + component * 32 for lane in range(32)]
                    self.assertEqual(indices, list(range((thread * 4 + component) * 32,
                                                          (thread * 4 + component + 1) * 32)))
                    self.assertLess(max(indices), 256)
                    if column < 324:
                        columns.append(column)
        self.assertEqual(columns, list(range(324)))

    def test_normalization_dot_barriers_and_activation_expressions_preserved(self):
        program = quad_program(self.source)
        for marker in ("sum +=", "total += sum;", "workgroupBarrier();", "inverseSqrt",
                       "sum += dot", "stable_sigmoid(f32(z_t))", "activated.SetByOffset", "tail.SetByOffset"):
            start = self.source.index("class HCPostNormProjectionProgram final")
            stop = self.source.index("class HCPostNormProjectionF16", start)
            original = self.source[start:stop]
            self.assertEqual([line for line in original.splitlines() if marker in line],
                             [line for line in program.splitlines() if marker in line])
        self.assertIn("for (var i = 0u; i < 32u; i++)", program)
        self.assertIn("var folded = vec4<f16>(0)", program)
        result = transform(self.source)
        self.assertIn(f"{FLAG} must be 0 or 1", result)
        self.assertIn("}, 0, quad_fold);", result)
        self.assertIn("HCPostNormProjectionProgram program{shared_, shared_scale}", result)

    def test_benchmark_cli_metadata_and_native_gate_wired(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-hc-quad-fold-windows", benchmark)
        self.assertIn("args.compare_hc_quad_fold_windows or", benchmark)
        self.assertIn('"hc_quad_fold_decode_windows" if args.compare_hc_quad_fold_windows', benchmark)
        self.assertIn("HC_QUAD_FOLD_FLAG if args.compare_hc_quad_fold_windows", benchmark)
        self.assertIn(f'"{FLAG}",', benchmark)
        self.assertIn('HC_QUAD_FOLD_FLAG: "hc_quad_fold"',
                      (root / "qsa_rank_decode_windows.py").read_text())
        self.assertIn('"hc-quad-fold-v1"', (root / "run_hc_up_parallel2_model.py").read_text())


if __name__ == "__main__":
    unittest.main()
