from pathlib import Path
import unittest

from build_fp16_dot4_matrix_f32_probe import FLAG, PROGRAM, comparator_transform, program_transform, transform
from build_fp16_dot4_matrix_probe import PROGRAM as ORIGINAL_PROGRAM
from test_hc_projection_inner_unroll import render


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


class Fp16Dot4MatrixF32Tests(unittest.TestCase):
    def test_actual_shader_changes_only_result_precision(self):
        for rows in (8, 16):
            original = render(ORIGINAL_PROGRAM, 0, False, False, "", FLAG,
                              class_name="Fp16Dot4MatrixProbeProgram", constructor_args=(rows,))
            candidate = render(PROGRAM, 0, False, False, "", FLAG,
                              class_name="Fp16Dot4MatrixF32ProbeProgram", constructor_args=(rows,))
            expected = original.replace("result: array<f16, 256>", "result: array<f32, 256>").replace(
                "subgroup_matrix_result<f16, 16, kRows>", "subgroup_matrix_result<f32, 16, kRows>").replace(
                "= result[i]", "= f16(result[i])")
            self.assertEqual(candidate, expected)
            self.assertEqual(candidate.count("subgroupMatrixMultiplyAccumulate("), 1)
            self.assertIn("lhs: array<f16, 256>", candidate)
            self.assertIn("rhs: array<f16, 256>", candidate)
            self.assertIn("dot(xv, wv)", candidate)

    def test_diagnostic_only_and_typed_config_admission(self):
        source = SOURCE.read_text()
        candidate = transform(source)
        start = source.index("  const int grouped_prefill_rows", source.index("Status ApplyMatMulNBits("))
        self.assertTrue(candidate.endswith(source[start:]))
        self.assertIn("M == 16 && dispatch_M == M && N == 32 && K == 4", candidate)
        for rows in (8, 16):
            self.assertIn(f"Type::F16, Type::F32, {rows}, 16, 16, 32", candidate)
        self.assertNotIn("Type::F16, Type::F16, 16, 16, 16, 32", candidate)
        self.assertIn(f'"{FLAG} must be 0 or 1"', candidate)
        self.assertEqual(program_transform(ORIGINAL_PROGRAM), PROGRAM)

    def test_comparator_reports_complete_mismatch_partition(self):
        source = comparator_transform((ROOT / "fp16_dot4_matrix_compare.cpp").read_text())
        self.assertIn("matrix.ToFloat() == 0.0f && scalar.ToFloat() == 0.0f", source)
        self.assertIn("++zero_sign_mismatches", source)
        self.assertIn("++nonzero_mismatches", source)
        self.assertIn("if (mismatches != 0) return 2;", source)
        self.assertIn("!std::isfinite(matrix.ToFloat())", source)


if __name__ == "__main__":
    unittest.main()
