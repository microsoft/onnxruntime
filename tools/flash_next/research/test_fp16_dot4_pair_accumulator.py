from pathlib import Path
import unittest

from build_fp16_dot4_pair_accumulator import CLASS, FLAG, PROGRAM, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


class AccumulatorTests(unittest.TestCase):
    def test_actual_matrix_load_and_pair_mapping(self):
        for rows in (8, 16):
            code = render(PROGRAM, 0, False, False, "", FLAG,
                          class_name=CLASS, constructor_args=(rows,))
            self.assertEqual(code.count("subgroupMatrixMultiplyAccumulate("), 1)
            self.assertEqual(code.count("workgroupBarrier();"), 5)
            self.assertIn("subgroupMatrixLoad<subgroup_matrix_result<f32, 16, kRows>", code)
            self.assertIn("if (row == 0u) { bias = f32(rhs[col]); }", code)
            self.assertIn("if (row == 1u) { bias = f32(rhs[32u + col]); }", code)
            self.assertIn("if (row == 0u && k == 0u) { value = xv.w; }", code)
            self.assertIn("if (row == 1u && k == 2u) { value = xv.z; }", code)
            self.assertIn("if (k == 0u) { value = wv.w; }", code)
            self.assertIn("if (k == 2u) { value = wv.z; }", code)
            self.assertIn("row_major>(&matrix_weights, 0u, 16u)", code)
            self.assertLess(code.index("rhs[i] = value"), code.index("result[i] = bias"))
            self.assertLess(code.index("result[i] = bias"), code.index("subgroupMatrixLoad<subgroup_matrix_result"))
            self.assertIn("scalar_tree.x + scalar_tree.y", code)
            self.assertIn("pair0 = fma(xv.w, wv.w, rhs[col])", code)

    def test_default_admission_and_native_m16(self):
        source = SOURCE.read_text()
        result = transform(source)
        start = source.index("  const int grouped_prefill_rows", source.index("Status ApplyMatMulNBits("))
        self.assertTrue(result.endswith(source[start:]))
        self.assertIn("N == 160 && K == 4", result)
        self.assertIn("Type::F16, Type::F32, 16, 16, 16, 32, false", result)
        self.assertNotIn("Type::F16, Type::F32, 8, 16, 16, 32, false", result)
        self.assertIn(FLAG + " must be 0 or 1", result)
        self.assertIn(CLASS + " program", result)


if __name__ == "__main__":
    unittest.main()
