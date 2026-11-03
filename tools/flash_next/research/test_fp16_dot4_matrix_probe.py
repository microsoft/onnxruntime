from pathlib import Path
import unittest

from build_fp16_dot4_matrix_probe import FLAG, PROGRAM, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


class Fp16Dot4MatrixProbeTests(unittest.TestCase):
    def test_existing_paths_and_diagnostic_admission(self):
        source = SOURCE.read_text()
        candidate = transform(source)
        start = source.index("  const int grouped_prefill_rows", source.index("Status ApplyMatMulNBits("))
        self.assertTrue(candidate.endswith(source[start:]))
        self.assertIn("M == 16 && dispatch_M == M && N == 32 && K == 4", candidate)
        self.assertIn("Type::F16, Type::F16, 16, 16, 16, 32", candidate)
        self.assertIn("Type::F16, Type::F16, 8, 16, 16, 32", candidate)
        self.assertIn(f'"{FLAG} must be 0 or 1"', candidate)

    def test_real_shader_emitter_bounds_and_dot4_operands(self):
        for rows in (8, 16):
            code = render(PROGRAM, 0, False, False, helper="", flag=FLAG,
                          class_name="Fp16Dot4MatrixProbeProgram", constructor_args=(rows,))
            self.assertIn(f"const kRows: u32 = {rows}u;", code)
            self.assertEqual(code.count("workgroupBarrier();"), 2)
            self.assertIn("if (k < 4u)", code)
            self.assertIn("subgroupMatrixMultiplyAccumulate", code)
            self.assertIn("dot(xv, wv)", code)
            self.assertIn("rhs[48u + col]", code)
            for row in range(rows):
                for col in range(16):
                    self.assertLess(row * 16 + col, 256)
                    self.assertLess(row * 32 + 16 + col, rows * 32)


if __name__ == "__main__":
    unittest.main()
