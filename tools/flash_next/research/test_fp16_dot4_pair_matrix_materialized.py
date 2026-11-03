from pathlib import Path
import unittest

from build_fp16_dot4_pair_matrix_materialized import CLASS, FLAG, PROGRAM, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


class MaterializedPairTests(unittest.TestCase):
    def test_actual_renderer_materialization(self):
        for rows in (8, 16):
            code = render(PROGRAM, 0, False, False, "", FLAG,
                          class_name=CLASS, constructor_args=(rows,))
            self.assertEqual(code.count("subgroupMatrixMultiplyAccumulate("), 1)
            self.assertEqual(code.count("workgroupBarrier();"), 4)
            for name in ("native_pairs", "guarded_pairs"):
                self.assertIn(f"var<workgroup> {name}: array<vec2<f16>, 16>;", code)
            store = code.index("native_pairs[col] = vec2<f16>(native0, native1)")
            barrier = code.index("workgroupBarrier();", store)
            load = code.index("let rounded_native = native_pairs[col]")
            self.assertLess(store, barrier)
            self.assertLess(barrier, load)
            self.assertIn("rounded_native.x + rounded_native.y", code)
            self.assertIn("rounded_guarded.x + rounded_guarded.y", code)
            self.assertIn("pair0 = fma(xv.w, wv.w, rhs[col])", code)
            self.assertIn("pair1 = fma(xv.z, wv.z, rhs[32u + col])", code)
            self.assertIn("dot(xv, wv)", code)

    def test_default_and_admission_unchanged(self):
        source = SOURCE.read_text()
        candidate = transform(source)
        start = source.index("  const int grouped_prefill_rows", source.index("Status ApplyMatMulNBits("))
        self.assertTrue(candidate.endswith(source[start:]))
        self.assertIn("M == 16 && dispatch_M == M && N == 64 && K == 4", candidate)
        self.assertIn(".SetWorkgroupSize(32).SetDispatchGroupSize(16)", candidate)
        self.assertIn(FLAG + " must be 0 or 1", candidate)
        self.assertIn(CLASS + " program", candidate)
        for rows in (8, 16):
            self.assertIn(f"Type::F16, Type::F32, {rows}, 16, 16, 32", candidate)


if __name__ == "__main__":
    unittest.main()
