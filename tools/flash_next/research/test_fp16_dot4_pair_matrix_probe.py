from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

from build_fp16_dot4_pair_matrix_probe import FLAG, PROGRAM, transform
from prepare_fp16_dot4_pair_matrix_fixtures import PARENT, prepare
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


class Fp16Dot4PairMatrixTests(unittest.TestCase):
    def test_native_pairing_and_correction_renderer(self):
        for rows in (8, 16):
            code = render(PROGRAM, 0, False, False, "", FLAG,
                          class_name="Fp16Dot4PairMatrixProbeProgram", constructor_args=(rows,))
            self.assertEqual(code.count("subgroupMatrixMultiplyAccumulate("), 1)
            self.assertEqual(code.count("workgroupBarrier();"), 3)
            self.assertIn("if (k == 0u) { value = xv.x * wv.x; }", code)
            self.assertIn("if (k == 1u) { value = wv.w; }", code)
            self.assertIn("if (k == 2u) { value = xv.y * wv.y; }", code)
            self.assertIn("if (k == 3u) { value = wv.z; }", code)
            self.assertIn("pair0 = fma(xv.w, wv.w, rhs[col])", code)
            self.assertIn("pair1 = fma(xv.z, wv.z, rhs[32u + col])", code)
            self.assertIn("subgroup_matrix_result<f32, 16, kRows>", code)
            self.assertIn("0.00006103515625", code)
            self.assertIn("remainder >= 0x0fffu && remainder <= 0x1001u", code)
            self.assertIn("f16(mask)", code)

    def test_diagnostic_bounds_default_and_config(self):
        source = SOURCE.read_text()
        candidate = transform(source)
        start = source.index("  const int grouped_prefill_rows", source.index("Status ApplyMatMulNBits("))
        self.assertTrue(candidate.endswith(source[start:]))
        self.assertIn("M == 16 && dispatch_M == M && N == 64 && K == 4", candidate)
        self.assertIn(".SetWorkgroupSize(32).SetDispatchGroupSize(16)", candidate)
        for rows in (8, 16):
            self.assertIn(f"Type::F16, Type::F32, {rows}, 16, 16, 32", candidate)
        offsets = [row * 64 + plane * 16 + col for row in range(16)
                   for plane in range(4) for col in range(16)]
        self.assertEqual(sorted(offsets), list(range(1024)))
        for mask in range(64):
            correct = mask & 3
            reasons = ((mask >> 2) | (mask >> 4)) & 3
            valid = all(bool(mask & (1 << pair)) == bool(mask & ((4 | 16) << pair)) for pair in (0, 1))
            self.assertEqual(correct == reasons, valid)

    def test_fixtures_preserve_parent_bytes(self):
        with tempfile.TemporaryDirectory(prefix="dot4-pair-matrix-fixtures-") as directory:
            output = Path(directory) / "fixtures"
            prepare(output=output)
            for seed in range(8):
                old = {v.name: numpy_helper.to_array(v) for v in onnx.load(PARENT / f"seed{seed}.onnx").graph.initializer}
                model = onnx.load(output / f"seed{seed}.onnx")
                new = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
                self.assertEqual(old["a"].tobytes(), new["a"].tobytes())
                for name in ("b", "scales"):
                    self.assertTrue(np.array_equal(new[name], np.concatenate([old[name][:16]] * 4)))
                self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[1].dim_value, 64)


if __name__ == "__main__":
    unittest.main()
