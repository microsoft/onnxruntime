from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

from build_fp16_dot4_tree_probe import FLAG, PROGRAM, transform
from prepare_fp16_dot4_tree_fixtures import PARENT, prepare
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


class Fp16Dot4TreeProbeTests(unittest.TestCase):
    def test_original_paths_and_canonical_admission(self):
        source = SOURCE.read_text()
        candidate = transform(source)
        start = source.index("  const int grouped_prefill_rows", source.index("Status ApplyMatMulNBits("))
        self.assertTrue(candidate.endswith(source[start:]))
        self.assertIn("M == 16 && dispatch_M == M && N == 80 && K == 4", candidate)
        self.assertIn("static const int dot4_tree_probe", candidate)
        self.assertIn(f'"{FLAG} must be 0 or 1"', candidate)
        self.assertNotIn("SelectSubgroupMatrixConfig", candidate[:candidate.index("class GroupedPrefill")])

    def test_actual_renderer_roundtrip_and_output_bounds(self):
        code = render(PROGRAM, 0, False, False, helper="", flag=FLAG,
                      class_name="Fp16Dot4TreeProbeProgram", constructor_args=())
        self.assertEqual(code.count("workgroupBarrier();"), 1)
        self.assertLess(code.index("products[i] = xv * wv"), code.index("workgroupBarrier();"))
        self.assertGreater(code.index("let p = products[i]"), code.index("workgroupBarrier();"))
        self.assertIn("fma(xv.x, wv.x, p.w)", code)
        self.assertIn("fma(xv.z, wv.z, p.y)", code)
        self.assertIn("f16(f32(xv.x) * f32(wv.x) + f32(xv.w) * f32(wv.w))", code)
        self.assertIn("dot(xv, wv)", code)
        for row in range(16):
            for col in range(16):
                self.assertLess(row * 16 + col, 256)
                for slot in range(5):
                    self.assertLess(row * 80 + slot * 16 + col, 1280)

    def test_fixture_duplication_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixtures"
            prepare(output=output)
            for seed in range(8):
                original = onnx.load(PARENT / f"seed{seed}.onnx")
                candidate = onnx.load(output / f"seed{seed}.onnx")
                parent = {v.name: numpy_helper.to_array(v) for v in original.graph.initializer}
                arrays = {v.name: numpy_helper.to_array(v) for v in candidate.graph.initializer}
                np.testing.assert_array_equal(arrays["a"].view(np.uint16),
                                              parent["a"].view(np.uint16))
                for name in ("b", "scales"):
                    for slot in range(5):
                        np.testing.assert_array_equal(arrays[name][slot * 16:(slot + 1) * 16],
                                                      parent[name][:16])
                onnx.checker.check_model(candidate)
            with self.assertRaises(FileExistsError):
                prepare(output=output)


if __name__ == "__main__":
    unittest.main()
