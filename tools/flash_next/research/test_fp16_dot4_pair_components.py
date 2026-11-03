from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

from build_fp16_dot4_pair_components import CLASS, FLAG, PROGRAM, comparator, transform
from prepare_fp16_dot4_pair_matrix_fixtures import PARENT
from run_fp16_dot4_pair_components import prepare_components
from test_hc_projection_inner_unroll import render


ROOT = Path(__file__).parent


class PairComponentTests(unittest.TestCase):
    def test_actual_shader(self):
        for rows in (8, 16):
            code = render(PROGRAM, 0, False, False, "", FLAG,
                          class_name=CLASS, constructor_args=(rows,))
            self.assertEqual(code.count("subgroupMatrixMultiplyAccumulate("), 1)
            self.assertEqual(code.count("workgroupBarrier();"), 4)
            self.assertIn("var<workgroup> scalar_pairs: array<vec2<f16>, 16>", code)
            self.assertIn("f16(fma(f32(xv.z), f32(wv.z), f32(rhs[32u + col])))", code)
            for pair in ("scalar_pairs", "fp32_pairs"):
                self.assertLess(code.index(pair + "[col] ="), code.index("let scalar_tree"))
            self.assertIn("scalar_tree.x + scalar_tree.y", code)
            self.assertIn("fp32_tree.x + fp32_tree.y", code)

    def test_bounds_and_default(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc").read_text()
        candidate = transform(source)
        start = source.index("  const int grouped_prefill_rows", source.index("Status ApplyMatMulNBits("))
        self.assertTrue(candidate.endswith(source[start:]))
        self.assertIn("N == 160 && K == 4", candidate)
        self.assertEqual(sorted(row * 160 + plane * 16 + col for row in range(16)
                                for plane in range(10) for col in range(16)), list(range(2560)))

    def test_comparator_components(self):
        source = comparator((ROOT / "fp16_dot4_pair_matrix_compare.cpp").read_text())
        self.assertIn("{16, 160}", source)
        self.assertIn("return sum_mismatches[0] == 0 ? 0 : 2;", source)
        self.assertIn("COMPONENT pair=", source)
        self.assertIn("TREE kind=", source)
        self.assertNotIn("<< '\n';", source)

    def test_fixture_bytes(self):
        with tempfile.TemporaryDirectory(prefix="pair-component-fixtures-") as directory:
            output = Path(directory) / "fixtures"
            prepare_components(output)
            for seed in range(8):
                old = {v.name: numpy_helper.to_array(v) for v in onnx.load(PARENT / f"seed{seed}.onnx").graph.initializer}
                model = onnx.load(output / f"seed{seed}.onnx")
                new = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
                self.assertEqual(old["a"].tobytes(), new["a"].tobytes())
                for name in ("b", "scales"):
                    self.assertTrue(np.array_equal(new[name], np.concatenate([old[name][:16]] * 10)))
                self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[1].dim_value, 160)


if __name__ == "__main__":
    unittest.main()
