from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

from build_fp16_dot4_sweep_compare import transform
from prepare_fp16_dot4_accumulator_sweep import EDGE, OVERFLOW_EDGE, PARENT, bounded_scales, finite_bits, save_fixture


class AccumulatorSweepTests(unittest.TestCase):
    def test_batched_comparator_contract(self):
        source = transform((Path(__file__).parent / "fp16_dot4_cross_shader_compare.cpp").read_text())
        self.assertIn("argc != 5", source)
        self.assertIn("reference(fixtures.size() * 256)", source)
        self.assertIn("fixtures[file].c_str()", source)
        self.assertNotIn("argv[file]", source)
        self.assertNotIn("file - 4", source)
        self.assertIn("for (int pattern : {47, 95})", source)
        self.assertEqual(125 * 8 * 256 * 2, 512000)
        self.assertEqual(125 * 8 * 256 * 4, 1024000)

    def test_complete_finite_bit_space(self):
        bits = finite_bits()
        self.assertEqual(bits.size, 63488)
        self.assertEqual(np.unique(bits).size, 63488)
        self.assertTrue(np.all(np.isfinite(bits.view(np.float16))))
        self.assertEqual(bits.size // (8 * 16 * 4), 124)
        self.assertTrue(0 in bits and 0x8000 in bits and 0x7bff in bits and 0xfbff in bits)

    def test_coefficient_bound_and_overflow_edges(self):
        scales = bounded_scales(np.array([0, 0.1964, 2, 65504], dtype=np.float16), 1 / 128)
        self.assertLessEqual(float(np.max(scales)), 1 / 128)
        self.assertLess(4 * 65504 * 8 / 128, 65504)
        self.assertEqual(EDGE.shape, (16, 4))
        self.assertEqual(OVERFLOW_EDGE.shape, (16, 4))
        pairs = []
        for x in OVERFLOW_EDGE.astype(np.float64):
            p = (x * -0.5).astype(np.float16)
            q0 = np.float16(x[3] * -0.5 + float(p[0]))
            q1 = np.float16(x[2] * -0.5 + float(p[1]))
            self.assertTrue(np.isfinite(q0) and np.isfinite(q1) and np.isfinite(np.float16(q0 + q1)))
            pairs.extend((abs(float(q0)), abs(float(q1))))
        self.assertIn(65504, pairs)

    def test_reference_and_candidate_share_operands(self):
        template = onnx.load(PARENT / "seed0.onnx")
        values = {v.name: numpy_helper.to_array(v) for v in template.graph.initializer}
        with tempfile.TemporaryDirectory(prefix="accumulator-sweep-test-") as directory:
            models = []
            for columns in (1552, 160):
                path = Path(directory) / f"n{columns}.onnx"
                save_fixture(template, EDGE, values["b"][:16], values["scales"][:16], columns, path)
                model = onnx.load(path)
                self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[1].dim_value, columns)
                models.append({v.name: numpy_helper.to_array(v) for v in model.graph.initializer})
            self.assertEqual(models[0]["a"].tobytes(), models[1]["a"].tobytes())
            for name in ("b", "scales"):
                self.assertTrue(np.array_equal(models[0][name][:16], models[1][name][:16]))


if __name__ == "__main__":
    unittest.main()
