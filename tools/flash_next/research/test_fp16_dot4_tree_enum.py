from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

from build_fp16_dot4_tree_enum import COLUMNS, FLAG, PATTERNS, PROGRAM, comparator_transform, transform
from prepare_fp16_dot4_tree_enum_fixtures import PARENT, prepare
from test_hc_projection_inner_unroll import render


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


class Fp16Dot4TreeEnumTests(unittest.TestCase):
    def test_complete_pairings_roundings_and_output_planes(self):
        orders = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))
        patterns = [(tree, first, second, materialized) for materialized in (False, True)
                    for tree in range(3) for first in range(4) for second in range(4)]
        self.assertEqual(len(patterns), PATTERNS)
        addresses = set()
        for p, (tree, first, second, materialized) in enumerate(patterns):
            self.assertEqual(p, tree * 16 + first * 4 + second + int(materialized) * 48)
            self.assertEqual(sorted(orders[tree]), [0, 1, 2, 3])
            for row in range(16):
                for col in range(16):
                    address = row * COLUMNS + p * 16 + col
                    self.assertNotIn(address, addresses)
                    addresses.add(address)
        reference = {row * COLUMNS + PATTERNS * 16 + col for row in range(16) for col in range(16)}
        self.assertFalse(addresses & reference)
        self.assertEqual(addresses | reference, set(range(16 * COLUMNS)))

    def test_actual_renderer_admission_and_materialized_boundaries(self):
        code = render(PROGRAM, 0, False, False, "", FLAG, class_name="Fp16Dot4TreeEnumProgram", constructor_args=())
        self.assertIn("return fma(xv[a], wv[a], p[b]);", code)
        self.assertIn("return fma(xv[b], wv[b], p[a]);", code)
        self.assertIn("pairs[i] = vec2<f16>(left, right)", code)
        self.assertIn("stored.x + stored.y", code)
        self.assertIn("dot(xv, weights[i])", code)
        self.assertIn("pattern + 48u", code)
        self.assertIn("tree < 3u", code)
        self.assertIn("first < 4u", code)
        self.assertIn("second < 4u", code)
        source = SOURCE.read_text()
        candidate = transform(source)
        start = source.index("  const int grouped_prefill_rows", source.index("Status ApplyMatMulNBits("))
        self.assertTrue(candidate.endswith(source[start:]))
        self.assertIn("M == 16 && dispatch_M == M && N == 1552 && K == 4", candidate)

    def test_fixtures_and_comparator_exact_output_shape(self):
        with tempfile.TemporaryDirectory(prefix="dot4-tree-enum-fixtures-") as directory:
            output = Path(directory) / "fixtures"
            prepare(output=output)
            for seed in range(8):
                original = onnx.load(PARENT / f"seed{seed}.onnx")
                candidate = onnx.load(output / f"seed{seed}.onnx")
                old = {v.name: numpy_helper.to_array(v) for v in original.graph.initializer}
                new = {v.name: numpy_helper.to_array(v) for v in candidate.graph.initializer}
                self.assertEqual(new["a"].tobytes(), old["a"].tobytes())
                for name in ("b", "scales"):
                    self.assertTrue(np.array_equal(new[name], np.concatenate([old[name][:16]] * 97)))
                self.assertEqual(candidate.graph.output[0].type.tensor_type.shape.dim[1].dim_value, COLUMNS)
        cpp = comparator_transform((ROOT / "fp16_dot4_tree_compare.cpp").read_text())
        self.assertIn("std::array<size_t, 96>", cpp)
        self.assertEqual(cpp.count("pattern < 96"), 2)
        self.assertIn("{16, 1552}", cpp)
        self.assertIn("row * 1552 + 1536 + col", cpp)
        self.assertIn("round < 4", cpp)


if __name__ == "__main__":
    unittest.main()
