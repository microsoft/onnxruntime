from pathlib import Path
import unittest

from build_dense_paired_matrix import CLASS, FLAG, PROGRAM, STORAGE_BYTES, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


class PackedMatrixTests(unittest.TestCase):
    def test_actual_renderer_and_ordered_rounding(self):
        code = render(PROGRAM, 0, False, False, "", FLAG, class_name=CLASS, constructor_args=())
        self.assertEqual(code.count("subgroupMatrixMultiplyAccumulate("), 1)
        self.assertEqual(code.count("workgroupBarrier();"), 5)
        self.assertIn("var<workgroup> result: array<f32, 1024>", code)
        self.assertIn("subgroup_matrix_result<f32, 16, 16>", code)
        self.assertLess(code.index("first[offset] = xv.x * wv.x"),
                        code.index("result[base + i] = f32(first[base + i])"))
        self.assertIn("sums[j] += dots[j] + dot_value", code)
        self.assertIn("totals[j] += sums[j]", code)
        self.assertIn("value += partials[local_idx * 32u + stream]", code)
        self.assertNotIn("dot(", code)
        self.assertEqual(STORAGE_BYTES, 3 * 1024 * 2 + 1024 * 4 + 512 * 2)

    def test_mapping_covers_all_streams_columns_and_matrix_rows(self):
        outputs = set()
        for sg in range(4):
            cells = set()
            for lane in range(32):
                for j in range(4):
                    stream, col = divmod(lane + j * 32, 16)
                    outputs.add((sg * 8 + stream, col))
                    cells.update(((stream * 2, col), (stream * 2 + 1, col)))
            self.assertEqual(cells, {(row, col) for row in range(16) for col in range(16)})
        self.assertEqual(outputs, {(stream, col) for stream in range(32) for col in range(16)})

    def test_admission_default_fallback_replay(self):
        original = SOURCE.read_text()
        candidate = transform(original)
        self.assertIn("static const int paired_matrix_mode", candidate)
        self.assertIn(FLAG + " must be 0 or 1", candidate)
        self.assertIn("paired_matrix_mode == 1 && dense_bitcast_route && block32_mode == 1", candidate)
        self.assertIn("Type::F16, Type::F32, 16, 16, 16, 32, false", candidate)
        self.assertIn(".SetWorkgroupSize(128).SetDispatchGroupSize(N_ / 16)", candidate)
        self.assertIn("paired_matrix_config.has_value() ? 4", candidate)
        self.assertTrue(candidate.endswith(original[original.index("\nStatus ApplyMatMulNBits("):]))


if __name__ == "__main__":
    unittest.main()
