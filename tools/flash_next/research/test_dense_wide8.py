from pathlib import Path
import unittest

from build_dense_wide8 import FLAG, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-dense-tile4/build-v1/matmul_nbits.cc")


class DenseWide8Tests(unittest.TestCase):
    def test_only_private_column_count_workgroup_and_identity_change(self):
        source = SOURCE.read_text()
        actual = transform(source)
        start = actual.rindex("Status ApplyDenseWide8MatMulNBits(")
        stop = actual.index("\n}\n\n}  // namespace webgpu", start)
        restored = (actual[start:stop]
                    .replace("ApplyDenseWide8MatMulNBits", "ApplyDenseTile4MatMulNBits")
                    .replace("DenseWide8Program", "DenseTile4Program")
                    .replace("workgroup_size = 256;\n  constexpr uint32_t tile_size = 8;",
                             "workgroup_size = 128;\n  constexpr uint32_t tile_size = 4;"))
        start = source.rindex("Status ApplyDenseTile4MatMulNBits(")
        stop = source.index("\n}\n\n}  // namespace webgpu", start)
        self.assertEqual(restored, source[start:stop])
        self.assertIn(source[start:stop], actual)

    def test_32_lanes_per_column_limits_and_fallback_replay(self):
        actual = transform(SOURCE.read_text())
        self.assertEqual(256 // 32, 8)
        for expected in (
                f'"{FLAG}", 0', f'"{FLAG} must be 0 or 1"',
                "maxComputeWorkgroupSizeX >= 256", "maxComputeInvocationsPerWorkgroup >= 256",
                "if (dense_bitcast_route && use_dense_wide8)",
                "dense_bitcast_route ? (use_dense_wide8 ? 3 : 1 + dense_tile4) : 0"):
            self.assertIn(expected, actual)
        self.assertNotIn("DENSE_TILE2", actual)
        self.assertNotIn("dense_tile2", actual)
        self.assertIn('Program{"MatMulNBitsTile4F16"}', actual)
        self.assertIn('Program{"MatMulNBitsBitcastF16"}', actual)
        self.assertIn('Program{"MatMulNBitsWide8F16"}', actual)


if __name__ == "__main__":
    unittest.main()
