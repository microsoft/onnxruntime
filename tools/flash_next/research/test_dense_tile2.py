from pathlib import Path
import unittest

from build_dense_tile2 import FLAG, private_function, private_program, transform, transform_compare


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-dense-tile4/build-v1/matmul_nbits.cc")


class DenseTile2Tests(unittest.TestCase):
    def test_comparator_preserves_existing_diagnostics_and_requires_new_selector(self):
        source = (SOURCE.parent / "native_compare.cpp").read_text()
        with self.assertRaises(AssertionError):
            transform_compare(source)
        candidate = source.replace("ORT_WEBGPU_DENSE_TILE4", FLAG)
        self.assertEqual(transform_compare(candidate), candidate)

    def test_only_private_tiling_workgroup_and_identity_change(self):
        source = SOURCE.read_text()
        original_start = source.rindex("Status ApplyDenseTile4MatMulNBits(")
        original_stop = source.index("\n}\n\n}  // namespace webgpu", original_start)
        restored = (private_function(source)
                    .replace("ApplyDenseTile2MatMulNBits", "ApplyDenseTile4MatMulNBits")
                    .replace("DenseTile2Program", "DenseTile4Program")
                    .replace("workgroup_size = 64;\n  constexpr uint32_t tile_size = 2;",
                             "workgroup_size = 128;\n  constexpr uint32_t tile_size = 4;"))
        self.assertEqual(restored, source[original_start:original_stop])
        start = source.index("class DenseTile4Program")
        stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
        restored = (private_program(source).replace("DenseTile2Program", "DenseTile4Program")
                    .replace("ApplyDenseTile2MatMulNBits", "ApplyDenseTile4MatMulNBits")
                    .replace("MatMulNBitsTile2F16", "MatMulNBitsTile4F16"))
        self.assertEqual(restored, source[start:stop])

    def test_each_column_retains_32_lanes_and_all_reduction_positions(self):
        for tile, threads in ((4, 128), (2, 64)):
            self.assertEqual(threads // 32, tile)
            slots = {(thread // 32, thread % 32) for thread in range(threads)}
            self.assertEqual(slots, {(column, lane) for column in range(tile) for lane in range(32)})
            for n in (1, 2, 3, 324, 2560, 7168):
                columns = [group * tile + column
                           for group in range((n + tile - 1) // tile)
                           for column in range(tile) if group * tile + column < n]
                self.assertEqual(columns, list(range(n)))

    def test_all_parent_programs_and_selector_fallbacks_preserved(self):
        source = SOURCE.read_text()
        actual = transform(source)
        start = source.index("class DenseTile4Program")
        stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
        self.assertIn(source[start:stop], actual)
        self.assertIn(f'"{FLAG}", 0', actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("dense_bitcast_route ? (dense_tile2 == 1 ? 3 : 1 + dense_tile4) : 0", actual)
        self.assertLess(actual.index("if (dense_bitcast_route && dense_tile2 == 1)"),
                        actual.index("if (dense_bitcast_route && dense_tile4 == 1)"))


if __name__ == "__main__":
    unittest.main()
