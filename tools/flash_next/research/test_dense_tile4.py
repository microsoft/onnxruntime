from pathlib import Path
import unittest

from build_dense_tile4 import FLAG, private_function, private_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-dense-bitcast-unpack/build-v1/matmul_nbits.cc")


class DenseTile4Tests(unittest.TestCase):
    def test_only_private_column_grouping_changes(self):
        source = SOURCE.read_text()
        start = source.rindex("Status ApplyDenseBitcastMatMulNBits(")
        stop = source.index("\n}\n\n}  // namespace webgpu", start)
        restored = (private_function(source).replace("ApplyDenseTile4MatMulNBits", "ApplyDenseBitcastMatMulNBits")
                    .replace("DenseTile4Program", "DenseBitcastProgram")
                    .replace("constexpr uint32_t tile_size = 4;", "constexpr uint32_t tile_size = 8;"))
        self.assertEqual(restored, source[start:stop])
        self.assertIn("constexpr uint32_t workgroup_size = 128;", private_function(source))
        start = source.index("class DenseBitcastProgram")
        stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
        restored = (private_program(source).replace("DenseTile4Program", "DenseBitcastProgram")
                    .replace("ApplyDenseTile4MatMulNBits", "ApplyDenseBitcastMatMulNBits")
                    .replace("MatMulNBitsTile4F16", "MatMulNBitsBitcastF16"))
        self.assertEqual(restored, source[start:stop])

    def test_original_shared_function_and_private_family_preserved(self):
        source = SOURCE.read_text()
        actual = transform(source)
        start = source.index("Status ApplyMatMulNBits(")
        stop = source.index("\nnamespace {\nStatus ApplyDenseBitcastMatMulNBits", start)
        self.assertIn(source[start:stop], actual)
        start = source.rindex("Status ApplyDenseBitcastMatMulNBits(")
        stop = source.index("\n}\n\n}  // namespace webgpu", start)
        self.assertIn(source[start:stop], actual)
        self.assertIn(f'"{FLAG}", 0', actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("if (dense_bitcast_route && dense_tile4 == 1)", actual)
        self.assertIn("compute, 0, dense_bitcast_route ? 1 + dense_tile4 : 0);", actual)
        self.assertEqual(source[source.index("  const bool dense_bitcast_route ="):
                                source.index("  auto compute = [&]()")],
                         actual[actual.index("  const bool dense_bitcast_route ="):
                                actual.index("  auto compute = [&]()")])


if __name__ == "__main__":
    unittest.main()
