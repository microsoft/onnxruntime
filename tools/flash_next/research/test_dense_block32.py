from pathlib import Path
import unittest

from build_dense_block32 import block32_template, transform


BASE = Path("/home/sayanshaw/repos")


class DenseBlock32Tests(unittest.TestCase):
    def test_only_integer_indexing_and_function_identity_change(self):
        source = (BASE / "qwen38-int4-dense-bitcast-unpack/build-v1/dense_bitcast_template.inc").read_text()
        actual = block32_template(source)
        expected = source.replace("ApplyDenseBitcastTemplate", "ApplyDenseBlock32Template").replace(
            'ss << ") / uniforms.block_size;\\n";', 'ss << ") / 32u;\\n";')
        expected = expected.replace(
            'ss << "  let batch = workgroup_idx / (uniforms.dispatch_M * uniforms.num_N_tile);\\n";',
            'ss << "  let batch = 0u;\\n";').replace(
            'ss << "  let a_global = (workgroup_idx / uniforms.num_N_tile) % uniforms.dispatch_M;\\n";',
            'ss << "  let a_global = 0u;\\n";').replace(
            'ss << "  let b_global_base = (workgroup_idx % uniforms.num_N_tile) * ";',
            'ss << "  let b_global_base = workgroup_idx * ";')
        self.assertEqual(actual, expected)
        for k in (640, 2560, 6144, 10240):
            for tile in range(0, k, 1024):
                for lane in range(32):
                    self.assertEqual((tile + lane * 8) // 32, (tile + lane * 8) >> 5)
        for n in (1, 320, 2560, 6144, 7168, 10240, 16384):
            tiles = (n + 3) // 4
            for group in range(tiles):
                self.assertEqual(group // tiles, 0)
                self.assertEqual((group // tiles) % 1, 0)
                self.assertEqual((group % tiles) * 4, group * 4)

    def test_original_program_and_canonical_route_are_preserved(self):
        source = (BASE / "qwen38-int4-dense-tile4/build-v1/matmul_nbits.cc").read_text()
        actual = transform(source)
        start = source.index("class DenseTile4Program")
        stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
        self.assertIn(source[start:stop], actual)
        route = source.split("const bool dense_bitcast_route =", 1)[1].split("auto compute =", 1)[0]
        self.assertIn(route, actual)
        self.assertIn("block_size_ == 32", route)
        self.assertIn("dense_bitcast_route && block32_mode == 1", actual)
        self.assertIn("block32_mode == 1 ? 3 : 1 + dense_tile4", actual)


if __name__ == "__main__":
    unittest.main()
