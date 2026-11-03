from pathlib import Path
import unittest

from build_dense_preload import FLAG, preload_template, private_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-dense-tile4/build-v1/matmul_nbits.cc")
TEMPLATE = Path("/home/sayanshaw/repos/qwen38-int4-dense-bitcast-unpack/build-v1/dense_bitcast_template.inc")


class DensePreloadTests(unittest.TestCase):
    def test_dot_products_accumulation_and_serial_reduction_preserved(self):
        source = TEMPLATE.read_text()
        actual = preload_template(source)
        expressions = [line for line in source.splitlines()
                       if line.startswith('ss << "') and any(token in line for token in
                           ("sum +=", "inter_results[", "output_value +=", "decode_low_nibbles"))]
        for expression in expressions:
            self.assertIn(expression, actual)
        self.assertEqual(actual.count('ss << "    workgroupBarrier();\\n";'), 2)
        self.assertLess(actual.index("loadSHMA(batch, a_row, 0u, id)"), actual.index("for (var kidx"))
        self.assertIn('var a_offset = kidx / " << __param_component_a', actual)

    def test_shared_preload_addresses_equal_original_global_addresses(self):
        for k in (640, 2560, 6144, 10240):
            self.assertLessEqual(k * 2 + 256, 27648)
            for tile_start in range(0, k, 1024):
                for lane in range(32):
                    for offset in range(8):
                        address = tile_start // 4 + lane * 8 + offset
                        if address < k // 4:
                            self.assertEqual(address * 4, tile_start + lane * 32 + offset * 4)

    def test_original_shaders_storage_admission_replay_and_program_cache(self):
        source = SOURCE.read_text()
        actual = transform(source)
        start = source.index("class DenseTile4Program")
        stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
        self.assertIn(source[start:stop], actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("(storage_limit - 256) / sizeof(MLFloat16)", actual)
        self.assertIn("dense_bitcast_route ? (preload ? 3 : 1 + dense_tile4) : 0", actual)
        self.assertIn("broadcast_a, acc_f32, K);", actual)
        program = private_program(source)
        self.assertIn("a_length_per_tile = full_k_ / components_a", program)
        self.assertIn("full_k_(full_k)", program)


if __name__ == "__main__":
    unittest.main()
