from pathlib import Path
import unittest

from build_dense_qkv_pad34 import FLAG, PROGRAM, transform, transform_template


REPOS = Path("/home/sayanshaw/repos")
SOURCE = REPOS / "qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc"
TEMPLATE = REPOS / "qwen38-int4-dense-block32/block32-template-v1/dense_block32_template.inc"


class DenseQkvPad34Tests(unittest.TestCase):
    def test_template_only_pads_storage_not_logical_math(self):
        original = TEMPLATE.read_text()
        actual = transform_template(original)
        restored = actual.replace("ApplyDenseQkvPad34Template(", "ApplyDenseBlock32Template(")
        restored = restored.replace("(__param_tile_size_k_vec + 2u)", "__param_tile_size_k_vec")
        self.assertEqual(restored, original)
        self.assertEqual(actual.count("(__param_tile_size_k_vec + 2u)"), 1)

    def test_four_writer_lanes_address_distinct_four_byte_banks(self):
        for k_lane in range(32):
            original = [((column * 32 + k_lane) * 2 // 4) % 32 for column in range(4)]
            padded = [((column * 34 + k_lane) * 2 // 4) % 32 for column in range(4)]
            self.assertEqual(len(set(original)), 2)
            self.assertEqual(len(set(padded)), 4)
        slots = [column * 34 + lane for column in range(4) for lane in range(32)]
        self.assertEqual(len(set(slots)), 128)
        self.assertLess(max(slots), 4 * 34)
        self.assertEqual(2048 + 4 * 34 * 2, 2320)

    def test_original_function_and_geometry_are_preserved(self):
        original = SOURCE.read_text()
        actual = transform(original)
        start = original.rindex("Status ApplyDenseBlock32MatMulNBits(")
        end = original.index("\n}\n\n}  // namespace webgpu", start)
        self.assertIn(original[start:end], actual)
        self.assertIn(f'Program{{"{PROGRAM}"}}', actual)
        self.assertIn(f"{FLAG} must be 0 or 1", actual)
        self.assertIn("K_ == 2560 && N_ == 16384", actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 2320", actual)
        self.assertIn("qkv_pad34_route ? 22", actual)
        self.assertIn("!context.EnableMatmulFp32Accumulation()", actual)


if __name__ == "__main__":
    unittest.main()
