from pathlib import Path
import unittest

from build_small_m_prefill import transform


ROOT = Path(__file__).parent


class SmallMPrefillTests(unittest.TestCase):
    def test_prepacked_m16_tiles_preserve_every_input(self):
        for k in (32, 320, 2560, 10240):
            indices = [((i % 256) // 16) * k + (i // 256) * 16 + i % 16
                       for i in range(16 * k)]
            self.assertEqual(sorted(indices), list(range(16 * k)))
            for block in range(0, k, 16):
                for row in range(16):
                    for col in range(16):
                        self.assertEqual(indices[block * 16 + row * 16 + col],
                                         row * k + block + col)

    def test_exact_row_and_column_store_coverage(self):
        for n in (16, 17, 127, 128, 129, 324, 640, 16384):
            stores = []
            for group in range((n + 127) // 128):
                for subgroup in range(4):
                    for tile in range(2):
                        for lane in range(32):
                            for i in range(lane, 256, 32):
                                row = i // 16
                                col = group * 128 + subgroup * 32 + tile * 16 + i % 16
                                if col < n:
                                    stores.append(row * n + col)
            self.assertEqual(sorted(stores), list(range(16 * n)))

    def test_shared_weight_tile_coverage_and_padding(self):
        slots = []
        for thread in range(128):
            for half in range(2):
                n = thread // 2 + half * 64
                for i in range(16):
                    slots.append(n * 40 + (thread % 2) * 16 + i)
        expected = [n * 40 + k for n in range(128) for k in range(32)]
        self.assertEqual(sorted(slots), expected)
        self.assertLess(max(slots), 5120)

    def test_guard_validation_and_original_fallback(self):
        path = Path("/home/sayanshaw/repos/qwen38-int4-generic-matmul-replay/build-v1/matmul_nbits.cc")
        source = transform(path.read_text())
        self.assertEqual(source.count("ORT_WEBGPU_SMALL_M_PREFILL must be 0 or 1"), 2)
        self.assertIn("M == 16 && N >= 16", source)
        self.assertIn("!has_zero_points && !has_bias && !has_weight_idx", source)
        self.assertIn("subgroup_matrix_config->Is(16, 16, 16)", source)
        self.assertIn("return ApplySubgroupMatrixMatMulNBits", source)
        shader = (ROOT / "small_m_prefill.inc").read_text()
        self.assertIn("config.resultComponentType == wgpu::SubgroupMatrixComponentType::F32", shader)
        self.assertIn("k_block += 32u", shader)
        self.assertIn("k += 16u", shader)


if __name__ == "__main__":
    unittest.main()
