from pathlib import Path
import unittest

from build_qmoe_fc2_padding import FLAG, transform, transform_shader


ROOT = Path(__file__).parent
PARENT = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-fc2-final-mix/build-v2/overlay/contrib_ops/webgpu/moe")


class PaddingTests(unittest.TestCase):
    def test_padding_only_changes_shared_layout(self):
        source = (ROOT / "qmoe_fc2_final_mix.inc").read_text()
        padded = transform_shader(source)
        self.assertIn("(padded_ ? 33 : 32)", padded)
        self.assertIn('"QmoeFC2FinalMixF16Padded"', padded)
        self.assertEqual(source[source.index("    auto& code"):source.index(" private:")],
                         padded[padded.index("    auto& code"):padded.index(" private:")])

    def test_cache_keys_and_capability_guard(self):
        source = transform((PARENT / "qmoe.cc").read_text())
        self.assertIn("key.push_back(fc2_padding);", source)
        self.assertIn(".CacheHint(fc2_experts_bias_optional != nullptr, padded);", source)
        self.assertIn("maxComputeWorkgroupStorageSize >= 14120", source)
        self.assertIn(f'"{FLAG} must be 0 or 1"', source)

    def test_storage_and_cell_ownership(self):
        self.assertEqual(1600 * 8 + 20 * 33 * 2, 14120)
        cells = {(row, lane) for row in range(20) for lane in range(32)}
        self.assertEqual(len({row * 33 + lane for row, lane in cells}), len(cells))
        self.assertLess(max(row * 33 + lane for row, lane in cells), 20 * 33)


if __name__ == "__main__":
    unittest.main()
