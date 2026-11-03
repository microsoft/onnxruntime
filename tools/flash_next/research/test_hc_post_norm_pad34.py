from pathlib import Path
import unittest

from build_hc_post_norm_pad34 import FLAG, padded_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class HCPostNormPad34Tests(unittest.TestCase):
    def test_only_shared_addresses_storage_and_private_identity_change(self):
        source = SOURCE.read_text()
        original = source[source.index("class HCPostNormProjectionProgram"):
                          source.index("class HCPostNormProjectionF16")]
        padded = padded_program(source)
        restored = (padded.replace("HCPostNormProjectionPad34Program", "HCPostNormProjectionProgram")
                    .replace('Program{"HCPostNormProjectionPad34F16"}', 'Program{"HCPostNormProjectionF16"}')
                    .replace("34u", "36u").replace("array<f16, 10880>", "array<f16, 11520>"))
        self.assertEqual(restored, original)
        self.assertEqual(padded.count("34u"), 4)

    def test_all_logical_values_and_dot_reads_map_to_same_unique_values(self):
        addresses = [(logical // 32) * 34 + logical % 32 for logical in range(10240)]
        self.assertEqual(len(set(addresses)), 10240)
        self.assertLess(max(addresses), 10880)
        self.assertLessEqual(10880 * 2 + 1024 * 4 + 256 * 2, 27648)
        for block in range(320):
            for i in range(4):
                for lane in range(8):
                    self.assertEqual(block * 34 + i * 8 + lane,
                                     addresses[block * 32 + i * 8 + lane])

    def test_bank_address_hypothesis_not_a_hardware_speed_claim(self):
        self.assertEqual(len({(lane * 36 // 2) % 32 for lane in range(32)}), 16)
        self.assertEqual(len({(lane * 34 // 2) % 32 for lane in range(32)}), 32)

    def test_original_shader_fallback_limits_and_replay_layout_preserved(self):
        source = SOURCE.read_text()
        actual = transform(source)
        self.assertIn(source[source.index("class HCPostNormProjectionProgram"):
                             source.index("class HCPostNormProjectionF16")], actual)
        self.assertIn(source[source.index("    HCPostNormProjectionBaselineNormProgram norm"):], actual)
        self.assertIn(f'"{FLAG}", 0', actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("}, 0, pad34);", actual)
        self.assertEqual(actual.count("limits.maxComputeWorkgroupStorageSize >= 27648"), 1)
        self.assertEqual(actual.count(".SetWorkgroupSize(256).SetDispatchGroupSize(41)"), 1)


if __name__ == "__main__":
    unittest.main()
