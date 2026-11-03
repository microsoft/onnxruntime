from pathlib import Path
import unittest

import numpy as np

from build_mlp_bitcast_unpack import FLAG, transform


class BitcastUnpackTests(unittest.TestCase):
    def test_every_nibble_value_and_position_matches_original_half_bits(self):
        rng = np.random.default_rng(20261029)
        words = np.concatenate((
            np.arange(65536, dtype=np.uint32),
            np.arange(65536, dtype=np.uint32) << 16,
            rng.integers(0, 2**32, 65536, dtype=np.uint32)))
        for shift in (0, 4):
            value = words >> shift
            bottom = ((value & 0x000f000f) | 0x64006400).view(np.float16).reshape(-1, 2) - np.float16(1032)
            top = (((value >> 8) & 0x000f000f) | 0x64006400).view(np.float16).reshape(-1, 2) - np.float16(1032)
            actual = np.stack((bottom[:, 0], top[:, 0], bottom[:, 1], top[:, 1]), axis=1)
            expected = np.stack([((value >> offset) & 15).astype(np.float16) - np.float16(8)
                                 for offset in (0, 8, 16, 24)], axis=1)
            np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))

    def test_selector_validation_and_independent_replay_families(self):
        source = transform((Path(__file__).parent / "mlp_input_projections.cc").read_text())
        self.assertIn(f'"{FLAG} must be 0 or 1"', source)
        self.assertIn("bitcast == 1 ? bitcast_replay_ : replay_", source)
        self.assertIn('bitcast ? "MlpInputProjectionsBitcastF16" : "MlpInputProjectionsF16"', source)
        self.assertEqual(source.count("decode_low_nibbles("), 5)
        self.assertEqual(source.count("unpack4xU8("), 4)


if __name__ == "__main__":
    unittest.main()
