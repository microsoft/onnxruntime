from pathlib import Path
import random
import struct
import unittest

from build_hc_post_norm_small_cta import FLAG, STORAGE_BYTES, small_program, transform


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


class SmallCtaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/"
                          "build-v2/hc_post_norm_projection.cc").read_text()

    def test_virtual_norm_lanes_preserve_each_original_serial_sum(self):
        virtual = [(physical, lane) for physical in range(128) for lane in range(physical, 256, 128)]
        self.assertEqual(sorted(lane for _, lane in virtual), list(range(256)))
        visited = [h for _, lane in virtual for h in range(lane, 2560, 256)]
        self.assertEqual(sorted(visited), list(range(2560)))
        result = small_program(self.source)
        self.assertIn("norm_lane < 256u; norm_lane += 128u", result)
        self.assertIn("var h = norm_lane; h < uniforms.hidden; h += 256u", result)
        self.assertEqual(result.count("h += 128u"), 2)

    def test_virtual_tree_preserves_all_original_f32_additions(self):
        rng = random.Random(94813)
        for _ in range(20):
            expected = [f32(rng.uniform(-1e10, 1e10)) for _ in range(1024)]
            actual = expected.copy()
            for width in (128, 64, 32, 16, 8, 4, 2, 1):
                for lane in range(width):
                    for stream in range(4):
                        index = stream * 256 + lane
                        expected[index] = f32(expected[index] + expected[index + width])
                owners = [lane for physical in range(128) for lane in range(physical, width, 128)]
                self.assertEqual(sorted(owners), list(range(width)))
                for lane in owners:
                    for stream in range(4):
                        index = stream * 256 + lane
                        actual[index] = f32(actual[index] + actual[index + width])
                self.assertEqual(actual, expected)

    def test_complete_output_and_shared_storage_ownership(self):
        columns = [group * 4 + lane // 32 for group in range(81)
                   for lane in range(128) if lane % 32 == 0]
        self.assertEqual(columns, list(range(324)))
        self.assertEqual(STORAGE_BYTES, 27392)
        for lane in range(0, 128, 32):
            self.assertLess(lane + 31, 128)
        writers = [h for lane in range(128) for h in range(lane, 2560, 128)]
        self.assertEqual(sorted(writers), list(range(2560)))
        addresses = [(stream * 2560 + h) // 32 * 36 + (stream * 2560 + h) % 32
                     for stream in range(4) for h in writers]
        self.assertEqual(len(set(addresses)), 10240)
        self.assertLess(max(addresses), 11520)

    def test_original_program_math_fallbacks_and_key_are_preserved(self):
        start = self.source.index("class HCPostNormProjectionProgram final")
        stop = self.source.index("class HCPostNormProjectionF16", start)
        original = self.source[start:stop]
        result = transform(self.source)
        candidate = small_program(self.source)
        self.assertIn(original, result)
        self.assertIn(self.source[self.source.index("    HCPostNormProjectionBaselineNormProgram norm"):], result)
        for marker in ("sum += dot(", "total += sum;", "value += partials[local_idx + i];",
                       "let low =", "let high =", "let inv_rms"):
            self.assertEqual([line for line in candidate.splitlines() if marker in line],
                             [line for line in original.splitlines() if marker in line])
        self.assertEqual(candidate.count("workgroupBarrier();"), original.count("workgroupBarrier();"))
        self.assertIn(f'"{FLAG} must be 0 or 1"', result)
        self.assertIn("}, 0, small_cta);", result)
        self.assertIn("SetWorkgroupSize(small_cta ? 128 : 256)", result)
        self.assertIn("SetDispatchGroupSize(small_cta ? 81 : 41)", result)
        self.assertLess(result.index('const int cta_mode'), result.index('context.Input(0)'))


if __name__ == "__main__":
    unittest.main()
