from pathlib import Path
import unittest

from build_hc_up_parallel4 import FLAG, packed_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-up-parallel2/build-v1/hc_up_premix.cc")


class HCUpParallel4Tests(unittest.TestCase):
    def test_only_storage_column_index_and_identity_change(self):
        source = SOURCE.read_text()
        start = source.index("class HCUpPreMixPacked2Program")
        stop = source.index("class HCUpPreMixF16", start)
        restored = (packed_program(source).replace("HCUpPreMixPacked4Program", "HCUpPreMixPacked2Program")
                    .replace("HCUpPreMixPacked4F16", "HCUpPreMixPacked2F16")
                    .replace("array<f16, 512>", "array<f16, 256>")
                    .replace("array<f16, 16>", "array<f16, 8>")
                    .replace("workgroup_idx * 4u + column", "workgroup_idx * 2u + column"))
        self.assertEqual(restored, source[start:stop])

    def test_all_partial_slots_disjoint_and_all_outputs_covered(self):
        partials = []
        gates = []
        for thread in range(256):
            lane, branch, column = thread % 16, (thread // 16) % 4, thread // 64
            start = column * 128 + branch * 32 + lane
            partials.extend((start, start + 16))
            if lane == 0:
                gates.append(column * 4 + branch)
        self.assertEqual(sorted(partials), list(range(512)))
        self.assertEqual(sorted(gates), list(range(16)))
        self.assertEqual([group * 4 + column for group in range(640) for column in range(4)],
                         list(range(2560)))

    def test_original_shaders_fallback_and_replay_keys_preserved(self):
        source = SOURCE.read_text()
        actual = transform(source)
        self.assertIn(source[:source.index("class HCUpPreMixF16")], actual)
        self.assertIn(source[source.index("    auto projected ="):], actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 1056", actual)
        self.assertIn("}, 0, parallel4 ? 2 : parallel2);", actual)
        self.assertIn("maxComputeWorkgroupSizeX >= 256", actual)


if __name__ == "__main__":
    unittest.main()
