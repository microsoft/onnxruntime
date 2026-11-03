from pathlib import Path
import struct
import unittest

from hc_up_zero_tail_transform import FLAG, NAME, selected_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-up-parallel2/build-v1/hc_up_premix.cc")


class HCUpZeroTailTests(unittest.TestCase):
    def test_only_the_proven_zero_tail_changes_shader_arithmetic(self):
        source = SOURCE.read_text()
        start = source.index("class HCUpPreMixPacked2Program final")
        stop = source.index("class HCUpPreMixF16 final", start)
        actual = selected_program(source)
        restored = actual.replace("HCUpPreMixZeroTailProgram", "HCUpPreMixPacked2Program")
        restored = restored.replace(NAME, "HCUpPreMixPacked2F16")
        restored = restored.replace("i < 10u; i++) { value += partials",
                                    "i < 32u; i++) { value += partials")
        restored = restored.replace('         << "  value += f16(0);\\n"\n', "")
        self.assertEqual(restored, source[start:stop])
        self.assertEqual(actual.count("workgroupBarrier()"), 2)

    def test_every_omitted_partial_has_one_original_positive_zero_writer(self):
        writes = {}
        for thread in range(128):
            column, branch, lane = thread // 64, (thread // 16) % 4, thread % 16
            base = column * 128 + branch * 32
            for slot, value in ((base + lane, "active" if lane < 10 else 0),
                                (base + lane + 16, 0)):
                self.assertNotIn(slot, writes)
                writes[slot] = value
        self.assertEqual(sorted(writes), list(range(256)))
        for column in range(2):
            for branch in range(4):
                base = column * 128 + branch * 32
                self.assertEqual([writes[base + i] for i in range(10, 32)], [0] * 22)

    def test_zero_addition_is_idempotent_for_every_finite_half_including_negative_zero(self):
        def zero_add(bits):
            value = struct.unpack("<e", struct.pack("<H", bits))[0]
            return struct.unpack("<H", struct.pack("<e", value + 0.0))[0]

        checked = 0
        for bits in range(65536):
            if bits & 0x7c00 == 0x7c00:
                continue
            once = zero_add(bits)
            original = bits
            for _ in range(22):
                original = zero_add(original)
            self.assertEqual(once, original, hex(bits))
            checked += 1
        self.assertEqual(checked, 63488)
        self.assertEqual(zero_add(0x8000), 0)
        self.assertEqual(zero_add(1), 1)
        self.assertEqual(zero_add(0x8001), 0x8001)

    def test_default_shaders_fallback_geometry_and_object_layout_are_unchanged(self):
        source = SOURCE.read_text()
        actual = transform(source)
        original_programs = source[:source.index("class HCUpPreMixF16 final")]
        self.assertTrue(actual.startswith(original_programs))
        self.assertIn(source[source.index("    auto projected"):], actual)
        self.assertIn(source[source.index("\n private:"):], actual)
        self.assertIn(f'static const int zero_tail = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)', actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("zero_tail == 1 && parallel2 == 1", actual)
        self.assertIn("}, 0, selected ? 2 : parallel2);", actual)
        self.assertIn("params.rows == 1 && !context.EnableMatmulFp32Accumulation()", actual)
        self.assertIn("program.SetWorkgroupSize(128).SetDispatchGroupSize(parallel2 == 1 ? 1280 : 2560)", actual)


if __name__ == "__main__":
    unittest.main()
