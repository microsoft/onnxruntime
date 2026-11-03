from pathlib import Path
import unittest

from hc_up_packed80_transform import FLAG, NAME, selected_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-up-parallel2/build-v1/hc_up_premix.cc")


class HCUpPacked80Tests(unittest.TestCase):
    def test_all_original_32_values_have_unique_initialized_locations(self):
        writes, gates, outputs = {}, [], []
        for thread in range(80):
            column, branch, lane = thread // 40, (thread // 10) % 4, thread % 10
            base = column * 132 + branch * 33
            for slot in (lane, *range(lane + 10, 32, 10)):
                index = base + slot
                self.assertLess(index, 264)
                self.assertNotIn(index, writes)
                writes[index] = ("active", lane) if slot == lane else 0
            if lane == 0:
                gates.append(column * 4 + branch)
                if branch == 0:
                    outputs.append(column)
        self.assertEqual(len(writes), 256)
        self.assertEqual(gates, list(range(8)))
        self.assertEqual(outputs, [0, 1])
        for column in range(2):
            for branch in range(4):
                base = column * 132 + branch * 33
                self.assertEqual([writes[base + slot] for slot in range(32)],
                                 [("active", lane) for lane in range(10)] + [0] * 22)
                self.assertNotIn(base + 32, writes)

    def test_padding_avoids_same_bank_gate_fold_reads_within_each_warp(self):
        for warp in range(3):
            banks = []
            for thread in range(warp * 32, min((warp + 1) * 32, 80)):
                column, branch, lane = thread // 40, (thread // 10) % 4, thread % 10
                if lane == 0:
                    banks.append(((column * 132 + branch * 33) * 2 // 4) % 32)
            self.assertEqual(len(banks), len(set(banks)))

    def test_dot_and_all_32_fold_additions_are_unchanged(self):
        source = SOURCE.read_text()
        actual = selected_program(source)
        start = source.index("class HCUpPreMixPacked2Program final")
        stop = source.index("class HCUpPreMixF16 final", start)
        original = source[start:stop]
        restored = actual.replace("HCUpPreMixPacked80Program", "HCUpPreMixPacked2Program")
        for old, new in ((NAME, "HCUpPreMixPacked2F16"), ("array<f16, 264>", "array<f16, 256>"),
                         ("local_idx % 10u", "local_idx % 16u"),
                         ("local_idx / 10u", "local_idx / 16u"),
                         ("local_idx / 40u", "local_idx / 64u"),
                         ("column * 132u + branch * 33u", "column * 128u + branch * 32u")):
            restored = restored.replace(old, new)
        restored = restored.replace(
            '<< "for (var slot = lane + 10u; slot < 32u; slot += 10u) {\\n"\n'
            '         << "  partials[partial_idx + slot - lane] = f16(0);\\n"\n'
            '         << "}\\n"', '<< "partials[partial_idx + 16u] = f16(0);\\n"')
        self.assertEqual(restored, original)
        self.assertIn("i < 32u; i++) { value += partials[partial_idx + i]", actual)
        self.assertEqual(actual.count("workgroupBarrier()"), 2)
        self.assertNotIn("value += f16(0);", actual)

    def test_default_programs_fallback_and_abi_are_preserved(self):
        source = SOURCE.read_text()
        actual = transform(source)
        self.assertTrue(actual.startswith(source[:source.index("class HCUpPreMixF16 final")]))
        self.assertIn(source[source.index("    auto projected"):], actual)
        self.assertIn(source[source.index("\n private:"):], actual)
        self.assertIn(f'"{FLAG}", 0', actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("program.SetWorkgroupSize(selected ? 80 : 128)", actual)
        self.assertIn("}, 0, selected ? 2 : parallel2);", actual)


if __name__ == "__main__":
    unittest.main()
