from pathlib import Path
import unittest

import hc_up_packed80_transform as packed80
from hc_up_packed160_transform import FLAG, NAME, selected_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-up-parallel2/build-v1/hc_up_premix.cc")


class HCUpPacked160Tests(unittest.TestCase):
    def test_exactly_five_full_warps_and_unique_original_valued_partial_writers(self):
        writes, gates = {}, []
        for thread in range(160):
            column, branch, lane = thread // 40, (thread // 10) % 4, thread % 10
            base = column * 132 + branch * 33
            for slot in (lane, *range(lane + 10, 32, 10)):
                self.assertLess(base + slot, 528)
                self.assertNotIn(base + slot, writes)
                writes[base + slot] = ("active", lane) if slot == lane else 0
            if lane == 0:
                gates.append(column * 4 + branch)
        self.assertEqual(len(writes), 512)
        self.assertEqual(gates, list(range(16)))
        for column in range(4):
            for branch in range(4):
                base = column * 132 + branch * 33
                self.assertEqual([writes[base + slot] for slot in range(32)],
                                 [("active", lane) for lane in range(10)] + [0] * 22)
        for warp in range(5):
            banks = [(((thread // 40) * 132 + ((thread // 10) % 4) * 33) // 2) % 32
                     for thread in range(warp * 32, (warp + 1) * 32) if thread % 10 == 0]
            self.assertEqual(len(banks), len(set(banks)))

    def test_all_output_columns_are_written_once(self):
        outputs, projections = [], []
        for group in range(640):
            for thread in range(0, 160, 10):
                column, branch = thread // 40, (thread // 10) % 4
                output = group * 4 + column
                projections.append(output + branch * 2560)
                if branch == 0:
                    outputs.append(output)
        self.assertEqual(outputs, list(range(2560)))
        self.assertEqual(sorted(projections), list(range(10240)))

    def test_only_geometry_changes_the_already_exact_packed_dot_and_32_term_fold(self):
        source = SOURCE.read_text()
        actual = selected_program(source)
        restored = actual.replace("HCUpPreMixPacked160Program", "HCUpPreMixPacked80Program")
        restored = restored.replace(NAME, packed80.NAME)
        restored = restored.replace("array<f16, 528>", "array<f16, 264>")
        restored = restored.replace("array<f16, 16>", "array<f16, 8>")
        restored = restored.replace("workgroup_idx * 4u + column", "workgroup_idx * 2u + column")
        self.assertEqual(restored, packed80.selected_program(source))
        complete = transform(source)
        self.assertNotIn(packed80.FLAG, complete)
        self.assertIn(f'"{FLAG}", 0', complete)
        self.assertIn("SetWorkgroupSize(selected ? 160 : 128)", complete)
        self.assertIn("SetDispatchGroupSize(selected ? 640 : (parallel2 == 1 ? 1280 : 2560))", complete)
        self.assertIn(source[source.index("    auto projected"):], complete)


if __name__ == "__main__":
    unittest.main()
