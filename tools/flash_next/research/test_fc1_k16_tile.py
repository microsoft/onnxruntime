from pathlib import Path
import unittest

from build_fc1_k16_tile import FLAG, transform, transform_program
from run_hc_up_parallel2_model import ROW4_REPEAT_VARIANTS


class Fc1K16TileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        cls.program = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1/overlay/"
                           "contrib_ops/webgpu/moe/qmoe_fc1_activation.inc").read_text()

    def test_each_column_preserves_original_partial_lane_update_sequence(self):
        original = {(col, lane): [] for col in range(8) for lane in range(32)}
        candidate = {(col, lane): [] for col in range(8) for lane in range(32)}
        for kidx in range(0, 2560, 1024):
            for thread in range(128):
                group, lane = divmod(thread, 32)
                block = kidx // 32 + lane
                if block < 80:
                    for offset in (0, 4):
                        original[group + offset, lane].append(block)
        for kidx in range(0, 2560, 512):
            for thread in range(128):
                col, tile_lane = divmod(thread, 16)
                lane = (kidx // 32) % 32 + tile_lane
                candidate[col, lane].append(kidx // 32 + tile_lane)
                for word in range(4):
                    for pair in (0, 1):
                        self.assertEqual(kidx + (tile_lane * 8 + word * 2 + pair) * 4,
                                         (kidx // 32 + tile_lane) * 32 + word * 8 + pair * 4)
        self.assertEqual(candidate, original)

    def test_original_dot_expressions_folds_and_geometry_preserved(self):
        actual = transform_program(self.program)
        self.assertIn("partials[offset + group][lane] += sum", actual)
        start = self.program.index('         << "if (local_idx < 8u')
        stop = self.program.index("    return Status::OK();", start)
        self.assertIn(self.program[start:stop], actual)
        self.assertIn("kidx += 512u", actual)
        self.assertIn("kidx += 1024u", actual)
        self.assertIn("tile_a: array<vec4<f16>, 128>", actual)
        actual = transform(self.source, self.program)
        self.assertEqual(actual.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(fc1_k16_tile)", actual)
        self.assertIn(".SetWorkgroupSize(128).SetDispatchGroupSize(tiles, routed_rows)", actual)

    def test_model_wiring_and_repeat_prerequisites_required(self):
        self.assertIn("fc1-k16-tile-v1", ROW4_REPEAT_VARIANTS)
        benchmark = Path(__file__).with_name("benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc1-k16-tile-windows", benchmark)
        self.assertIn("FC1_K16_TILE_FLAG if args.compare_fc1_k16_tile_windows", benchmark)


if __name__ == "__main__":
    unittest.main()
