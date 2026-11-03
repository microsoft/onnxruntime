from pathlib import Path
import unittest

from build_fc1_k8_tile import FLAG, transform
from run_hc_up_parallel2_model import ROW4_REPEAT_VARIANTS


class Fc1K8TileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        cls.program = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1/overlay/"
                           "contrib_ops/webgpu/moe/qmoe_fc1_activation.inc").read_text()

    def test_each_projected_column_keeps_original_lane_update_sequence(self):
        for col in range(16):
            for lane in range(32):
                original = [kidx // 32 + lane for kidx in range(0, 2560, 1024)
                            if kidx // 32 + lane < 80]
                candidate = [kidx // 32 + tile_lane for kidx in range(0, 2560, 256)
                             for tile_lane in range(8) if (kidx // 32) % 32 + tile_lane == lane]
                self.assertEqual(candidate, original, (col, lane))
        outputs = [(group // 80, (group % 80) * 8 + writer)
                   for group in range(800) for writer in range(8)]
        self.assertEqual(sorted(outputs), [(expert, col) for expert in range(10) for col in range(640)])

    def test_original_arithmetic_fallback_and_replay_preserved(self):
        actual = transform(self.source, self.program)
        self.assertEqual(actual.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(fc1_k8_tile)", actual)
        for expression in ("partials[offset + group][lane] += sum", "kidx += 256u", "kidx += 1024u",
                           "tile_a: array<vec4<f16>, 64>", "tile_a: array<vec4<f16>, 256>",
                           "value += partials[local_idx][i]",
                           "f16(glu * (linear_value + uniforms.beta))",
                           "fc1_k8_tile == 1 ? 16u : 8u",
                           ".SetWorkgroupSize(128).SetDispatchGroupSize(tiles, routed_rows)"):
            self.assertIn(expression, actual)
        self.assertNotIn("ORT_WEBGPU_QMOE_FC1_K16_TILE", actual)

    def test_model_selector_and_long_repeat_gates_wired(self):
        benchmark = Path(__file__).with_name("benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc1-k8-tile-windows", benchmark)
        self.assertIn("FC1_K8_TILE_FLAG if args.compare_fc1_k8_tile_windows", benchmark)
        self.assertIn("fc1-k8-tile-v1", ROW4_REPEAT_VARIANTS)


if __name__ == "__main__":
    unittest.main()
