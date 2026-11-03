from pathlib import Path
import unittest

from build_fc1_paired_dot import FLAG, transform, transform_program
from run_hc_up_parallel2_model import ROW4_REPEAT_VARIANTS


class Fc1PairedDotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        cls.program = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1/overlay/"
                           "contrib_ops/webgpu/moe/qmoe_fc1_activation.inc").read_text()

    def test_original_columns_and_32_element_k_partition(self):
        for group in range(4):
            self.assertEqual([group, group + 4], [offset + group for offset in (0, 4)])
        for tile in range(160):
            for group in range(4):
                self.assertLess(tile * 8 + group + 4, 1280)
        actual = transform_program(self.program)
        self.assertIn("index1 = index0 + 4u * blocks", actual)
        self.assertIn("kidx += 1024u", actual)
        self.assertIn("partials[group][lane] += sums.x", actual)
        self.assertIn("partials[group + 4u][lane] += sums.y", actual)

    def test_original_path_fold_bias_and_swiglu_preserved(self):
        actual = transform_program(self.program)
        start = self.program.index('         << "  for (var offset')
        stop = self.program.index('         << "  workgroupBarrier();\\n"\n         << "}\\n"', start)
        self.assertIn(self.program[start:stop], actual)
        start = self.program.index('         << "if (local_idx < 8u')
        stop = self.program.index("    return Status::OK();", start)
        self.assertIn(self.program[start:stop], actual)
        self.assertIn("sums += vec2<f16>(dot(a0, w00) + dot(a1, w01), dot(a0, w10) + dot(a1, w11))", actual)

    def test_fixed_admission_validation_replay_and_original_geometry(self):
        actual = transform(self.source, self.program)
        self.assertEqual(actual.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(fc1_paired_dot)", actual)
        self.assertIn("fc1_paired_dot == 1 && moe_params.inter_size == 640", actual)
        self.assertIn(".SetWorkgroupSize(128).SetDispatchGroupSize(tiles, routed_rows)", actual)

    def test_model_wiring_and_repeat_prerequisites_required(self):
        self.assertIn("fc1-paired-dot-v1", ROW4_REPEAT_VARIANTS)
        benchmark = Path(__file__).with_name("benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc1-paired-dot-windows", benchmark)
        self.assertIn("FC1_PAIRED_DOT_FLAG if args.compare_fc1_paired_dot_windows", benchmark)


if __name__ == "__main__":
    unittest.main()
