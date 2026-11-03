from pathlib import Path
import unittest

from build_fc2_four_columns import transform_program, transform_qmoe


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-expert-bitcast-unpack/build-v2/overlay/contrib_ops/webgpu/moe")


class FourColumnTests(unittest.TestCase):
    def test_program_keeps_arithmetic_and_expands_only_column_dimensions(self):
        source = (SOURCE / "qmoe_fc2_final_mix.inc").read_text()
        result = transform_program(source)
        self.assertIn('four_columns ? (bitcast ? "QmoeFC2FinalMixFourColumnsBitcastF16"', result)
        self.assertIn(': "QmoeFC2FinalMixFourColumnsF16")', result)
        self.assertIn("columns_{four_columns ? 4u : 2u}", result)
        self.assertIn('<< 10u * columns_ << ">;\\n";', result)
        self.assertIn('partials[r * " << columns_ << "u + local_idx][0]', result)
        for line in source.splitlines():
            if any(expression in line for expression in
                   ("sum += dot", "value += partials", "acc += router_value",
                    "decode_low_nibbles", "tile_a[id]")):
                self.assertIn(line.replace("partials[r * 2u + local_idx]",
                    'partials[r * " << columns_ << "u + local_idx]'), result)

    def test_workspace_selector_and_replay_key_are_wired(self):
        result = transform_qmoe((SOURCE / "qmoe.cc").read_text())
        self.assertEqual(result.count("ORT_WEBGPU_QMOE_FC2_FOUR_COLUMNS must be 0 or 1"), 2)
        self.assertIn("key.push_back(fc2_four_columns);", result)
        self.assertIn("maxComputeWorkgroupStorageSize >= 15360", result)
        self.assertIn(".SetDispatchGroupSize(four_columns ? 640 : 1280)", result)


if __name__ == "__main__":
    unittest.main()
