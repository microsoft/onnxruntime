from pathlib import Path
import unittest

from build_expert_bitcast_unpack import FLAG, transform_program, transform_qmoe


FROZEN = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-hierarchical-gate/build-v2/overlay/contrib_ops/webgpu/moe")


class ExpertBitcastTests(unittest.TestCase):
    def test_both_programs_preserve_reference_math_and_have_distinct_shader_names(self):
        for kind, file, name in (("fc1", "qmoe_fc1_activation.inc", "QmoeFC1SwiGLU"),
                                 ("fc2", "qmoe_fc2_final_mix.inc", "QmoeFC2FinalMix")):
            source = transform_program((FROZEN / file).read_text(), kind)
            self.assertIn(f'bitcast ? "{name}BitcastF16" : "{name}F16"', source)
            self.assertEqual(source.count("decode_low_nibbles("), 3)
            self.assertEqual(source.count("unpack4xU8("), 2)

    def test_selector_validation_precedes_replay_and_is_in_its_key(self):
        source = transform_qmoe((FROZEN / "qmoe.cc").read_text())
        self.assertIn(f'"{FLAG} must be 0 or 1"', source)
        self.assertIn("key.push_back(expert_bitcast)", source)
        self.assertIn("fc1_experts_bias_optional != nullptr, expert_bitcast == 1", source)
        self.assertIn("fc2_experts_bias_optional != nullptr, expert_bitcast == 1", source)


if __name__ == "__main__":
    unittest.main()
