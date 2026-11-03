from pathlib import Path
import unittest

import test_dense_columns8_double_buffer as protocol
from build_qmoe_fc1_double_buffer import FLAG, HELPER, STORAGE_BYTES, transform, transform_program
from test_hc_projection_inner_unroll import render


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1")
HEADER = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe/qmoe_fc1_activation.inc")


class QmoeFC1DoubleBufferTests(unittest.TestCase):
    def test_transform_admission_and_replay_surfaces(self):
        reference = HEADER.read_text()
        program = transform_program(reference)
        self.assertEqual(program[program.index("bool QMoE::CanFuseFc1Activation"):],
                         reference[reference.index("bool QMoE::CanFuseFc1Activation"):])
        candidate = transform((DIRECTORY / "qmoe.cc").read_text(), reference)
        self.assertEqual(candidate.count("const int fc1_double_buffer = QmoeFc1DoubleBufferMode();"), 2)
        self.assertIn("key.push_back(fc1_double_buffer);", candidate)
        self.assertIn("fc1_double_buffer == 1 && expert_bitcast == 1", candidate)
        self.assertIn("maxComputeWorkgroupStorageSize >= 4608", candidate)
        self.assertIn(".CacheHint(fc1_experts_bias_optional != nullptr, use_fc1_double_buffer);", candidate)
        self.assertIn(".SetWorkgroupSize(128).SetDispatchGroupSize(tiles, routed_rows)", candidate)
        self.assertIn("static const int mode", HELPER)
        self.assertIn(f"{FLAG} must be 0 or 1", HELPER)
        self.assertEqual(STORAGE_BYTES, 512 * 8 + 8 * 32 * 2)

    def test_randomized_four_warp_buffer_protocol(self):
        protocol.Columns8DoubleBufferTests().test_randomized_warp_scheduling_preserves_buffers()

    def test_rendered_default_and_candidate_arithmetic(self):
        reference = HEADER.read_text().split("bool QMoE::CanFuseFc1Activation", 1)[0]
        candidate = transform_program(HEADER.read_text()).split("bool QMoE::CanFuseFc1Activation", 1)[0]
        for bias in (False, True):
            for bitcast in (False, True):
                original = render(reference, 0, False, False, "", FLAG,
                                  class_name="QmoeFC1SwiGLUProgram", constructor_args=(bias, bitcast))
                default = render(candidate, 0, False, False, "", FLAG,
                                 class_name="QmoeFC1SwiGLUProgram", constructor_args=(bias, bitcast, False))
                self.assertEqual(default, original)
                active = render(candidate, 1, False, False, "", FLAG,
                                class_name="QmoeFC1SwiGLUProgram", constructor_args=(bias, bitcast, True))
                normalized = active.replace("tile_a: array<vec4<f16>, 512>", "tile_a: array<vec4<f16>, 256>")
                normalized = normalized.replace("  let buffer = ((kidx / 1024u) % 2u) * 256u;\n", "")
                normalized = normalized.replace("tile_a[buffer + ", "tile_a[")
                normalized = normalized.replace("}\nworkgroupBarrier();\nif (local_idx < 8u",
                                                "  workgroupBarrier();\n}\nif (local_idx < 8u")
                self.assertEqual(normalized, original)


if __name__ == "__main__":
    unittest.main()
