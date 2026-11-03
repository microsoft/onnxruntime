from pathlib import Path
import unittest

from build_hc_staged_projection import FLAG, private_programs, stage_replay, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")
REPLAY = Path("/home/sayanshaw/repos/qwen38-int4-dense-bitcast-unpack/build-v1/overlay/static_node_replay.h")


class HCStagedProjectionTests(unittest.TestCase):
    def test_shader_arithmetic_is_identical_to_original_fallback(self):
        source = SOURCE.read_text()
        start = source.index("class HCPostNormProjectionBaselineNormProgram")
        stop = source.index("class HCPostNormProjectionProgram final", start)
        restored = (private_programs(source)
                    .replace("HCStagedNormProgram", "HCPostNormProjectionBaselineNormProgram")
                    .replace("HCStagedPostNormF16", "HCPostNormF16")
                    .replace("HCStagedProjectionProgram", "HCPostNormProjectionBaselineMatMulProgram")
                    .replace("HCStagedProjectionF16", "HCProjectionSiLUF16"))
        self.assertEqual(restored, source[start:stop])

    def test_stage_route_does_not_change_fused_or_fallback(self):
        source = SOURCE.read_text()
        actual = transform(source)
        original = source[source.index("    if (fused) {"):source.index("\n protected:")]
        self.assertIn(original, actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("if (fused && staged == 1)", actual)
        self.assertIn("stage_norm_.Run(context, {y, normalized}", actual)
        self.assertIn("stage_projection_.Run(context, {activated, tail}", actual)
        self.assertIn("}, 0, 0, {normalized});", actual)
        self.assertIn("std::initializer_list<int>{7, 8}", actual)
        self.assertIn("std::initializer_list<int>{5, 6}", actual)
        self.assertIn("std::initializer_list<int>{0, 1, 2, 3, 4, 5}", actual)

    def test_extra_input_is_shape_keyed_and_bound_before_weights_and_outputs(self):
        source = REPLAY.read_text()
        actual = stage_replay(source)
        self.assertIn("for (const auto* input : extra_inputs) append(input, false)", actual)
        self.assertLess(actual.index("for (const auto* tensor : extra_inputs)"),
                        actual.index("for (int index : gpu_inputs_)"))
        self.assertIn("entry.bindings.buffers == expected", actual)
        self.assertIn("candidate.buffers == buffers", actual)
        self.assertIn("entry.commands.size() == 1", actual)
        self.assertIn("NodeReplayProfilingEnabled()", actual)
        self.assertIn("cpu_values", actual)
        self.assertIn("HCStageReplay", actual)
        self.assertNotIn("class StaticNodeReplay", actual)


if __name__ == "__main__":
    unittest.main()
