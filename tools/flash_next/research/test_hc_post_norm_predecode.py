from pathlib import Path
import unittest

from build_hc_post_norm_predecode import FLAG, PREDECODE_PROGRAM, decoded_program, transform
from build_hc_staged_projection import stage_replay


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class HCPostNormPredecodeTests(unittest.TestCase):
    def test_normalization_dot_and_sequential_reduction_remain_unchanged(self):
        source = SOURCE.read_text()
        original = source[source.index("class HCPostNormProjectionProgram"):
                          source.index("class HCPostNormProjectionF16")]
        actual = decoded_program(source)
        normalization_start = '    shader.AdditionalImplementation() <<'
        normalization_stop = '         << "var total = f16(0);\\n"'
        self.assertIn(original[original.index(normalization_start):
                               original.index(normalization_stop)], actual)
        self.assertIn(original[original.index('         << "    let base = block * 36u'):], actual)
        for expression in ("low.x, high.x, low.y, high.y", "low.z, high.z, low.w, high.w"):
            self.assertIn(expression, PREDECODE_PROGRAM)

    def test_predecode_layout_matches_each_projection_block_and_lane(self):
        slots = set()
        for col in range(324):
            for block in (0, 31, 32, 319):
                for i in range(4):
                    for half in range(2):
                        index = (col * 320 + block) * 8 + i * 2 + half
                        self.assertEqual(index * 4, col * 10240 + block * 32 + i * 8 + half * 4)
                        slots.add(index)
        self.assertEqual(len(slots), 324 * 4 * 8)
        self.assertEqual(1620 * 64, 324 * 320)

    def test_cache_is_constant_only_and_replay_variant_is_separate(self):
        source = SOURCE.read_text()
        actual = transform(source)
        self.assertIn(f'"{FLAG}", 0', actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("predecode_mode == 1 && weights_constant_", actual)
        self.assertIn("info.TryGetConstantInput(shared ? 7 : 5", actual)
        self.assertIn("info.TryGetConstantInput(shared ? 8 : 6", actual)
        self.assertIn("std::call_once(decode_once_", actual)
        self.assertIn("ORT_THROW_IF_ERROR(context.RunProgram(program))", actual)
        self.assertIn("}, 0, 0, {decoded_weights_.get()});", actual)
        self.assertIn("decoded_replay_.Run(context", actual)
        self.assertIn(source[source.index("class HCPostNormProjectionProgram"):
                             source.index("class HCPostNormProjectionF16")], actual)
        self.assertIn(source[source.index("    HCPostNormProjectionBaselineNormProgram norm"):
                             source.index(" protected:")], actual)

    def test_explicit_extra_buffer_binding_retains_capture_validation(self):
        source = (SOURCE.parents[2] / "qwen38-int4-dense-bitcast-unpack/build-v1/overlay/static_node_replay.h")
        replay = stage_replay(source.read_text())
        self.assertIn("for (const auto* input : extra_inputs) append(input, false);", replay)
        self.assertIn("entry.bindings.buffers == expected", replay)
        self.assertIn("tensor->Location().name == WEBGPU_BUFFER", replay)
        program = decoded_program(SOURCE.read_text())
        self.assertLess(program.index('shader.AddInput("b"'), program.index('shader.AddInput("streams"'))
        self.assertIn('program.AddInput({decoded_weights_.get()', transform(SOURCE.read_text()))


if __name__ == "__main__":
    unittest.main()
