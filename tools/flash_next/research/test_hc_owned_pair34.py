from pathlib import Path
import unittest

from build_hc_owned_pair34 import CLASS, FLAG, make_program, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class OwnedPairTests(unittest.TestCase):
    def test_actual_renderer_preserves_rms_and_ordered_projection(self):
        program = make_program(SOURCE.read_text())
        for shared, scale in ((False, False), (True, False), (True, True)):
            shader = render(program, 1, shared, scale, flag=FLAG, class_name=CLASS)
            self.assertIn("array<vec2<f16>, 5440>", shader)
            self.assertEqual(shader.count("component < 2u"), 2)
            self.assertIn("for (var h = local_idx; h < uniforms.hidden; h += 256u)", shader)
            self.assertIn("for (var width = 128u; width > 0u; width >>= 1u)", shader)
            self.assertIn("sum += dot(x0, w0) + dot(x1, w1)", shader)
            self.assertIn("value += partials[local_idx + i]", shader)
            self.assertIn("let x0 = vec4<f16>(post_values[base / 2u], post_values[base / 2u + 1u])", shader)
            self.assertNotIn("36u", shader)
            self.assertEqual(shader.count("post_values[vector_address"), 8)

    def test_single_pair_owner_padding_bounds_and_projection_mapping(self):
        owners = {}
        for thread in range(256):
            for pair in range(thread, 1280, 256):
                for stream in range(4):
                    logical = stream * 1280 + pair
                    address = logical // 16 * 17 + logical % 16
                    self.assertLess(address, 5440)
                    self.assertNotIn(address, owners)
                    owners[address] = thread
                    for component in range(2):
                        scalar = stream * 2560 + pair * 2 + component
                        self.assertEqual(address * 2 + component, scalar // 32 * 34 + scalar % 32)
        self.assertEqual(len(owners), 5120)
        for block in range(320):
            for index in range(32):
                self.assertIn((block * 34 + index) // 2, owners)
        self.assertEqual(len({lane * 17 % 32 for lane in range(32)}), 32)

    def test_default_program_fallback_geometry_and_replay_are_preserved(self):
        original = SOURCE.read_text()
        changed = transform(original)
        start = original.index("class HCPostNormProjectionProgram final")
        stop = original.index("class HCPostNormProjectionF16", start)
        self.assertIn(original[start:stop], changed)
        suffix = original.index("    HCPostNormProjectionBaselineNormProgram")
        self.assertEqual(changed[changed.index("    HCPostNormProjectionBaselineNormProgram"):], original[suffix:])
        self.assertIn(f'static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)', changed)
        self.assertIn("selector == 0 || selector == 1", changed)
        self.assertIn(".SetWorkgroupSize(256).SetDispatchGroupSize(41)", changed)
        self.assertNotIn("}, 0, owned_pair34)", changed)


if __name__ == "__main__":
    unittest.main()
