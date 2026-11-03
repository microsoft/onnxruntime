from pathlib import Path
import unittest

from build_hc_owned_vector import CLASS, FLAG, make_program, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class OwnedVectorTests(unittest.TestCase):
    def test_actual_shader_renderer_preserves_original_arithmetic(self):
        original = SOURCE.read_text()
        program = make_program(original)
        self.assertNotIn("fn store_post", program)
        for shared, scale in ((False, False), (True, False), (True, True)):
            shader = render(program, 1, shared, scale, flag=FLAG, class_name=CLASS)
            self.assertEqual(shader.count("for (var quad = local_idx;"), 2)
            self.assertEqual(shader.count("post_values[vector_address"), 8)
            self.assertIn("for (var h = local_idx; h < uniforms.hidden; h += 256u)", shader)
            self.assertIn("for (var width = 128u; width > 0u; width >>= 1u)", shader)
            self.assertIn("sum += dot(x0, w0) + dot(x1, w1)", shader)
            self.assertIn("total += sum", shader)
            self.assertIn("value += partials[local_idx + i]", shader)
            self.assertIn("let x0 = post_values[base / 4u]", shader)
            self.assertIn("let x1 = post_values[base / 4u + 1u]", shader)

    def test_each_vector_and_component_has_one_owner(self):
        owners = {}
        for thread in range(256):
            for quad in range(thread, 640, 256):
                for stream in range(4):
                    logical = stream * 640 + quad
                    address = (logical // 8) * 9 + logical % 8
                    self.assertNotIn(address, owners)
                    owners[address] = thread
                    for component in range(4):
                        scalar = stream * 2560 + quad * 4 + component
                        scalar_address = (scalar // 32) * 36 + scalar % 32
                        self.assertEqual(address * 4 + component, scalar_address)
        self.assertEqual(len(owners), 2560)

    def test_immutable_selector_original_geometry_replay_and_fallback(self):
        original = SOURCE.read_text()
        result = transform(original)
        self.assertIn(f'static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)', result)
        self.assertIn(".SetWorkgroupSize(256).SetDispatchGroupSize(41)", result)
        self.assertNotIn("}, 0, owned_vector)", result)
        suffix = original.index("    HCPostNormProjectionBaselineNormProgram")
        self.assertEqual(result[result.index("    HCPostNormProjectionBaselineNormProgram"):], original[suffix:])


if __name__ == "__main__":
    unittest.main()
