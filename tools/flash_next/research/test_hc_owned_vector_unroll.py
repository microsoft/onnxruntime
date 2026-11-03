from pathlib import Path
import unittest

from build_hc_owned_vector_unroll import CLASS, FLAG, make_program, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class OwnedVectorUnrollTests(unittest.TestCase):
    def test_actual_rendered_constant_components(self):
        program = make_program(SOURCE.read_text())
        for shared, scale in ((False, False), (True, False), (True, True)):
            shader = render(program, 1, shared, scale, flag=FLAG, class_name=CLASS)
            self.assertNotIn("for (var component", shader)
            for component in range(4):
                self.assertEqual(shader.count(f"let component = {component}u"), 2)
            self.assertEqual(shader.count("post_values[vector_address"), 8)
            self.assertIn("for (var h = local_idx; h < uniforms.hidden; h += 256u)", shader)
            self.assertIn("for (var width = 128u; width > 0u; width >>= 1u)", shader)
            self.assertIn("sum += dot(x0, w0) + dot(x1, w1)", shader)
            self.assertIn("total += sum", shader)
            self.assertIn("value += partials[local_idx + i]", shader)

    def test_selector_replay_original_fallback(self):
        source = SOURCE.read_text()
        result = transform(source)
        self.assertIn(f'static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)', result)
        self.assertIn(".SetWorkgroupSize(256).SetDispatchGroupSize(41)", result)
        self.assertNotIn("}, 0, owned_vector_unroll)", result)
        suffix = source.index("    HCPostNormProjectionBaselineNormProgram")
        self.assertEqual(result[result.index("    HCPostNormProjectionBaselineNormProgram"):], source[suffix:])
