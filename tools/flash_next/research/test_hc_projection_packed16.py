import ast
from pathlib import Path
import unittest

from build_hc_projection_inner_unroll import program_parts
from build_hc_projection_packed16 import FLAG, HELPER, transform_program, transform_source
from prepare_hc_projection_packed16_proofs import base
from test_hc_projection_inner_unroll import render


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class HCProjectionPacked16Tests(unittest.TestCase):
    def test_renderer_default_normalization_and_fold(self):
        program = program_parts(SOURCE.read_text())[2]
        transformed = transform_program(program)
        for shared, shared_scale in ((False, False), (False, True), (True, False), (True, True)):
            original = render(program, 0, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(render(transformed, 0, shared, shared_scale, HELPER, FLAG), original)
            candidate = render(transformed, 1, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(candidate.split("let lane =")[0], original.split("let lane =")[0])
            self.assertEqual(candidate[candidate.index("  if (col < uniforms.prefix)"):],
                             original[original.index("  if (col < uniforms.prefix)"):])
            self.assertEqual(candidate.count("workgroupBarrier();"), original.count("workgroupBarrier();"))
            self.assertIn("sum += vec2<f16>(dot(x0a, w0a) + dot(x1a, w1a), dot(x0b, w0b) + dot(x1b, w1b));", candidate)
            self.assertIn("total += sum;", candidate)
            self.assertIn("value += partials[column * 32u + i]", candidate)
            self.assertNotIn("return;", candidate)
            for suffix in ("a", "b"):
                self.assertIn(f'let packed{suffix} = b[col * 320u + block{suffix}]', candidate)

    def test_exact_stream_mapping_and_padded_columns(self):
        mapping = {}
        for physical in range(128):
            column, lane = divmod(physical, 16)
            for component in range(2):
                logical = column * 32 + lane + component * 16
                self.assertNotIn(logical, mapping)
                mapping[logical] = (physical, component)
                self.assertEqual([tile * 32 + lane + component * 16 for tile in range(10)],
                                 [tile * 32 + logical % 32 for tile in range(10)])
        self.assertEqual(sorted(mapping), list(range(256)))
        for group in range(41):
            columns = [group * 8 + physical // 16 for physical in range(128)]
            self.assertEqual(sorted(set(columns)), list(range(group * 8, group * 8 + 8)))
        for column in range(8):
            self.assertEqual(mapping[column * 32], (column * 16, 0))

    def test_surrounding_source_and_proof_wiring(self):
        source = SOURCE.read_text()
        start, end, _ = program_parts(source)
        candidate = transform_source(source)
        self.assertEqual(source[end:], candidate[candidate.index("class HCPostNormProjectionF16"):])
        self.assertEqual(source[:start], candidate[len("#include <cstdlib>\n#include <fstream>\n"):
                                                   candidate.index(HELPER)])
        self.assertIn("value == 0 || value == 1", candidate)
        for kind in ("native", "model", "screen"):
            expected = base((ROOT / f"run_hc_projection_lane8_{kind}.py").read_text())
            actual = (ROOT / f"run_hc_projection_packed16_{kind}.py").read_text()
            self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))


if __name__ == "__main__":
    unittest.main()
