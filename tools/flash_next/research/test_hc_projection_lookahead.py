import ast
from pathlib import Path
import unittest

from build_hc_projection_inner_unroll import program_parts
from build_hc_projection_lookahead import FLAG, HELPER, transform_program, transform_source
from prepare_hc_projection_lookahead_proofs import base
from test_hc_projection_inner_unroll import render


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class LookaheadTests(unittest.TestCase):
    def test_actual_renderer_default_and_arithmetic(self):
        program = program_parts(SOURCE.read_text())[2]
        transformed = transform_program(program)
        for shared, shared_scale in ((False, False), (False, True), (True, False), (True, True)):
            original = render(program, 0, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(render(transformed, 0, shared, shared_scale, HELPER, FLAG), original)
            candidate = render(transformed, 1, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(candidate.split("let lane =")[0], original.split("let lane =")[0])
            self.assertEqual(candidate[candidate.index("partials[local_idx] = total;"):],
                             original[original.index("partials[local_idx] = total;"):])
            self.assertEqual(candidate.count("sum += dot(x0, w0) + dot(x1, w1);"), 1)
            self.assertEqual(candidate.count("for (var tile = 0u; tile < 10u; tile++)"), 1)
            self.assertEqual(candidate.count("for (var i = 0u; i < 4u; i++)"), 1)
            self.assertEqual(candidate.count("workgroupBarrier();"), original.count("workgroupBarrier();"))
            self.assertLess(candidate.index("next_packed = b["), candidate.index("var sum = f16(0)"))
            self.assertLess(candidate.index("total += sum;"), candidate.index("packed = next_packed;"))
            self.assertIn("if (tile < 9u)", candidate)

    def test_load_mapping_and_surrounding_source(self):
        for lane in range(32):
            block = lane
            used = []
            for tile in range(10):
                used.append(block)
                if tile < 9:
                    block = tile * 32 + lane + 32
                    self.assertLess(block, 320)
            self.assertEqual(used, [tile * 32 + lane for tile in range(10)])
        source = SOURCE.read_text()
        start, end, _ = program_parts(source)
        candidate = transform_source(source)
        self.assertEqual(source[end:], candidate[candidate.index("class HCPostNormProjectionF16"):])
        self.assertEqual(source[:start], candidate[len("#include <cstdlib>\n#include <fstream>\n"):
                                                   candidate.index(HELPER)])
        self.assertIn(r'lookahead_tiles\":1', HELPER)

    def test_generated_proof_wiring(self):
        for kind in ("native", "model", "screen"):
            expected = base((ROOT / f"run_hc_projection_tile_unroll_{kind}.py").read_text())
            actual = (ROOT / f"run_hc_projection_lookahead_{kind}.py").read_text()
            self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))
        model = (ROOT / "run_hc_projection_lookahead_model.py").read_text()
        self.assertIn('row["lookahead_tiles"] == 1', model)
        self.assertIn('choices=("control", "lookahead")', model)
        screen = (ROOT / "run_hc_projection_lookahead_screen.py").read_text()
        self.assertIn("choices=(2048, 8192, 32768)", screen)
        self.assertIn("initial_lookahead_decode_gain_both_modes", screen)


if __name__ == "__main__":
    unittest.main()
