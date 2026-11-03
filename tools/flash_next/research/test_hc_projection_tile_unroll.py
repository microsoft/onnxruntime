import ast
from pathlib import Path
import unittest

from build_hc_projection_inner_unroll import program_parts
from build_hc_projection_tile_unroll import FLAG, HELPER, transform_program, transform_source
from prepare_hc_projection_tile_unroll_proofs import base
from test_hc_projection_inner_unroll import render


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class TileUnrollTests(unittest.TestCase):
    def test_actual_renderer_order_and_default(self):
        program = program_parts(SOURCE.read_text())[2]
        candidate_program = transform_program(program)
        for shared, shared_scale in ((False, False), (False, True), (True, False), (True, True)):
            original = render(program, 0, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(render(candidate_program, 0, shared, shared_scale, HELPER, FLAG), original)
            candidate = render(candidate_program, 1, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(candidate.split("let lane =")[0], original.split("let lane =")[0])
            self.assertEqual(candidate[candidate.index("partials[local_idx] = total;"):],
                             original[original.index("partials[local_idx] = total;"):])
            positions = [candidate.index(f"let block = {tile * 32}u + lane;") for tile in range(10)]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(candidate.count("sum += dot(x0, w0) + dot(x1, w1);"), 10)
            self.assertEqual(candidate.count("total += sum;"), 10)
            self.assertEqual(candidate.count("for (var i = 0u; i < 4u; i++)"), 10)
            self.assertEqual(candidate.count("workgroupBarrier();"), original.count("workgroupBarrier();"))
            self.assertNotIn("var tile", candidate)

    def test_surrounding_source_and_indices(self):
        source = SOURCE.read_text()
        start, end, _ = program_parts(source)
        candidate = transform_source(source)
        self.assertEqual(source[end:], candidate[candidate.index("class HCPostNormProjectionF16"):])
        self.assertEqual(source[:start], candidate[len("#include <cstdlib>\n#include <fstream>\n"):
                                                   candidate.index(HELPER)])
        for lane in range(32):
            self.assertEqual([tile * 32 + lane for tile in range(10)], list(range(lane, 320, 32)))
        self.assertIn("value == 0 || value == 1", candidate)

    def test_generated_proof_wiring(self):
        for kind in ("native", "model", "screen"):
            expected = base((ROOT / f"run_hc_projection_packed16_{kind}.py").read_text())
            actual = (ROOT / f"run_hc_projection_tile_unroll_{kind}.py").read_text()
            self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))
        model = (ROOT / "run_hc_projection_tile_unroll_model.py").read_text()
        self.assertIn('row["tile_steps_per_lane"] == 10', model)
        self.assertNotIn("logical_streams_per_thread", model)


if __name__ == "__main__":
    unittest.main()
