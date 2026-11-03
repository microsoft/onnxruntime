import ast
import json
from pathlib import Path
import unittest

from build_hc_projection_inner_unroll import program_parts
from build_hc_projection_lane8 import FLAG, HELPER, transform_program, transform_source
from test_hc_projection_inner_unroll import render


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class HCProjectionLane8Tests(unittest.TestCase):
    def test_actual_renderer_preserves_default_and_all_other_code(self):
        program = program_parts(SOURCE.read_text())[2]
        candidate_program = transform_program(program)
        for shared, shared_scale in ((False, False), (False, True), (True, False), (True, True)):
            original = render(program, 0, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(render(candidate_program, 0, shared, shared_scale, HELPER, FLAG), original)
            candidate = render(candidate_program, 1, shared, shared_scale, HELPER, FLAG)
            expected = original.replace(
                "let lane = local_idx % 32u;\nlet col = workgroup_idx * 8u + local_idx / 32u;\n",
                "let lane = ((local_idx % 128u) / 32u) * 8u + local_idx % 8u;\n"
                "let column = (local_idx / 128u) * 4u + (local_idx % 32u) / 8u;\n"
                "let col = workgroup_idx * 8u + column;\n"
                "let logical_idx = column * 32u + lane;\n")
            expected = expected.replace("partials[local_idx] = total", "partials[logical_idx] = total")
            expected = expected.replace("partials[local_idx + i]", "partials[logical_idx + i]")
            self.assertEqual(candidate, expected)
            self.assertEqual(candidate.count("workgroupBarrier();"), original.count("workgroupBarrier();"))

    def test_mapping_bijection_warp_geometry_and_ordered_fold(self):
        mapping = {}
        for physical in range(256):
            lane = ((physical % 128) // 32) * 8 + physical % 8
            column = (physical // 128) * 4 + (physical % 32) // 8
            logical = column * 32 + lane
            self.assertTrue(0 <= lane < 32 and 0 <= column < 8)
            self.assertNotIn(logical, mapping)
            mapping[logical] = physical
        self.assertEqual(sorted(mapping), list(range(256)))
        for warp in range(8):
            cells = [(logical // 32, logical % 32) for logical, physical in mapping.items()
                     if physical // 32 == warp]
            self.assertEqual(len({col for col, _ in cells}), 4)
            for col in {col for col, _ in cells}:
                lanes = sorted(lane for column, lane in cells if column == col)
                self.assertEqual(lanes, list(range((warp % 4) * 8, (warp % 4 + 1) * 8)))
        for column in range(8):
            writer = mapping[column * 32]
            self.assertEqual(((writer % 128) // 32) * 8 + writer % 8, 0)
            self.assertEqual([column * 32 + i for i in range(32)],
                             list(range(column * 32, (column + 1) * 32)))

    def test_surrounding_compute_and_fallbacks_unchanged(self):
        source = SOURCE.read_text()
        start, end, _ = program_parts(source)
        candidate = transform_source(source)
        self.assertEqual(source[end:], candidate[candidate.index("class HCPostNormProjectionF16"):])
        self.assertEqual(source[:start], candidate[len("#include <cstdlib>\n#include <fstream>\n"):
                                                   candidate.index(HELPER)])
        self.assertIn("static const int mode", candidate)
        self.assertIn("value == 0 || value == 1", candidate)

    def test_proof_generation_and_2k_8k_32k_wiring(self):
        from prepare_hc_projection_lane8_proofs import base
        from run_hc_projection_lane8_model import DIRECTORY, TEMPLATE, model_command, verify_coverage

        for kind in ("native", "model", "screen"):
            expected = base((ROOT / f"run_hc_rms_vector_layout_{kind}.py").read_text())
            actual = (ROOT / f"run_hc_projection_lane8_{kind}.py").read_text()
            self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "unroll"):
                    command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                            ROOT / "hc-projection-lane8-command-test", context, full, mode, "standard")
                    self.assertIn(f"--contexts {context}", command[-1])
                    self.assertIn("--tokens 256 --repetitions 3", command[-1])
                    self.assertIn(f"export {FLAG}={int(mode == 'unroll')}; ", command[-1])
                    self.assertNotIn(f"export {FLAG}_STATS_FILE=", command[-1])
        rows = [dict(operator=name, hits=69, binding_hits=52, rebinds=17)
                for name, count in (("MatMulNBitsExact", 171), ("HCPostNormProjectionF16", 94))
                for _ in range(count)]
        self.assertEqual(verify_coverage(rows)["actual_hc_replay_instances"], 94)
        with self.assertRaises(AssertionError):
            verify_coverage(rows[:-1])


if __name__ == "__main__":
    unittest.main()
