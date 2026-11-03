import ast
import json
from pathlib import Path
import unittest

from build_hc_projection_inner_unroll import program_parts
from build_hc_rms_vector_layout import FLAG, HELPER, transform_program, transform_source
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")
ROOT = Path(__file__).parent


class HCRmsVectorLayoutTests(unittest.TestCase):
    def test_actual_renderer_preserves_each_stream_tree_and_default(self):
        program = program_parts(SOURCE.read_text())[2]
        candidate_program = transform_program(program)
        for shared, shared_scale in ((False, False), (False, True), (True, False), (True, True)):
            original = render(program, 0, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(render(candidate_program, 0, shared, shared_scale, HELPER, FLAG), original)
            candidate = render(candidate_program, 1, shared, shared_scale, HELPER, FLAG)
            writes = "".join(f"sums[{stream * 256}u + local_idx] = sum{stream};\n" for stream in range(4))
            adds = "".join(f"sums[{stream * 256}u + local_idx] += sums["
                           f"{stream * 256}u + local_idx + width];\n" for stream in range(4))
            expected = original.replace("sums: array<f32, 1024>", "sums: array<vec4<f32>, 256>")
            expected = expected.replace(writes, "sums[local_idx] = vec4<f32>(sum0, sum1, sum2, sum3);\n")
            expected = expected.replace(adds, "sums[local_idx] += sums[local_idx + width];\n")
            for stream, component in enumerate("xyzw"):
                expected = expected.replace(f"inverseSqrt(sums[{stream * 256}u]",
                                            f"inverseSqrt(sums[0u].{component}")
            self.assertEqual(candidate, expected)
            self.assertEqual(candidate.count("workgroupBarrier();"), original.count("workgroupBarrier();"))

    def test_all_component_addresses_and_tree_edges(self):
        self.assertEqual(256 * 16, 1024 * 4)
        for stream in range(4):
            original = [(stream * 256 + lane) * 4 for lane in range(256)]
            vector = [lane * 16 + stream * 4 for lane in range(256)]
            self.assertEqual(len(set(vector)), 256)
            for width in (128, 64, 32, 16, 8, 4, 2, 1):
                for lane in range(width):
                    self.assertEqual((original[lane + width] - original[lane]) // 4, width)
                    self.assertEqual((vector[lane + width] - vector[lane]) // 16, width)
        addresses = [lane * 16 + stream * 4 for lane in range(256) for stream in range(4)]
        self.assertEqual(sorted(addresses), list(range(0, 4096, 4)))

    def test_projection_compute_fallbacks_and_admission_unchanged(self):
        source = SOURCE.read_text()
        start, end, program = program_parts(source)
        candidate = transform_source(source)
        self.assertEqual(source[end:], candidate[candidate.index("class HCPostNormProjectionF16"):])
        self.assertEqual(source[:start], candidate[len("#include <cstdlib>\n#include <fstream>\n"):
                                                   candidate.index(HELPER)])
        self.assertIn(program[program.index('    code << "for (var h = local_idx;',
                                           program.index('"let inv_rms"')):], candidate)
        self.assertIn("static const int mode", candidate)
        self.assertIn("value == 0 || value == 1", candidate)

    def test_generated_native_coverage_and_long_context_wiring(self):
        from prepare_hc_rms_vector_layout_proofs import base
        from run_hc_rms_vector_layout_model import DIRECTORY, TEMPLATE, model_command, verify_coverage

        for kind in ("native", "model", "screen"):
            expected = base((ROOT / f"run_hc_rms_tree_unroll_{kind}.py").read_text())
            actual = (ROOT / f"run_hc_rms_vector_layout_{kind}.py").read_text()
            self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "unroll"):
                    command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                            ROOT / "hc-rms-vector-command-test", context, full, mode, "standard")
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
