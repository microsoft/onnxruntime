import ast
import json
from pathlib import Path
import unittest

from build_hc_projection_inner_unroll import program_parts
from build_hc_rms_tree_unroll import FLAG, HELPER, transform_program, transform_source
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class HCRmsTreeUnrollTests(unittest.TestCase):
    def test_original_projection_compute_and_fallbacks_preserved(self):
        source = SOURCE.read_text()
        _, end, _ = program_parts(source)
        candidate = transform_source(source)
        self.assertEqual(source[end:], candidate[candidate.index("class HCPostNormProjectionF16"):])
        original_projection = source[source.index('         << "let lane = local_idx % 32u;'):end]
        self.assertIn(original_projection, candidate)
        self.assertIn('"HCPostNormProjectionRmsTreeUnrollF16" : "HCPostNormProjectionF16"', candidate)
        self.assertIn("static const int mode", candidate)
        self.assertIn("value == 0 || value == 1", candidate)

    def test_real_renderer_preserves_all_eight_reduction_levels(self):
        program = program_parts(SOURCE.read_text())[2]
        candidate_program = transform_program(program)
        for shared, shared_scale in ((False, False), (False, True), (True, False), (True, True)):
            original = render(program, 0, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(render(candidate_program, 0, shared, shared_scale, HELPER, FLAG), original)
            candidate = render(candidate_program, 1, shared, shared_scale, HELPER, FLAG)
            start = original.index("for (var width = 128u;")
            body_start = original.index("if (local_idx < width) {", start)
            end = original.index("let inv_rms0", body_start)
            body = original[body_start:end - len("}\n")]
            expected = "".join(f"{{\nlet width = {width}u;\n" + body + "}\n"
                               for width in (128, 64, 32, 16, 8, 4, 2, 1))
            self.assertEqual(candidate, original[:start] + expected + original[end:])
            self.assertEqual(body.count("workgroupBarrier();"), 1)
            self.assertEqual(body.count("sums["), 8)

    def test_native_entrypoint_and_model_command_wiring(self):
        from prepare_hc_rms_tree_unroll_proofs import ROOT, native_source
        from run_hc_rms_tree_unroll_model import DIRECTORY, TEMPLATE, model_command

        tree = ast.parse(native_source((ROOT / "run_hc_bitcast_native.py").read_text()))
        self.assertEqual(ast.dump(tree), ast.dump(ast.parse(
            (ROOT / "run_hc_rms_tree_unroll_native.py").read_text())))
        entrypoints = [node for node in tree.body if isinstance(node, ast.If)]
        self.assertEqual(len(entrypoints), 1)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        self.assertLess(main.lineno, entrypoints[0].lineno)
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "unroll"):
                    command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                            ROOT / "hc-rms-command-test", context, full, mode, "standard")
                    self.assertIn(f"--contexts {context}", command[-1])
                    self.assertIn("--tokens 256 --repetitions 3", command[-1])
                    self.assertIn(f"export {FLAG}={int(mode == 'unroll')}; ", command[-1])
                    self.assertIn("unset ORT_WEBGPU_HC_RMS_TREE_UNROLL_STATS_FILE; ", command[-1])
                    self.assertNotIn("export ORT_WEBGPU_HC_RMS_TREE_UNROLL_STATS_FILE=", command[-1])
                    self.assertNotIn("--profile-prefix", command[-1])

    def test_model_coverage_requires_all_hc_nodes_and_binding_hits(self):
        from run_hc_rms_tree_unroll_model import verify_coverage

        def row(operator):
            return dict(operator=operator, hits=69, binding_hits=52, rebinds=17)

        dense = [row("MatMulNBitsExact") for _ in range(171)]
        hc = [row("HCPostNormProjectionF16") for _ in range(94)]
        proof = verify_coverage(dense + hc)
        self.assertEqual(proof["actual_hc_replay_instances"], 94)
        self.assertEqual(proof["hc_replay_hits"], 6486)
        with self.assertRaises(AssertionError):
            verify_coverage(dense + hc[:-1])
        hc[0].update(binding_hits=0, rebinds=69)
        with self.assertRaises(AssertionError):
            verify_coverage(dense + hc)


if __name__ == "__main__":
    unittest.main()
