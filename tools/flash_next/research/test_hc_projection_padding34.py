import ast
import json
from pathlib import Path
import unittest

from build_hc_projection_inner_unroll import program_parts
from build_hc_projection_padding34 import FLAG, HELPER, transform_program, transform_source
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")
ROOT = Path(__file__).parent


class HCProjectionPadding34Tests(unittest.TestCase):
    def test_real_renderer_only_changes_shared_layout(self):
        program = program_parts(SOURCE.read_text())[2]
        candidate_program = transform_program(program)
        for shared, shared_scale in ((False, False), (False, True), (True, False), (True, True)):
            original = render(program, 0, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(render(candidate_program, 0, shared, shared_scale, HELPER, FLAG), original)
            candidate = render(candidate_program, 1, shared, shared_scale, HELPER, FLAG)
            self.assertEqual(candidate, original.replace("array<f16, 11520>", "array<f16, 10880>")
                             .replace("* 36u", "* 34u"))
            self.assertEqual(original.count("* 36u"), 13)

    def test_mapping_and_projection_bank_words(self):
        for stride in (34, 36):
            addresses = [logical // 32 * stride + logical % 32 for logical in range(10240)]
            self.assertEqual(len(set(addresses)), 10240)
            self.assertLess(max(addresses), 320 * stride)
            for tile in range(10):
                for lane in range(32):
                    block = tile * 32 + lane
                    for component in range(4):
                        for element in range(8):
                            logical = block * 32 + component * 8 + element
                            self.assertEqual(addresses[logical], block * stride + component * 8 + element)
                for component in range(4):
                    for element in range(8):
                        banks = [((tile * 32 + lane) * stride + component * 8 + element) // 2 % 32
                                 for lane in range(32)]
                        self.assertEqual(len(set(banks)), 32 if stride == 34 else 16)

    def test_original_compute_fallbacks_and_admission_preserved(self):
        source = SOURCE.read_text()
        start, end, _ = program_parts(source)
        candidate = transform_source(source)
        self.assertEqual(source[end:], candidate[candidate.index("class HCPostNormProjectionF16"):])
        self.assertEqual(source[:start], candidate[len("#include <cstdlib>\n#include <fstream>\n"):
                                                   candidate.index(HELPER)])
        self.assertIn('"HCPostNormProjectionPadding34F16" : "HCPostNormProjectionF16"', candidate)
        self.assertIn("static const int mode", candidate)
        self.assertIn("value == 0 || value == 1", candidate)

    def test_generated_proofs_and_long_context_wiring(self):
        from prepare_hc_projection_padding34_proofs import base, model_source, native_source
        from run_hc_projection_padding34_model import DIRECTORY, TEMPLATE, model_command, verify_coverage

        for old, new, transform in (
            ("native", "native", native_source), ("model", "model", model_source), ("screen", "screen", base)):
            expected = transform((ROOT / f"run_hc_rms_tree_unroll_{old}.py").read_text())
            actual = (ROOT / f"run_hc_projection_padding34_{new}.py").read_text()
            self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(actual)))
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "unroll"):
                    command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                            ROOT / "hc-padding34-command-test", context, full, mode, "standard")
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
