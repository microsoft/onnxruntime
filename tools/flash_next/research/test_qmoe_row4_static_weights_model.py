import ast
import json
from pathlib import Path
import unittest

from compare_qmoe_prefill_samples import compare_prefill_samples


ROOT = Path(__file__).parent


def samples(ttft, decode=80):
    return [dict(prompt_sha256=str(index), output_token_ids=[index],
                 ttft_seconds=ttft, decode_tps=decode) for index in range(3)]


class QmoeRow4StaticWeightsModelTests(unittest.TestCase):
    def test_completed_native_modes_reverify_without_gpu_execution(self):
        from build_qmoe_row4_static_weights import PROGRAM
        from run_qmoe_grouped_prefill_native import PROGRAMS
        from run_qmoe_row4_static_weights_native import DIRECTORY, VARIANT, verify_completed_mode

        build = json.loads((DIRECTORY / "build-manifest.json").read_text())
        for mode in (0, 1):
            additional = {**{name: 16 for name in PROGRAMS},
                          PROGRAM if mode == 1 else "MatMulNBitsGroupedRows4F16": 32}
            proof = verify_completed_mode(ROOT / f"{VARIANT}-native-mode{mode}-gpu7-20261101",
                                          mode, build, additional)
            self.assertEqual(proof["finite_bitwise_comparisons"], 300810240)

    def test_invalid_probe_preserves_qualified_environment(self):
        from build_qmoe_row4_static_weights import FLAG
        from run_qmoe_row4_static_weights_native import VARIANT, invalid_command

        qualified = json.loads((ROOT / f"{VARIANT}-native-mode1-gpu7-20261101.guard.json").read_text())["command"]
        result = invalid_command(qualified, "/native", "/provider", "/fixture", "/invalid.log")
        original_head = qualified[-1].split("; exec ", 1)[0]
        self.assertTrue(result.startswith(original_head.replace(f"{FLAG}=1 ", f"{FLAG}=2 ")))
        self.assertIn(f"unset {FLAG}_STATS_FILE; if /native /provider /fixture", result)
        for setting in ("ORT_WEBGPU_QMOE_INT4_PREFILL=1", "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0",
                        "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1", "ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1"):
            self.assertIn(setting, result)

    def test_requires_all_three_ttft_gains_not_decode_gains(self):
        control = samples(10)
        candidate = samples(9.9)
        result = compare_prefill_samples(candidate, control, control)
        self.assertTrue(result["supported"])
        self.assertEqual(result["positive_requests"], 0)
        self.assertEqual(result["positive_ttft_requests"], 3)
        candidate[0]["ttft_seconds"] = 10
        self.assertFalse(compare_prefill_samples(candidate, control, control)["supported"])
        candidate[0]["ttft_seconds"] = 10.01
        self.assertFalse(compare_prefill_samples(candidate, control, control)["supported"])

    def test_exact_one_percent_decode_regression_bound(self):
        control = samples(10, 100)
        self.assertTrue(compare_prefill_samples(samples(9, 99), control, control)["supported"])
        self.assertFalse(compare_prefill_samples(samples(9, 98.999), control, control)["supported"])
        self.assertFalse(compare_prefill_samples(samples(10, 101), control, control)["supported"])
        self.assertFalse(compare_prefill_samples(samples(10.01, 101), control, control)["supported"])

    def test_original_prompt_output_and_finite_metric_gates(self):
        control = samples(10)
        for field, value in (("prompt_sha256", "changed"), ("output_token_ids", [99]),
                             ("ttft_seconds", 0), ("decode_tps", float("nan"))):
            candidate = samples(9)
            candidate[0][field] = value
            with self.assertRaises(AssertionError):
                compare_prefill_samples(candidate, control, control)

    def test_generated_model_and_long_context_runtime_wiring(self):
        from build_qmoe_row4_static_weights import FLAG
        from prepare_qmoe_row4_static_weights_model import model_source, screen_source
        from run_qmoe_row4_static_weights_model import DIRECTORY, TEMPLATE, model_command, verify_coverage

        for kind, transform in (("model", model_source), ("screen", screen_source)):
            expected = transform((ROOT / f"run_hc_rms_tree_unroll_{kind}.py").read_text())
            actual = (ROOT / f"run_qmoe_row4_static_weights_{kind}.py").read_text()
            self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "unroll"):
                    command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                            ROOT / "row4-static-command-test", context, full, mode, "standard")
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
