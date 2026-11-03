import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from qualify_qmoe_prefill_rows4_standard import verify_proof
from run_qmoe_prefill_rows4_standard import (
    FOLD_FLAG, GROUP_FLAG, OLD_PROVIDER, TEMPLATE, standard_command, verify_result,
)


class Rows4StandardTests(unittest.TestCase):
    def setUp(self):
        self.command = ["docker", "bash", "-lc",
                        f"export ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE={TEMPLATE}-stats.jsonl; "
                        f"export ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE={TEMPLATE}-static-stats.jsonl; "
                        f"python benchmark_webgpu_int4.py --plugin {OLD_PROVIDER} "
                        f"--output {TEMPLATE}.json --contexts 2048 --chunk 1024 "
                        "--tokens 256 --repetitions 3 --webgpu-validation-mode full"]

    def test_natural_request_shape_and_both_validation_modes(self):
        for context in (2048, 8192, 32768):
            for full in (True, False):
                result = standard_command(self.command, "/home/sayanshaw/repos/rows4/provider.so",
                                          Path("/evidence/fresh"), context, full)
                self.assertIn("--tokens 256 --repetitions 3", result[-1])
                self.assertIn(f"--contexts {context}", result[-1])
                self.assertIn("/workspace/rows4/provider.so", result[-1])
                self.assertIn("fresh-stats.jsonl", result[-1])
                self.assertIn("fresh-static-stats.jsonl", result[-1])
                self.assertIn("fresh.json", result[-1])
                self.assertIn(f"export {GROUP_FLAG}=1 {FOLD_FLAG}=0;", result[-1])
                self.assertEqual(" --webgpu-validation-mode full" in result[-1], full)
                self.assertNotIn("--profile-", result[-1])
                self.assertNotIn("--compare-", result[-1])
        self.assertIn(OLD_PROVIDER, self.command[-1])

    def test_invalid_or_ambiguous_templates_are_rejected(self):
        for extra in (f" export {GROUP_FLAG}=0", f" export {FOLD_FLAG}=1",
                      " --contexts 2048", " --compare-fc2-subgroup-fold-windows",
                      " --profile-decode-after 35", f" {TEMPLATE}"):
            with self.assertRaises(AssertionError):
                standard_command([*self.command[:-1], self.command[-1] + extra],
                                 Path("/provider"), Path("/prefix"), 2048, True)
        with self.assertRaises(AssertionError):
            standard_command(self.command, Path("/provider"), Path("/prefix"), 1024, True)

    def test_selector_checks_do_not_bypass_existing_standard_verifier(self):
        flags = {flag: "1" for flag in (GROUP_FLAG, "ORT_WEBGPU_DSA_QUERY_CACHE",
                                       "ORT_WEBGPU_DENSE_BLOCK32", "ORT_WEBGPU_QSA_PREFILL_SCORE16")}
        flags[FOLD_FLAG] = "0"
        raw = {"load_experiment_flags": flags}
        with patch("run_qmoe_prefill_rows4_standard.verify_standard", return_value={"verified": True}) as standard:
            with patch("run_qmoe_prefill_rows4_standard.verify_prefill_flags") as prefill:
                self.assertEqual(verify_result(raw, {}, {}, {}, [], 32768, False), {"verified": True})
                standard.assert_called_once_with(raw, {}, {}, {}, [], 32768, False, chunk=1024)
                prefill.assert_called_once_with(raw, 1024)
                for flag in flags:
                    invalid = copy.deepcopy(raw)
                    invalid["load_experiment_flags"][flag] = "1" if flag == FOLD_FLAG else "0"
                    with self.assertRaises(AssertionError):
                        verify_result(invalid, {}, {}, {}, [], 32768, False)

    def test_aggregate_requires_complete_exact_natural_cases(self):
        native = {"provider_sha256": "provider"}
        parent = dict(**native, graph_sha256="graph", config_sha256="config")
        result = {"median_decode_tps": 80.0}
        proof = dict(status="original_sequences_qualified", phase="standard", context=32768,
                     full_validation=False, **parent, original_output_prefixes_match=True,
                     zero_major_faults=True, chunk=1024, output_tokens=256, measured_repetitions=3,
                     excluded_warmup_requests=1, original_sequence_comparisons=1024, result=result,
                     fixed_grouped_prefill_selector=1, rejected_fc2_fold_selector=0,
                     comparison_kind="standalone_standard_not_isolated_causal_pair",
                     quality_qualified=False, headline_eligible=False)
        verify_proof(proof, result, native, parent, 32768, False)
        for key, value in (("phase", "windows"), ("full_validation", True),
                           ("provider_sha256", "different"), ("zero_major_faults", False),
                           ("output_tokens", 36), ("measured_repetitions", 4),
                           ("original_sequence_comparisons", 288), ("rejected_fc2_fold_selector", 1),
                           ("headline_eligible", True), ("result", {"median_decode_tps": 100.0})):
            invalid = copy.deepcopy(proof)
            invalid[key] = value
            with self.assertRaises(AssertionError):
                verify_proof(invalid, result, native, parent, 32768, False)


if __name__ == "__main__":
    unittest.main()
