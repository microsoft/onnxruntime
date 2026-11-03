import copy
import unittest
from unittest.mock import patch

from run_qsa_prefill_score16_standard import FLAG, OLD_PROVIDER, screen_gate, standard_command, verify_case


class PrefillStandardTests(unittest.TestCase):
    def screens(self):
        rows = [dict(context=context, full_validation=full, provider_sha256="new",
                     baseline_provider_sha256="old", graph_sha256="graph", config_sha256="config",
                     all_original_sequences_match=True, repeated_sequences_match=True, zero_major_faults=True,
                     original_sequence_comparisons=288, output_tokens=36, order=[0, 1, 1, 0], measured_requests=4,
                     status="paired_ttft_win_qualified", ttft_win_screen_qualified=True,
                     all_pairs_favor_candidate=True, geometric_mean_matched_ttft_ratio=1.1,
                     matched_ttft_reduction_percent=9.09)
                for context in (2048, 8192, 32768) for full in (True, False)]
        rows[1].update(status="exact_sequences_qualified_no_paired_ttft_win",
                       ttft_win_screen_qualified=False, all_pairs_favor_candidate=False,
                       geometric_mean_matched_ttft_ratio=0.996, matched_ttft_reduction_percent=-0.36)
        return rows

    def gate(self, rows):
        return screen_gate(rows, dict(provider_sha256="new", baseline_provider_sha256="old"),
                           dict(provider_sha256="old", graph_sha256="graph", config_sha256="config"))

    def test_all_six_screens_required_and_retained_2k_neutral_is_explicit(self):
        result = self.gate(self.screens())
        self.assertFalse(result["retained_2k_screen_ttft_win"])
        self.assertEqual(result["retained_2k_matched_ttft_reduction_percent"], -0.36)
        for rows in (self.screens()[:-1], self.screens()[:5] + [self.screens()[0]]):
            with self.assertRaises(AssertionError):
                self.gate(rows)

    def test_fault_mismatch_provider_and_long_context_nonwin_rejected(self):
        for field, value in (("zero_major_faults", False), ("all_original_sequences_match", False),
                             ("provider_sha256", "wrong"), ("ttft_win_screen_qualified", False)):
            rows = copy.deepcopy(self.screens())
            rows[-1][field] = value
            with self.assertRaises(AssertionError):
                self.gate(rows)

    def test_standard_command_preserves_length_repetitions_context_and_all_paths(self):
        shell = (f"{OLD_PROVIDER} --contexts 32768 --chunk 1024 --tokens 256 --repetitions 3 "
                 "--webgpu-validation-mode full /old.json /old-stats.jsonl /old.log")
        original = ["bash", "-lc", shell]
        for full in (True, False):
            result = standard_command(original, "old", "new", "/home/sayanshaw/repos/new/provider.so", full)
            self.assertEqual(original[-1], shell)
            self.assertEqual(result[-1].count("new"), 4)
            self.assertNotIn("/old", result[-1])
            self.assertIn("--contexts 32768 --chunk 1024 --tokens 256 --repetitions 3", result[-1])
            self.assertIn(f"export {FLAG}=1;", result[-1])
            self.assertEqual("--webgpu-validation-mode full" in result[-1], full)

    def test_inactive_selectors_accept_unset_or_explicit_zero_but_not_enabled(self):
        flags = {FLAG: "1", "ORT_WEBGPU_QMOE_PREFILL1024": "1",
                 "ORT_WEBGPU_DSA_QUERY_CACHE": "1", "ORT_WEBGPU_DENSE_BLOCK32": "1",
                 "ORT_WEBGPU_DENSE_DIRECT_A": "0", "ORT_WEBGPU_DENSE_PRIVATE_PARTIALS": "0"}
        with patch("run_qsa_prefill_score16_standard.verify_standard", return_value={"checked": True}):
            for disabled in (None, "0"):
                flags["ORT_WEBGPU_DENSE_DIRECT_A"] = disabled
                self.assertEqual(verify_case({"load_experiment_flags": flags}, {}, {}, {}, [], 2048, True),
                                 {"checked": True})
            for key in ("ORT_WEBGPU_DENSE_DIRECT_A", "ORT_WEBGPU_DENSE_PRIVATE_PARTIALS"):
                for bad in ("1", "2", ""):
                    invalid = dict(flags, **{key: bad})
                    with self.assertRaises(AssertionError):
                        verify_case({"load_experiment_flags": invalid}, {}, {}, {}, [], 2048, True)


if __name__ == "__main__":
    unittest.main()
