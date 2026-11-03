import copy
import json
import unittest

from build_qsa_score16 import FLAG
from qsa_rank_decode_windows import WINDOW_PATTERN
from run_qsa_score16_model import verify_paired


class PairedScore16Tests(unittest.TestCase):
    def setUp(self):
        context = 32768
        self.references = [dict(prompt_sha256=str(i), output_token_ids=list(range(256))) for i in range(4)]
        summary = dict(qsa_score160_median_window_tps=1., qsa_score161_median_window_tps=2.,
                       matched_window_pairs=16, all_request_output_tokens_match=True,
                       geometric_mean_matched_window_ratio=2.,
                       per_request=[dict(repetition=i, qsa_score160_median_window_tps=1.,
                                         qsa_score161_median_window_tps=2.) for i in range(4)])
        self.raw = dict(comparison="qsa_score16_decode_windows",
                        window_protocol=dict(experiment_flag=FLAG),
                        window_summary={str(context): summary}, window_samples=[],
                        samples=[dict(context=context, prompt_sha256="1", output_tokens=288,
                                      output_token_ids=list(range(288)), same_prompt_output_matches=True,
                                      process_resource_usage=dict(total=dict(major_page_faults=0)))
                                 for _ in range(4)])
        for repetition in range(4):
            for window in range(8):
                mode = WINDOW_PATTERN[window] ^ (repetition % 2)
                self.raw["window_samples"].append(dict(
                    context=context, repetition=repetition, window=window, qsa_score16_flag=mode,
                    decode_tokens=32, logical_sequence_tokens_before=context + 32 + window * 32,
                    decode_tps=float(mode + 1), decode_seconds=32 / (mode + 1)))

    def test_exact_metrics_and_frozen_prefixes(self):
        self.assertEqual(verify_paired(self.raw, self.references, 32768)["matched_window_pairs"], 16)

    def test_parallel_column_prefix_uses_the_same_independent_checks(self):
        flag = "ORT_WEBGPU_HC_UP_PARALLEL2"
        raw = json.loads(json.dumps(self.raw).replace("qsa_score16", "hc_up_parallel2").replace(FLAG, flag))
        options = dict(prefix="hc_up_parallel2", experiment_flag=flag)
        self.assertEqual(verify_paired(raw, self.references, 32768, **options)["matched_window_pairs"], 16)
        raw["window_summary"]["32768"]["hc_up_parallel21_median_window_tps"] = 3.
        with self.assertRaises(AssertionError):
            verify_paired(raw, self.references, 32768, **options)

    def test_rejects_median_and_ratio_corruption(self):
        for key in ("qsa_score160_median_window_tps", "qsa_score161_median_window_tps",
                    "geometric_mean_matched_window_ratio"):
            raw = copy.deepcopy(self.raw)
            raw["window_summary"]["32768"][key] = 3.
            with self.assertRaises(AssertionError):
                verify_paired(raw, self.references, 32768)
        raw = copy.deepcopy(self.raw)
        raw["window_summary"]["32768"]["per_request"][0]["qsa_score161_median_window_tps"] = 3.
        with self.assertRaises(AssertionError):
            verify_paired(raw, self.references, 32768)

    def test_rejects_faults_and_both_prefix_and_tail_changes(self):
        for token in (0, 256, 287):
            raw = copy.deepcopy(self.raw)
            raw["samples"][1]["output_token_ids"][token] = -1
            with self.assertRaises(AssertionError):
                verify_paired(raw, self.references, 32768)
        raw = copy.deepcopy(self.raw)
        raw["samples"][0]["process_resource_usage"]["total"]["major_page_faults"] = 1
        with self.assertRaises(AssertionError):
            verify_paired(raw, self.references, 32768)


if __name__ == "__main__":
    unittest.main()
