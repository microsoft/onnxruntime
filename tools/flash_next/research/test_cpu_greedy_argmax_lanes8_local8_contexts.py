import copy
import math
import unittest

from qualify_cpu_greedy_argmax_lanes8_local8_contexts import candidates_above_target, verify_screen


def screen(context=2048, repeat=False):
    rows = []
    for full in (True, False):
        comparison = dict(request_ratios=[1.01, 1.02, 1.03], positive_requests=3,
            matched_request_decode_ratio=math.exp(sum(math.log(v) for v in (1.01, 1.02, 1.03)) / 3),
            matched_request_ttft_ratio=1.01, supported=True)
        rows.append(dict(full_validation=full, comparison=comparison,
            **{phase: dict(median_decode_tps=101.0, mean_ttft_seconds=3.0)
               for phase in ("before", "candidate", "after")}))
    return dict(context=context, repeat=repeat,
        status=("repeated_" if repeat else "initial_") + "cpu_greedy_decode_gain_both_modes",
        original_output_prefixes_match=True, zero_major_faults=True,
        main_affinity_unchanged=True, manual_affinity_override=True,
        fresh_reuses_initial_ending_controls=repeat, rows=rows)


class CpuGreedyLocal8ContextsTests(unittest.TestCase):
    def test_valid_support_and_explicit_rejection(self):
        proof = screen()
        self.assertTrue(verify_screen(proof, 2048, False))
        proof["rows"][0]["comparison"].update(request_ratios=[0.99] * 3, positive_requests=0,
            matched_request_decode_ratio=0.99, supported=False)
        proof["status"] = "no_supported_decode_gain"
        self.assertFalse(verify_screen(proof, 2048, False))

    def test_bad_ratio_policy_or_output_shape_rejected(self):
        proof = screen()
        proof["rows"][0]["comparison"]["matched_request_decode_ratio"] = 1.1
        with self.assertRaises(AssertionError):
            verify_screen(proof, 2048, False)
        for field in ("manual_affinity_override", "original_output_prefixes_match", "zero_major_faults"):
            proof = screen()
            proof[field] = False
            with self.assertRaises(AssertionError):
                verify_screen(proof, 2048, False)
        proof = screen()
        proof["rows"].reverse()
        with self.assertRaises(AssertionError):
            verify_screen(proof, 2048, False)

    def test_strict_all_context_all_mode_target(self):
        rows = [dict(context=context, fresh=screen(context, True))
                for context in (2048, 8192, 32768)]
        self.assertTrue(candidates_above_target(rows))
        for index in range(3):
            for mode in range(2):
                candidate = copy.deepcopy(rows)
                candidate[index]["fresh"]["rows"][mode]["candidate"]["median_decode_tps"] = 100.0
                self.assertFalse(candidates_above_target(candidate))
        self.assertFalse(candidates_above_target(rows[:2]))


if __name__ == "__main__":
    unittest.main()
