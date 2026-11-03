import copy
import math
import unittest

from qualify_row4_cpu_policy_contexts import candidates_above_target, verify_screen


def screen(context=2048, repeat=False, ratios=(1.01, 1.02, 1.03), ttft=1.0):
    ratio = math.exp(sum(math.log(value) for value in ratios) / 3)
    valid = ratio > 1 and all(value > 1 for value in ratios) and ttft <= 1.02
    row = dict(before=dict(median_decode_tps=78.0, mean_ttft_seconds=3.5),
               candidate=dict(median_decode_tps=79.0, mean_ttft_seconds=3.5),
               after=dict(median_decode_tps=78.0, mean_ttft_seconds=3.5),
               comparison=dict(request_ratios=list(ratios), matched_request_decode_ratio=ratio,
                               matched_request_ttft_ratio=ttft, positive_requests=sum(value > 1 for value in ratios),
                               supported=valid))
    return dict(context=context, repeat=repeat,
                status=(("repeated_" if repeat else "initial_") + "default_policy_decode_gain_both_modes"
                        if valid else "no_supported_decode_gain"),
                default_worker_policy_preserved_in_controls=True, original_output_prefixes_match=True,
                zero_major_faults=True, process_only_affinity=True, memory_policy_unchanged=True,
                rows=[dict(copy.deepcopy(row), full_validation=full) for full in (True, False)])


class CpuPolicyContextTests(unittest.TestCase):
    def test_positive_and_honest_unsupported_screens(self):
        self.assertTrue(verify_screen(screen(), 2048, False))
        self.assertTrue(verify_screen(screen(32768, True), 32768, True))
        self.assertFalse(verify_screen(screen(ratios=(0.99, 1.02, 1.03)), 2048, False))
        self.assertFalse(verify_screen(screen(ttft=1.021), 2048, False))

    def test_misreported_support_and_geometry_rejected(self):
        proof = screen()
        for mutate in (lambda value: value.update(context=8192),
                       lambda value: value.update(default_worker_policy_preserved_in_controls=False),
                       lambda value: value["rows"][0]["comparison"].update(positive_requests=2),
                       lambda value: value["rows"][0]["comparison"].update(matched_request_decode_ratio=1.2)):
            changed = copy.deepcopy(proof)
            mutate(changed)
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                verify_screen(changed, 2048, False)

    def test_every_context_and_both_modes_must_exceed_100(self):
        rows = [dict(context=context, fresh=screen(context, True)) for context in (2048, 8192, 32768)]
        self.assertFalse(candidates_above_target(rows))
        for row in rows:
            for mode in row["fresh"]["rows"]:
                mode["candidate"]["median_decode_tps"] = 100.001
        self.assertTrue(candidates_above_target(rows))
        rows[2]["fresh"]["rows"][1]["candidate"]["median_decode_tps"] = 100.0
        self.assertFalse(candidates_above_target(rows))
        self.assertFalse(candidates_above_target(rows[:2]))
        rows[2]["fresh"]["rows"] = []
        self.assertFalse(candidates_above_target(rows))


if __name__ == "__main__":
    unittest.main()
