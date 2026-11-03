import copy
import unittest

from qualify_qmoe_prefill_rows4 import verify_screen
from qualify_qsa_prefill_score16_screens import ttft_metrics


class Rows4QualificationTests(unittest.TestCase):
    def test_only_complete_positive_exact_screens_are_admitted(self):
        native = {"provider_sha256": "provider"}
        values = [4.0, 3.6, 3.6, 4.0]
        proof = dict(status="paired_ttft_win_qualified", context=32768, full_validation=True,
                     provider_sha256="provider", all_original_sequences_match=True,
                     repeated_sequences_match=True, zero_major_faults=True,
                     original_sequence_comparisons=288, output_tokens=36, order=[0, 1, 1, 0],
                     trials=[{"ttft_seconds": value} for value in values], **ttft_metrics(values))
        verify_screen(proof, native, 32768, True)
        for field, value in (("full_validation", False), ("zero_major_faults", False),
                             ("repeated_sequences_match", False), ("matched_pairs", 1),
                             ("geometric_mean_matched_ttft_ratio", 1.5)):
            invalid = copy.deepcopy(proof)
            invalid[field] = value
            with self.assertRaises(AssertionError):
                verify_screen(invalid, native, 32768, True)


if __name__ == "__main__":
    unittest.main()
