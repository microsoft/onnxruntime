import math
import unittest

from qualify_qsa_prefill_score16_screens import ttft_metrics


class PairedPrefillTtftTests(unittest.TestCase):
    def test_abba_pairs_and_exact_reduction(self):
        result = ttft_metrics([10, 8, 4, 5])
        self.assertEqual(result["baseline_median_ttft_seconds"], 7.5)
        self.assertEqual(result["candidate_median_ttft_seconds"], 6)
        self.assertEqual(result["baseline_over_candidate_pair_ratios"], [1.25, 1.25])
        self.assertEqual(result["geometric_mean_matched_ttft_ratio"], 1.25)
        self.assertAlmostEqual(result["matched_ttft_reduction_percent"], 20)
        self.assertTrue(result["ttft_win_screen_qualified"])

    def test_neutral_and_mixed_pairs_are_not_wins(self):
        for values in ([10, 10, 10, 10], [20, 10, 10, 9]):
            self.assertFalse(ttft_metrics(values)["ttft_win_screen_qualified"])

    def test_incomplete_and_invalid_timings_rejected(self):
        for values in ([1, 2], [1, 2, 3, 0], [1, 2, 3, -1], [1, math.inf, 1, 1], [1, math.nan, 1, 1]):
            with self.assertRaises(AssertionError):
                ttft_metrics(values)


if __name__ == "__main__":
    unittest.main()
