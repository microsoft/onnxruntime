import unittest

from run_prefill_dense_model import verify_windows
from qsa_rank_decode_windows import WINDOW_PATTERN


class PairedProtocolTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for repetition in range(4):
            for window in range(8):
                mode = WINDOW_PATTERN[window] ^ (repetition % 2)
                tps = 80 if mode else 64
                rows.append(dict(context=2048, repetition=repetition, window=window,
                    dense_bitcast_unpack_flag=mode, decode_tokens=32, decode_seconds=32 / tps,
                    decode_tps=tps, logical_sequence_tokens_before=2080 + window * 32))
        return dict(window_samples=rows, window_summary={"2048": dict(matched_window_pairs=16,
            all_request_output_tokens_match=True, geometric_mean_matched_window_ratio=1.25)})

    def test_exact_complementary_pairs(self):
        self.assertEqual(verify_windows(self.fixture(), 2048)["matched_window_pairs"], 16)

    def test_hc_selector_is_independently_verified(self):
        data = self.fixture()
        for row in data["window_samples"]:
            row["hc_post_norm_bitcast_flag"] = row.pop("dense_bitcast_unpack_flag")
        self.assertEqual(verify_windows(data, 2048, "hc_post_norm_bitcast")["matched_window_pairs"], 16)
        data["window_samples"][0]["hc_post_norm_bitcast_flag"] = 1
        with self.assertRaises(AssertionError):
            verify_windows(data, 2048, "hc_post_norm_bitcast")

    def test_rejects_duplicate_windows(self):
        data = self.fixture()
        data["window_samples"][-1] = data["window_samples"][0]
        with self.assertRaises(AssertionError):
            verify_windows(data, 2048)

    def test_rejects_wrong_mode(self):
        data = self.fixture()
        data["window_samples"][0]["dense_bitcast_unpack_flag"] = 1
        with self.assertRaises(AssertionError):
            verify_windows(data, 2048)

    def test_rejects_misreported_gain(self):
        data = self.fixture()
        data["window_summary"]["2048"]["geometric_mean_matched_window_ratio"] = 1.3
        with self.assertRaises(AssertionError):
            verify_windows(data, 2048)


if __name__ == "__main__":
    unittest.main()
