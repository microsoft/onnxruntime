import copy
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import qualify_gdn_tree32 as qualification


def rows(ratio=1.005, wins=3):
    return [{"context": context, "full_validation": full, "result": {
        "geometric_mean_matched_window_ratio": ratio, "matched_window_pairs": 16,
        "all_request_output_tokens_match": True,
        "per_request": [
            {"repetition": i, "gdn_tree320_median_window_tps": 70,
             "gdn_tree321_median_window_tps": 70.4 if i < wins else 69.9} for i in range(4)]}}
        for context in qualification.CONTEXTS for full in (True, False)]


class GdnQualificationTests(unittest.TestCase):
    def test_positive_screen_requires_every_context_mode_and_three_request_wins(self):
        self.assertTrue(qualification.controls_supported(rows()))
        for ratio, wins in ((1.0, 4), (0.99, 4), (1.01, 2)):
            self.assertFalse(qualification.controls_supported(rows(ratio, wins)))
        cases = rows()
        cases[-1]["result"]["geometric_mean_matched_window_ratio"] = 0.99
        self.assertFalse(qualification.controls_supported(cases))
        with self.assertRaises(AssertionError):
            qualification.controls_supported(rows()[:-1])

    def test_invalid_numeric_or_protocol_data_fails_explicitly(self):
        for value in (float("nan"), float("inf"), 0):
            cases = rows()
            cases[-1]["result"]["geometric_mean_matched_window_ratio"] = value
            with self.assertRaises(AssertionError):
                qualification.controls_supported(cases)
        for key, value in (("matched_window_pairs", 15), ("all_request_output_tokens_match", False)):
            cases = rows()
            cases[0]["result"][key] = value
            with self.assertRaises(AssertionError):
                qualification.controls_supported(cases)

    def test_initial_real_controls_have_supported_shape_and_gain(self):
        root = Path(__file__).parent
        cases = []
        for context in qualification.CONTEXTS:
            for full in (True, False):
                name = (f"gdn-tree32-v1-chunk1024-gpu7-{'full-' if full else ''}"
                        f"windows{context}-20261031.qualification.json")
                proof = json.loads((root / name).read_text())
                cases.append({"context": context, "full_validation": full, "result": proof["result"]})
        self.assertTrue(qualification.controls_supported(cases))
        original = copy.deepcopy(cases)
        cases[3]["result"]["per_request"][0]["gdn_tree321_median_window_tps"] = float("nan")
        with self.assertRaises(AssertionError):
            qualification.controls_supported(cases)
        self.assertTrue(qualification.controls_supported(original))

    def test_standard_verifier_has_correct_chunk_and_experiment_selector(self):
        raw = {"load_experiment_flags": {qualification.FLAG: "1"}}
        with patch.object(qualification, "verify_standard", return_value={"valid": True}) as verify:
            result = qualification.verify_gdn_standard(raw, {}, {}, {}, [], 32768, False)
        self.assertEqual(result, {"valid": True})
        verify.assert_called_once_with(raw, {}, {}, {}, [], 32768, False, chunk=1024)
        self.assertEqual(inspect.signature(qualification.verify_standard).parameters["chunk"].default, 256)
        raw["load_experiment_flags"][qualification.FLAG] = "0"
        with self.assertRaises(AssertionError):
            qualification.verify_gdn_standard(raw, {}, {}, {}, [], 32768, False)


if __name__ == "__main__":
    unittest.main()
