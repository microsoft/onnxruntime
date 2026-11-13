# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.python.generate_matmul_nbits_matrix import bias_tail_cases, matrix_cases
from tools.python.profile_matmul_nbits import EVENT, counter_value, validate_results, write_json, write_reports


class TestProfileMatMulNBits(unittest.TestCase):
    def test_bias_tail_matrix_pairs_bias_and_covers_boundaries(self):
        cases = bias_tail_cases((32, 64, 128))
        self.assertEqual(len(cases), len({case.identity() for case in cases}))
        groups = {}
        for case in cases:
            key = (case.a_shape, case.n, case.block_size, case.accuracy_level)
            groups.setdefault(key, set()).add(case.bias)
            self.assertEqual(case.dtype, "float32")
            self.assertFalse(case.zero_points)
        self.assertTrue(all(bias == {False, True} for bias in groups.values()))
        for b in (32, 64, 128):
            for accuracy in (0, 4):
                for k in (b - 1, b, b + 1, 2 * b - 1, 2 * b, 2 * b + 1, 3 * b, 4 * b):
                    for m in (1, 8):
                        self.assertIn(((m, k), 32, b, accuracy), groups)
                for n in (3, 4, 5, 31, 32, 33):
                    self.assertIn(((8, 4 * b), n, b, accuracy), groups)
                for m in (2, 3, 7, 8, 9, 32):
                    self.assertIn(((m, 4 * b), 32, b, accuracy), groups)
                self.assertIn(((3, 2 * b + 1), 33, b, accuracy), groups)
        self.assertEqual(bias_tail_cases((32, 32)), bias_tail_cases())
        with self.assertRaises(ValueError):
            bias_tail_cases((24,))

    def test_matrix_has_original_84_unique_requests(self):
        cases = matrix_cases()
        self.assertEqual(len(cases), 84)
        self.assertEqual(len({case.identity() for case in cases}), 84)
        self.assertEqual(
            {(case.dtype, case.accuracy_level) for case in cases},
            {("float32", i) for i in range(5)} | {("float16", 0), ("float16", 4)},
        )
        self.assertTrue(all(case.bits == 4 and case.block_size == 32 for case in cases))

    def test_counter_parser_rejects_unusable_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counter.csv"
            for value, running in (("<not supported>", "100"), ("0", "100"), ("123", "50"), ("123", "nan")):
                path.write_text(f"{value};;{EVENT};123;{running};;\n")
                with self.subTest(value=value, running=running), self.assertRaises(ValueError):
                    counter_value(path, EVENT)
            path.write_text(f"# comment\n123;;{EVENT};999;100.00;;\n")
            self.assertEqual(counter_value(path, EVENT), 123)
            path.write_text(path.read_text() * 2)
            with self.assertRaises(ValueError):
                counter_value(path, EVENT)

    def fixture(self, root, values=(100, 110, 120)):
        collection = root / "collection"
        collection.mkdir()
        results = {
            "complete": True,
            "cases": [
                {
                    "case_id": "test_case",
                    "status": "pass",
                    "settings": {"a_shape": [2, 3, 64], "n": 32, "dtype": "float32", "accuracy_level": 0},
                    "route": "packed_qnbit",
                    "compute_mode": "sqnbit_fp32",
                    "expected_flops": {"expected_flops": "100"},
                }
            ],
        }
        write_json(
            collection / "metadata.json",
            {
                "trials": len(values),
                "repetitions": [1000, 2000],
                "event": EVENT,
                "warmup_runs": 100,
                "threads": 1,
            },
        )
        index = []
        for trial, value in enumerate(values, 1):
            for repeats, count in ((1000, 1000000), (2000, 1000000 + 1000 * value)):
                path = collection / f"trial-{trial}-{repeats}.csv"
                path.write_text(f"{count};;{EVENT};12345;100.00;;\n")
                index.append(
                    {
                        "case_id": "test_case",
                        "trial": trial,
                        "repeats": repeats,
                        "returncode": 0,
                        "counter_file": path.name,
                    }
                )
        write_json(collection / "index.json", index)
        output = root / "output"
        output.mkdir()
        return collection, results, output

    def test_report_statistics_and_accuracy_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            collection, results, output = self.fixture(Path(temporary))
            write_reports(collection, results, output)
            with (output / "summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["measured_mean"]), 110)
            self.assertEqual(float(row["measured_sample_stddev"]), 10)
            self.assertEqual(float(row["signed_difference_percent"]), 10)
            self.assertEqual(int(row["m"]), 6)
            self.assertIn("## Accuracy request key", (output / "report.md").read_text())
            self.assertIn("not operator accuracy percentages", (output / "report.md").read_text())
            with self.assertRaises(FileExistsError):
                write_reports(collection, results, output)

    def test_identical_trials_have_zero_standard_deviation(self):
        with tempfile.TemporaryDirectory() as temporary:
            collection, results, output = self.fixture(Path(temporary), (100, 100, 100))
            write_reports(collection, results, output)
            with (output / "summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["measured_sample_stddev"]), 0)

    def test_single_trial_has_no_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as temporary:
            collection, results, output = self.fixture(Path(temporary), (100,))
            write_reports(collection, results, output)
            self.assertIn("N/A", (output / "report.md").read_text())

    def test_incomplete_duplicate_failed_and_escaping_measurements_rejected(self):
        for mutation in ("missing", "duplicate", "failed", "escape"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                collection, results, output = self.fixture(Path(temporary))
                index = json.loads((collection / "index.json").read_text())
                if mutation == "missing":
                    index.pop()
                elif mutation == "duplicate":
                    index.append(index[0])
                elif mutation == "failed":
                    index[0]["returncode"] = 1
                else:
                    index[0]["counter_file"] = "../outside.csv"
                write_json(collection / "index.json", index)
                with self.assertRaises(ValueError):
                    write_reports(collection, results, output)
                self.assertFalse((output / "report.md").exists())

    def test_nonpass_or_duplicate_estimator_results_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, results, _ = self.fixture(Path(temporary))
            results["cases"][0]["status"] = "estimate_unavailable"
            with self.assertRaises(ValueError):
                validate_results(results)
            results["cases"][0]["status"] = "pass"
            results["cases"].append(results["cases"][0])
            with self.assertRaises(ValueError):
                validate_results(results)


if __name__ == "__main__":
    unittest.main()
