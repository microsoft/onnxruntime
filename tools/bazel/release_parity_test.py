# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import contextlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.bazel import release_parity


class ReleaseParityTest(unittest.TestCase):
    def test_payload_files_rejects_missing_or_empty_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "does not exist or is not a directory"):
                release_parity._payload_files(root / "missing")
            with self.assertRaisesRegex(ValueError, "is empty"):
                release_parity._payload_files(root)

    def test_compressed_size_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "runtime.so").write_bytes(b"native payload")
            first = release_parity._compressed_size(root)
            self.assertGreater(first, 0)
            self.assertEqual(first, release_parity._compressed_size(root))

    def test_stripped_measurements_make_temporary_elf_writable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            binary = root / "runtime.so"
            binary.write_bytes(b"ELF")
            binary.chmod(stat.S_IRUSR)

            def check_writable(command, **_kwargs):
                self.assertNotEqual(0, Path(command[-1]).stat().st_mode & stat.S_IWUSR)
                return mock.Mock()

            with (
                mock.patch.object(release_parity, "_is_elf", return_value=True),
                mock.patch.object(release_parity, "_elf_section_size", return_value=3),
                mock.patch.object(release_parity.subprocess, "run", side_effect=check_writable),
            ):
                self.assertEqual((3, 3), release_parity._stripped_measurements(root, "strip", "size"))

    def test_compare_benchmarks_passes_confident_parity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps({"workloads": {"conv": [100, 101, 99, 100, 100]}}))
            candidate.write_text(json.dumps({"workloads": {"conv": [99, 100, 98, 99, 100]}}))

            results, failures = release_parity._compare_benchmarks(baseline, candidate, 2)

            self.assertEqual([], failures)
            self.assertEqual("pass", results["conv"]["status"])
            confidence_interval = results["conv"]["paired_ratio_95_percent_confidence_interval"]
            self.assertLessEqual(
                confidence_interval[0],
                results["conv"]["paired_candidate_to_baseline_ratio_median"],
            )
            self.assertGreaterEqual(
                confidence_interval[1],
                results["conv"]["paired_candidate_to_baseline_ratio_median"],
            )

    def test_compare_benchmarks_rejects_regression(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps({"workloads": {"conv": [100, 100, 100, 100, 100]}}))
            candidate.write_text(json.dumps({"workloads": {"conv": [105, 105, 105, 105, 105]}}))

            results, failures = release_parity._compare_benchmarks(baseline, candidate, 2)

            self.assertNotEqual([], failures)
            self.assertEqual("regression", results["conv"]["status"])

    def test_compare_benchmarks_enforces_zero_threshold(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps({"workloads": {"conv": [100] * 5}}))
            for samples, expected in (([100] * 5, "pass"), ([100.01] * 5, "regression")):
                with self.subTest(expected=expected):
                    candidate.write_text(json.dumps({"workloads": {"conv": samples}}))
                    results, failures = release_parity._compare_benchmarks(baseline, candidate, 0)
                    self.assertEqual(expected, results["conv"]["status"])
                    self.assertEqual(expected != "pass", bool(failures))

    def test_compare_benchmarks_rejects_inconclusive_interval(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps({"workloads": {"conv": [100] * 5}}))
            candidate.write_text(json.dumps({"workloads": {"conv": [98, 99, 100, 101, 102]}}))

            results, failures = release_parity._compare_benchmarks(baseline, candidate, 0)

            self.assertEqual("inconclusive", results["conv"]["status"])
            self.assertTrue(failures)
            low, high = results["conv"]["paired_ratio_95_percent_confidence_interval"]
            self.assertLess(low, 1)
            self.assertGreater(high, 1)

    def test_compare_benchmarks_uses_paired_ratio_for_point_and_confidence_interval(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps({"workloads": {"conv": [1, 2, 3, 100, 101]}}))
            candidate.write_text(json.dumps({"workloads": {"conv": [1, 4, 6, 100, 101]}}))

            results, _ = release_parity._compare_benchmarks(baseline, candidate, 0)

            self.assertEqual(2, results["conv"]["candidate_to_baseline_ratio_of_medians"])
            self.assertEqual(1, results["conv"]["paired_candidate_to_baseline_ratio_median"])
            confidence_interval = results["conv"]["paired_ratio_95_percent_confidence_interval"]
            self.assertLessEqual(
                confidence_interval[0],
                results["conv"]["paired_candidate_to_baseline_ratio_median"],
            )
            self.assertGreaterEqual(
                confidence_interval[1],
                results["conv"]["paired_candidate_to_baseline_ratio_median"],
            )

    def test_load_benchmarks_rejects_nonfinite_samples(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            benchmark = Path(temporary_directory) / "benchmark.json"
            benchmark.write_text(json.dumps({"workloads": {"conv": [1, 2, 3, 4, float("inf")]}}))

            with self.assertRaisesRegex(ValueError, "workload 'conv' samples must be finite"):
                release_parity._load_benchmarks(benchmark)

    def test_main_rejects_nonfinite_performance_threshold(self):
        arguments = [
            "release_parity.py",
            "--baseline-payload",
            "baseline",
            "--candidate-payload",
            "candidate",
            "--strip-tool",
            "strip",
            "--size-tool",
            "size",
            "--baseline-benchmarks",
            "baseline.json",
            "--candidate-benchmarks",
            "candidate.json",
            "--max-performance-regression-percent",
            "nan",
        ]

        with (
            mock.patch("sys.argv", arguments),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as error,
        ):
            release_parity.main()

        self.assertEqual(2, error.exception.code)
        self.assertIn("--max-performance-regression-percent must be finite", stderr.getvalue())

    def test_main_rejects_payload_growth(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            (baseline / "runtime.so").write_bytes(b"small")
            (candidate / "runtime.so").write_bytes(b"larger")
            arguments = [
                "release_parity.py",
                "--baseline-payload",
                str(baseline),
                "--candidate-payload",
                str(candidate),
                "--strip-tool",
                "strip",
                "--size-tool",
                "size",
            ]

            with mock.patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(1, release_parity.main())


if __name__ == "__main__":
    unittest.main()
