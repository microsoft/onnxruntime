# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse  # noqa: F401
import collections
import csv
import re  # noqa: F401
import sys

Comparison = collections.namedtuple("Comparison", ["name", "fn"])


class Comparisons:
    @staticmethod
    def eq():
        return Comparison(name="equal to", fn=(lambda actual, expected: actual == expected))

    @staticmethod
    def float_le(tolerance=None):
        actual_tolerance = 0.0 if tolerance is None else tolerance
        return Comparison(
            name="less than or equal to" + (f" (tolerance: {actual_tolerance!s})" if tolerance is not None else ""),
            fn=(lambda actual, expected: float(actual) <= float(expected) + actual_tolerance),
        )


def _printf_stderr(fmt, *args):
    print(fmt.format(*args), file=sys.stderr)


def _read_results_file(results_path):
    with open(results_path) as results_file:
        csv_reader = csv.DictReader(results_file)
        return [row for row in csv_reader]


def _compare_results(expected_results, actual_results, field_comparisons):
    if len(field_comparisons) == 0:
        return True

    if len(expected_results) != len(actual_results):
        _printf_stderr("Expected and actual result sets have different sizes.")
        return False

    mismatch_detected = False
    for row_idx, (expected_row, actual_row) in enumerate(zip(expected_results, actual_results)):
        for field_name, comparison in field_comparisons.items():
            actual, expected = actual_row[field_name], expected_row[field_name]
            if not comparison.fn(actual, expected):
                _printf_stderr(
                    "Comparison '{}' failed for {} in row {}, actual: {}, expected: {}",
                    comparison.name,
                    field_name,
                    row_idx,
                    actual,
                    expected,
                )
                mismatch_detected = True
    return not mismatch_detected


def compare_results_files(expected_results_path: str, actual_results_path: str, field_comparisons: dict):
    expected_results = _read_results_file(expected_results_path)
    actual_results = _read_results_file(actual_results_path)

    comparison_result = _compare_results(expected_results, actual_results, field_comparisons)

    if not comparison_result:
        with open(expected_results_path) as expected_results_file, open(actual_results_path) as actual_results_file:
            _printf_stderr(
                "===== Expected results =====\n{}\n=====  Actual results  =====\n{}",
                expected_results_file.read(),
                actual_results_file.read(),
            )

    return comparison_result
