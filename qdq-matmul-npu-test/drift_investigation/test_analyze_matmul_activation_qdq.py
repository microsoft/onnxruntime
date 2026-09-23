"""Regression tests for the provider-neutral QDQ weight-granularity study."""

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from drift_investigation.analyze_matmul_activation_qdq import (
    ACTIVATION_TYPES,
    REGIMES,
    WEIGHT_VARIANTS,
    activation_scales,
    cases,
    out_of_range,
    main,
    parse_args,
)
from generate_qdq_matmul_test_suite import CATEGORIES, VARIANTS


class AnalyzeMatmulActivationQdqTest(unittest.TestCase):
    def test_resume_retries_failed_case(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "matrix"
            argv = [
                "analyze_matmul_activation_qdq", "--output-dir", str(output),
                "--category", CATEGORIES[0].slug, "--weight-variant", VARIANTS[0].slug,
                "--activation-type", "uint16", "--regime", "input_clipped", "--limit", "1",
            ]
            with patch.object(sys, "argv", argv), patch(
                "drift_investigation.analyze_matmul_activation_qdq.run_case",
                side_effect=RuntimeError("compilation failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed cases"):
                    main()
            with patch.object(sys, "argv", argv + ["--resume"]), patch(
                "drift_investigation.analyze_matmul_activation_qdq.run_case",
                return_value={
                    "cpu_shift_mae": 1.0, "npu_shift_mae": 0.0,
                    "stress_fraction": 0.5, "npu_bitwise_unchanged": True,
                },
            ) as retried:
                main()
            retried.assert_called_once()
            result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(next(iter(result["results"].values()))["cpu_shift_mae"], 1.0)

    def test_all_weight_variants_cli_selects_existing_suite_matrix(self):
        with patch.object(sys, "argv", ["analyze_matmul_activation_qdq.py", "--all-weight-variants"]):
            args = parse_args()
        self.assertTrue(args.all_weight_variants)
        self.assertIsNone(args.weight_variant)

    def test_all_weight_categories_activation_types_and_stress_cases(self):
        combinations = cases(
            [category.slug for category in CATEGORIES],
            list(WEIGHT_VARIANTS), list(ACTIVATION_TYPES), list(REGIMES),
        )
        self.assertEqual(len(combinations), 40)
        self.assertEqual(len({case.name for case in combinations}), 40)
        for category in CATEGORIES:
            matching = [case for case in combinations if case.category == category]
            self.assertEqual(len(matching), 8)
        all_weights = cases(
            [category.slug for category in CATEGORIES],
            [variant.slug for variant in VARIANTS],
            list(ACTIVATION_TYPES), list(REGIMES),
        )
        self.assertEqual(len(all_weights), 200)

    def test_input_and_output_clipping_scales_are_matched_across_activation_types(self):
        for regime in REGIMES:
            scale16 = activation_scales("uint16", regime)
            scale8 = activation_scales("uint8", regime)
            self.assertEqual(
                tuple(value * 257 if value is not None else None for value in scale16),
                scale8,
            )
        with self.assertRaisesRegex(ValueError, "unsupported activation"):
            activation_scales("int8", "input_clipped")

    def test_stress_fraction_counts_only_out_of_range_values(self):
        values = np.asarray([-101, -100, 0, 65435], dtype=np.float32)
        self.assertEqual(out_of_range(values, 1.0, np.asarray(100, dtype=np.uint16)), 0.25)


if __name__ == "__main__":
    unittest.main()
