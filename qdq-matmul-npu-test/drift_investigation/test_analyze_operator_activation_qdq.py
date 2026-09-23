"""Check operator-study coverage and source-model QDQ inventory."""

import tempfile
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import onnx

from drift_investigation.analyze_operator_activation_qdq import main, source_operator_counts
from drift_investigation.generate_qdq_operator_model import OPERATORS, build_model


class AnalyzeOperatorActivationQdqTest(unittest.TestCase):
    def test_resume_retries_failed_case(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "matrix"
            argv = [
                "analyze_operator_activation_qdq", "--output-dir", str(output),
                "--op", "Add", "--activation-type", "uint16", "--regime", "input_clipped",
                "--limit", "1",
            ]
            with patch.object(sys, "argv", argv), patch(
                "drift_investigation.analyze_operator_activation_qdq.run_case",
                side_effect=RuntimeError("compilation failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed cases"):
                    main()
            with patch.object(sys, "argv", argv + ["--resume"]), patch(
                "drift_investigation.analyze_operator_activation_qdq.run_case",
                return_value={
                    "cpu_shift_mae": 1.0, "npu_shift_mae": 0.0,
                    "clipped_fraction": 0.5, "npu_bitwise_unchanged": True,
                },
            ) as retried:
                main()
            retried.assert_called_once()
            result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(next(iter(result["results"].values()))["cpu_shift_mae"], 1.0)

    def test_covers_fifteen_weight_free_operator_families(self):
        self.assertEqual(len(OPERATORS), 15)
        self.assertEqual(len(set(OPERATORS)), len(OPERATORS))
        with tempfile.TemporaryDirectory() as folder:
            model, _ = build_model("Add", "uint16", "input_clipped", 1009)
            path = Path(folder) / "add.onnx"
            onnx.save(model, path)
            self.assertEqual(source_operator_counts(path, ["Add"]), {"Add": 1})
            with self.assertRaisesRegex(ValueError, "no QDQ-wrapped operators"):
                source_operator_counts(path, ["Gelu"])


if __name__ == "__main__":
    unittest.main()
