# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

"""Regression tests for synthetic GEMM model generation."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import onnx


class CreateKleidiAISveFp32GemmModelTest(unittest.TestCase):
    def run_generator(self, output, *args):
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("create_kleidiai_sve_fp32_gemm_model.py")),
                "--output",
                str(output),
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_alignment_metadata_matches_custom_dimensions(self):
        for m, n, expected in [
            (6, 32, "M multiple of 6: True; N multiple of 32 FP32 columns: True"),
            (7, 32, "M multiple of 6: False; N multiple of 32 FP32 columns: True"),
            (6, 33, "M multiple of 6: True; N multiple of 32 FP32 columns: False"),
            (7, 33, "M multiple of 6: False; N multiple of 32 FP32 columns: False"),
        ]:
            with self.subTest(m=m, n=n), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "model.onnx"
                result = self.run_generator(output, "--m", str(m), "--n", str(n), "--k", "5")
                self.assertEqual(result.returncode, 0, result.stderr)
                model = onnx.load(output)
                onnx.checker.check_model(model)
                properties = {entry.key: entry.value for entry in model.metadata_props}
                self.assertEqual(properties["sve_256_alignment"], expected)
                self.assertEqual(properties["shape"], f"M={m},N={n},K=5")
                self.assertEqual(list(model.graph.initializer[0].dims), [5, n])
                tensor = onnx.TensorProto()
                tensor.ParseFromString((output.parent / "test_data_set_0/input_0.pb").read_bytes())
                self.assertEqual(list(tensor.dims), [m, 5])

    def test_default_shape_is_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            result = self.run_generator(output)
            self.assertEqual(result.returncode, 0, result.stderr)
            model = onnx.load(output)
            onnx.checker.check_model(model)
            properties = {entry.key: entry.value for entry in model.metadata_props}
            self.assertEqual(properties["shape"], "M=96,N=1024,K=1024")
            self.assertEqual(
                properties["sve_256_alignment"],
                "M multiple of 6: True; N multiple of 32 FP32 columns: True",
            )
            self.assertEqual(model.ir_version, 8)
            self.assertEqual(model.opset_import[0].version, 13)

    def test_nonpositive_dimensions_are_rejected(self):
        for argument in ["--m", "--n", "--k"]:
            for value in ["0", "-1"]:
                with self.subTest(argument=argument, value=value), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "model.onnx"
                    result = self.run_generator(output, argument, value)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("M, N, and K must all be positive", result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
