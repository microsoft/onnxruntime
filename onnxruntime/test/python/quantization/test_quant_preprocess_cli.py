# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import onnx

from onnxruntime.quantization.preprocess import parse_arguments


class TestQuantPreprocessCLI(unittest.TestCase):
    def test_skip_options_default_to_false(self):
        with patch.object(sys, "argv", ["preprocess", "--input", "input.onnx", "--output", "output.onnx"]):
            args = parse_arguments()
        self.assertFalse(args.skip_optimization)
        self.assertFalse(args.skip_onnx_shape)
        self.assertFalse(args.skip_symbolic_shape)

    def test_skip_options_parse_boolean_values(self):
        for option in ("skip_optimization", "skip_onnx_shape", "skip_symbolic_shape"):
            for value, expected in (
                ("True", True),
                ("tRuE", True),
                ("1", True),
                ("False", False),
                ("fAlSe", False),
                ("0", False),
            ):
                with self.subTest(option=option, value=value):
                    argv = ["preprocess", "--input", "input.onnx", "--output", "output.onnx", f"--{option}", value]
                    with patch.object(sys, "argv", argv):
                        args = parse_arguments()
                    self.assertIs(getattr(args, option), expected)

    def test_skip_options_reject_invalid_values(self):
        for option in ("skip_optimization", "skip_onnx_shape", "skip_symbolic_shape"):
            with self.subTest(option=option):
                argv = ["preprocess", "--input", "input.onnx", "--output", "output.onnx", f"--{option}", "invalid"]
                stderr = io.StringIO()
                with (
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as error,
                ):
                    parse_arguments()
                self.assertEqual(error.exception.code, 2)
                self.assertIn(f"argument --{option}: Expected true, false, 1, or 0", stderr.getvalue())

    def test_explicit_false_runs_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.onnx"
            output_path = Path(temp_dir) / "output.onnx"
            graph = onnx.helper.make_graph(
                [onnx.helper.make_node("Identity", ["input"], ["output"])],
                "identity",
                [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])],
                [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])],
            )
            model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=7)
            onnx.save_model(model, input_path)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "onnxruntime.quantization.preprocess",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--skip_optimization",
                    "False",
                    "--skip_onnx_shape",
                    "False",
                    "--skip_symbolic_shape",
                    "False",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertTrue(output_path.exists(), result.stderr)
            onnx.checker.check_model(onnx.load_model(output_path))


if __name__ == "__main__":
    unittest.main()
