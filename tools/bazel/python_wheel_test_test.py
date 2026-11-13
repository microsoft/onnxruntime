# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import python_wheel_test


class PythonWheelSubprocessTest(unittest.TestCase):
    def test_installed_package_precedes_hermetic_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installed = root / "installed"
            dependencies = root / "dependencies"
            for package_root, package, source in [
                (installed, "onnxruntime", "installed"),
                (dependencies, "onnxruntime", "dependency"),
                (dependencies, "numpy", "dependency"),
            ]:
                package_directory = package_root / package
                package_directory.mkdir(parents=True)
                (package_directory / "__init__.py").write_text(f"SOURCE = {source!r}\n", encoding="utf-8")

            env = dict(os.environ)
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONPATH"] = python_wheel_test._subprocess_pythonpath(installed, [dependencies])
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import numpy, onnxruntime; "
                        "assert onnxruntime.SOURCE == 'installed'; "
                        "assert numpy.SOURCE == 'dependency'"
                    ),
                ],
                cwd=installed,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if result.returncode:
                self.fail(result.stdout.decode(errors="replace"))

    def test_gpu_inference_requires_cuda_without_fallback(self):
        state = {}

        class FakeArray:
            dtype = "float32"

            def reshape(self, *_shape):
                return self

            def __add__(self, _other):
                return self

            def __mul__(self, _other):
                return self

        class FakeSessionOptions:
            def __init__(self):
                self.intra_op_num_threads = None

            def add_session_config_entry(self, key, value):
                state["config"] = (key, value)

        class FakeSession:
            def __init__(self, _model, sess_options, providers):
                state["options"] = sess_options
                state["providers"] = providers

            def disable_fallback(self):
                state["fallback_disabled"] = True

            def get_providers(self):
                return state["providers"]

            def run(self, _outputs, _inputs):
                return [FakeArray()]

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "onnxruntime"
            package.mkdir()
            package_file = package / "__init__.py"
            package_file.write_text("", encoding="utf-8")
            fake_numpy = types.SimpleNamespace(
                arange=lambda *_args, **_kwargs: FakeArray(),
                float32="float32",
                testing=types.SimpleNamespace(assert_array_equal=lambda *_args: None),
            )
            fake_ort = types.SimpleNamespace(
                __file__=str(package_file),
                __version__="1.2.3",
                InferenceSession=FakeSession,
                SessionOptions=FakeSessionOptions,
            )
            with mock.patch.dict(sys.modules, {"numpy": fake_numpy, "onnxruntime": fake_ort}):
                python_wheel_test.run_inference(
                    directory,
                    "model.onnx",
                    "1.2.3",
                    "onnxruntime-gpu",
                )

        self.assertEqual(["CUDAExecutionProvider"], state["providers"])
        self.assertEqual(("session.disable_cpu_ep_fallback", "1"), state["config"])
        self.assertTrue(state["fallback_disabled"])


if __name__ == "__main__":
    unittest.main()
