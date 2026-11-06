#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parents[2]
_GUARD = _ROOT / "cmake" / "onnxruntime_training_retirement.cmake"
_RETIRED_OPTIONS = (
    "onnxruntime_ENABLE_TRAINING",
    "onnxruntime_ENABLE_TRAINING_APIS",
    "onnxruntime_ENABLE_TRAINING_OPS",
    "onnxruntime_ENABLE_TRAINING_E2E_TESTS",
    "onnxruntime_ENABLE_TRAINING_TORCH_INTEROP",
    "onnxruntime_ENABLE_LAZY_TENSOR",
    "onnxruntime_ENABLE_EAGER_MODE",
    "onnxruntime_ENABLE_TRITON",
    "onnxruntime_BUILD_OPSCHEMA_LIB",
    "onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS",
)


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class TrainingRetirementCMakeTest(unittest.TestCase):
    def _run_guard(self, *definitions):
        return subprocess.run(["cmake", *definitions, "-P", str(_GUARD)], capture_output=True, text=True, check=False)

    def test_enabled_retired_options_are_rejected(self):
        for option in _RETIRED_OPTIONS:
            for value in ("ON", "TRUE", "YES", "1", "2", "enabled"):
                with self.subTest(option=option, value=value):
                    result = self._run_guard(f"-D{option}:BOOL={value}")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("ORT training has been retired.", result.stderr)
                    self.assertIn(f"{option} is no longer supported.", " ".join(result.stderr.split()))

    def test_disabled_retired_options_are_tolerated(self):
        for value in ("OFF", "FALSE", "NO", "0", "IGNORE", "NOTFOUND", "tool-NOTFOUND", ""):
            with self.subTest(value=value):
                result = self._run_guard(*(f"-D{option}:BOOL={value}" for option in _RETIRED_OPTIONS))
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_inference_options_and_unset_training_options_are_supported(self):
        result = self._run_guard(
            "-Donnxruntime_USE_NCCL=ON", "-Donnxruntime_ENABLE_ATEN=ON", "-Donnxruntime_ENABLE_DLPACK=ON"
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_top_level_configuration_rejects_training_before_toolchain_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(_ROOT / "cmake"),
                    "-B",
                    directory,
                    "-Donnxruntime_ENABLE_TRAINING_APIS=ON",
                    "-DCMAKE_C_COMPILER=missing-training-retirement-compiler",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ORT training has been retired.", result.stderr)
        self.assertNotIn("compiler identification", result.stdout)


class TrainingRetirementPackagingTest(unittest.TestCase):
    def test_inference_archive_names_are_unchanged(self):
        spec = importlib.util.spec_from_file_location(
            "native_nuget", _ROOT / "tools" / "nuget" / "generate_nuspec_for_native_nuget.py"
        )
        native_nuget = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native_nuget)
        for system, arch, provider, expected in (
            ("win", "x64", None, "onnxruntime-win-x64"),
            ("win", "arm64", "cuda", "onnxruntime-win-arm64-cuda"),
            ("linux", "aarch64", "tensorrt", "onnxruntime-linux-aarch64-tensorrt"),
            ("osx", "arm64", None, "onnxruntime-osx-arm64"),
        ):
            with self.subTest(system=system, arch=arch, provider=provider):
                self.assertEqual(native_nuget.get_package_name(system, arch, provider), expected)

    def test_nuget_rejects_retired_package_names(self):
        spec = importlib.util.spec_from_file_location(
            "native_nuget", _ROOT / "tools" / "nuget" / "generate_nuspec_for_native_nuget.py"
        )
        native_nuget = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native_nuget)
        for package_name in ("Microsoft.ML.OnnxRuntime.Training", "Microsoft.ML.OnnxRuntime.Training.Gpu"):
            with (
                self.subTest(package_name=package_name),
                self.assertRaisesRegex(ValueError, "ORT training has been retired"),
            ):
                native_nuget.generate_nuspec(SimpleNamespace(package_name=package_name))

    def test_setup_rejects_retired_package_options(self):
        for option in ("--enable_training", "--enable_training_apis", "--default_training_package_device"):
            for suffix in ("", "=false"):
                with self.subTest(option=option, suffix=suffix):
                    result = subprocess.run(
                        [sys.executable, str(_ROOT / "setup.py"), f"{option}{suffix}", "--name"],
                        cwd=_ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f"ORT training has been retired. {option} is no longer supported.", result.stderr)


class TrainingRetirementInferenceTest(unittest.TestCase):
    def test_kernel_registration_files_survive_training_removal(self):
        import op_registration_utils  # noqa: PLC0415

        for include_cuda, count in ((False, 2), (True, 4)):
            with self.subTest(include_cuda=include_cuda):
                files = op_registration_utils.get_kernel_registration_files(str(_ROOT), include_cuda=include_cuda)
                self.assertEqual(len(files), count)
                self.assertTrue(all(Path(filename).is_file() for filename in files))
                self.assertTrue(all("training" not in Path(filename).name for filename in files))

    def test_aten_executor_loads_without_training_package(self):
        native = SimpleNamespace(register_aten_op_executor=Mock())
        capi = SimpleNamespace(_pybind_state=native)
        extension = SimpleNamespace(
            is_tensor_argument_address=lambda: 123,
            execute_aten_operator_address=lambda: 456,
        )
        with patch.dict(
            sys.modules,
            {
                "onnxruntime": SimpleNamespace(capi=capi),
                "onnxruntime.capi": capi,
                "ort_torch_ext": SimpleNamespace(aten_op_executor=extension),
            },
        ):
            spec = importlib.util.spec_from_file_location(
                "inference_aten_loader",
                _ROOT / "onnxruntime" / "python" / "torch_cpp_extensions" / "aten_op_executor" / "__init__.py",
            )
            loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loader)
            loader.load_aten_op_executor_cpp_extension()
            loader.load_aten_op_executor_cpp_extension()
        native.register_aten_op_executor.assert_called_once_with("123", "456")


if __name__ == "__main__":
    unittest.main()
