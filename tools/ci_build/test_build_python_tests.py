#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib
import importlib.util
import itertools
import sys
import tempfile
import types
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
build = importlib.import_module("build")
build_args = importlib.import_module("build_args")
ROOT = Path(__file__).resolve().parents[2]


class PythonTestDispatchTest(unittest.TestCase):
    def test_dispatch_preserves_non_torch_suites_and_gates_torch(self):
        cases = itertools.product(
            ((3, 11), (3, 14), (3, 15)),
            (("linux", False), ("linux", True), ("windows", True)),
            (False, True),
            (False, True),
            (False, True),
        )
        for version, (platform, cuda), transformers, disable_contrib, skip_pip in cases:
            with (
                self.subTest(
                    version=version,
                    platform=platform,
                    cuda=cuda,
                    transformers=transformers,
                    disable_contrib=disable_contrib,
                    skip_pip=skip_pip,
                ),
                tempfile.TemporaryDirectory() as directory,
            ):
                flags = ["--build_dir", directory, "--enable_pybind", "--skip-keras-test"]
                if cuda:
                    flags.append("--use_cuda")
                if transformers:
                    flags.append("--enable_transformers_tool_test")
                if disable_contrib:
                    flags.append("--disable_contrib_ops")
                if skip_pip:
                    flags.append("--skip_pip_install")
                numpy = types.ModuleType("numpy")
                numpy.__version__ = "2.4.2"
                google = types.ModuleType("google")
                protobuf = types.ModuleType("google.protobuf")
                protobuf.__version__ = "6.33.6"
                google.protobuf = protobuf
                modules = {
                    "onnx": types.ModuleType("onnx"),
                    "numpy": numpy,
                    "google": google,
                    "google.protobuf": protobuf,
                }
                with (
                    mock.patch.object(sys, "argv", ["build.py", *flags]),
                    mock.patch.object(build_args, "is_linux", return_value=platform == "linux"),
                    mock.patch.object(build_args, "is_windows", return_value=platform == "windows"),
                    mock.patch.object(build_args, "is_macOS", return_value=False),
                    mock.patch.object(build_args.platform, "machine", return_value="x86_64"),
                ):
                    args = build_args.parse_arguments()
                version_info = namedtuple("VersionInfo", "major minor micro releaselevel serial")(
                    *version, 0, "final", 0
                )
                with (
                    mock.patch.object(build.sys, "version_info", version_info),
                    mock.patch.object(build, "is_linux", return_value=platform == "linux"),
                    mock.patch.object(build, "is_windows", return_value=platform == "windows"),
                    mock.patch.object(build, "is_macOS", return_value=False),
                    mock.patch.dict(sys.modules, modules),
                    mock.patch.object(build, "run_subprocess") as run,
                    mock.patch.object(build.log, "warning") as warning,
                ):
                    build.run_onnxruntime_tests(args, str(ROOT), "ctest", directory, ["Release"])
                commands = [call.args[0] for call in run.call_args_list]
                pip_commands = [cmd for cmd in commands if cmd[1:4] == ["-m", "pip", "install"]]
                expected = []
                if version < (3, 15) and not skip_pip and not disable_contrib:
                    if transformers:
                        extra = (
                            []
                            if cuda and platform == "linux"
                            else ["--index-url", "https://download.pytorch.org/whl/cpu"]
                        )
                        expected.append(
                            [sys.executable, "-m", "pip", "install", "torch==2.10.0", "torchvision==0.25.0", *extra]
                        )
                    elif cuda and platform == "linux":
                        expected.append(
                            [
                                sys.executable,
                                "-m",
                                "pip",
                                "install",
                                "torch==2.10.0",
                                "--index-url",
                                "https://download.pytorch.org/whl/cpu",
                            ]
                        )
                    if transformers:
                        expected.extend(
                            [
                                [
                                    sys.executable,
                                    "-m",
                                    "pip",
                                    "install",
                                    "-r",
                                    "requirements/transformers-test/requirements.txt",
                                ],
                                [sys.executable, "-m", "pip", "install", "numpy==2.4.2"],
                                [sys.executable, "-m", "pip", "install", "protobuf==6.33.6"],
                            ]
                        )
                self.assertEqual(pip_commands, expected)
                self.assertEqual(
                    [sys.executable, "-m", "pytest", "--durations=0", "transformers"] in commands,
                    transformers and not disable_contrib and version < (3, 15),
                )
                for script in (
                    "onnxruntime_test_python.py",
                    "onnxruntime_test_python_backend.py",
                    "onnxruntime_test_python_symlink_data.py",
                ):
                    self.assertIn([sys.executable, script], commands)
                self.assertEqual(
                    [sys.executable, "-m", "unittest", "discover", "-s", "quantization", "-v"] in commands,
                    not disable_contrib,
                )
                if cuda:
                    for script in ("iobinding", "cuda_kernel", "cudagraph", "ort_parallel"):
                        self.assertIn([sys.executable, f"onnxruntime_test_python_{script}.py"], commands)
                if (
                    version >= (3, 15)
                    and not disable_contrib
                    and (transformers or (not skip_pip and cuda and platform == "linux"))
                ):
                    self.assertTrue(
                        any("PyTorch" in str(call) and "3.15" in str(call) for call in warning.call_args_list)
                    )


class TorchPackerVersionSkipTest(unittest.TestCase):
    def test_parity_skip_is_specific_to_python315(self):
        path = ROOT / "onnxruntime/test/python/quantization/test_op_matmulnbits_prepacked_cuda.py"
        for version in ((3, 14), (3, 15)):
            with self.subTest(version=version):
                names = (
                    "numpy",
                    "onnx",
                    "onnxruntime",
                    "onnxruntime.capi",
                    "onnxruntime.capi._pybind_state",
                    "onnxruntime.quantization",
                    "onnxruntime.quantization.cuda_quantizer",
                    "onnxruntime.capi.onnxruntime_cuda_quant_preprocess",
                )
                modules = {name: types.ModuleType(name) for name in names}
                for name in ("ModelProto", "TensorProto", "helper", "numpy_helper"):
                    setattr(modules["onnx"], name, mock.Mock())
                modules["onnxruntime"].get_available_providers = lambda: ["CUDAExecutionProvider"]
                packer = mock.Mock()
                modules["onnxruntime.quantization.cuda_quantizer"]._pack_weights_for_cuda_mixed_gemm = packer
                spec = importlib.util.spec_from_file_location("_parity_version_test", path)
                module = importlib.util.module_from_spec(spec)
                with (
                    mock.patch.dict(sys.modules, {**modules, spec.name: module}),
                    mock.patch.object(sys, "version_info", (*version, 0, "final", 0)),
                ):
                    spec.loader.exec_module(module)
                case_class = module.TestCudaQuantizerTorchPackerParity
                self.assertFalse(getattr(case_class, "__unittest_skip__", False))
                self.assertFalse(
                    getattr(case_class.test_explicit_sm90_layout_does_not_fall_back_to_sm80, "__unittest_skip__", False)
                )
                with mock.patch.object(case_class, "_check") as check:
                    result = unittest.TestResult()
                    case_class("test_torch_packer_matches_cuda_oracle").run(result)
                self.assertEqual(result.testsRun, 1)
                self.assertEqual(result.errors, [])
                self.assertEqual(result.failures, [])
                self.assertEqual(len(result.skipped), int(version >= (3, 15)))
                self.assertEqual(check.call_count, 0 if version >= (3, 15) else 24)
                packer.assert_not_called()
                if result.skipped:
                    self.assertEqual(result.skipped[0][1], "PyTorch cp315 wheels are unavailable on PyPI")


if __name__ == "__main__":
    unittest.main()
