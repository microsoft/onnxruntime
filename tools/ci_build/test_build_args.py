#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

build = importlib.import_module("build")
build_args = importlib.import_module("build_args")

_RELEASED_OPSET_ENVIRONMENT = {
    "ALLOW_RELEASED_ONNX_OPSET_ONLY": "1",
    "ORT_BACKEND_TEST_ALLOW_UNRELEASED_OPSETS": "0",
}


class BuildArgsTest(unittest.TestCase):
    def _parse(self, *arguments: str, platform_name: str, machine: str = "x86_64"):
        argv = ["build.py", "--build_dir", "build/test", *arguments]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(build_args, "is_linux", return_value=platform_name == "linux"),
            mock.patch.object(build_args, "is_windows", return_value=platform_name == "windows"),
            mock.patch.object(build_args, "is_macOS", return_value=platform_name == "macos"),
            mock.patch.object(build_args.platform, "machine", return_value=machine),
        ):
            return build_args.parse_arguments()

    def test_native_supported_targets_enable_telemetry_by_default(self):
        for platform_name in ("linux", "windows", "macos"):
            with self.subTest(platform_name=platform_name):
                args = self._parse(platform_name=platform_name)
                self.assertTrue(args.use_telemetry)
                self.assertFalse(args.use_windows_telemetry)

    def test_no_telemetry_disables_supported_target(self):
        args = self._parse("--no_telemetry", platform_name="linux")
        self.assertFalse(args.use_telemetry)

    def test_windows_telemetry_backend_is_explicit(self):
        args = self._parse("--use_windows_telemetry", platform_name="windows")
        self.assertTrue(args.use_telemetry)
        self.assertTrue(args.use_windows_telemetry)

    def test_legacy_use_telemetry_retains_windows_backend(self):
        with self.assertWarnsRegex(FutureWarning, "--use_telemetry is deprecated"):
            args = self._parse("--use_telemetry", platform_name="windows")

        self.assertTrue(args.use_telemetry)
        self.assertTrue(args.use_windows_telemetry)

    def test_legacy_use_telemetry_retains_1ds_elsewhere(self):
        with self.assertWarnsRegex(FutureWarning, "--use_telemetry is deprecated"):
            args = self._parse("--use_telemetry", platform_name="linux")

        self.assertTrue(args.use_telemetry)
        self.assertFalse(args.use_windows_telemetry)

    def test_legacy_use_telemetry_selects_1ds_for_android_on_every_host(self):
        for platform_name in ("windows", "linux", "macos"):
            with (
                self.subTest(platform_name=platform_name),
                self.assertWarnsRegex(FutureWarning, "--use_telemetry is deprecated"),
            ):
                args = self._parse("--android", "--use_telemetry", platform_name=platform_name)
                self.assertTrue(args.use_telemetry)
                self.assertFalse(args.use_windows_telemetry)

    def test_windows_telemetry_backend_is_rejected_elsewhere(self):
        with self.assertRaises(SystemExit):
            self._parse("--use_windows_telemetry", platform_name="linux")

    def test_windows_telemetry_backend_is_rejected_for_unsupported_targets(self):
        for arguments in (
            ("--android",),
            ("--build_wasm",),
            ("--minimal_build", "--disable_exceptions"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self._parse("--use_windows_telemetry", *arguments, platform_name="windows")

    def test_telemetry_backend_and_opt_out_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self._parse("--use_windows_telemetry", "--no_telemetry", platform_name="windows")
        with self.assertRaises(SystemExit):
            self._parse("--use_telemetry", "--no_telemetry", platform_name="windows")

    def test_unsupported_targets_disable_telemetry(self):
        cases = (
            (("--build_wasm",), "linux", "x86_64"),
            (("--minimal_build", "--disable_exceptions"), "linux", "x86_64"),
            (("--rv64",), "linux", "riscv64"),
            (("--visionos",), "macos", "arm64"),
            (("--tvos",), "macos", "arm64"),
            (("--macos", "Catalyst", "--build_apple_framework"), "macos", "arm64"),
            ((), "linux", "riscv64"),
        )
        for arguments, platform_name, machine in cases:
            with self.subTest(arguments=arguments, platform_name=platform_name, machine=machine):
                args = self._parse(*arguments, platform_name=platform_name, machine=machine)
                self.assertFalse(args.use_telemetry)

    def test_android_enables_telemetry_by_default(self):
        args = self._parse("--android", platform_name="linux")
        self.assertTrue(args.use_telemetry)

    def test_use_acl_emits_deprecation_warning(self):
        with self.assertWarnsRegex(FutureWarning, "The ACL EP is deprecated"):
            self._parse("--use_acl", platform_name="linux")

    def test_acl_deprecation_warning_not_emitted_without_use_acl(self):
        with mock.patch.object(build_args.warnings, "warn") as warn:
            self._parse(platform_name="linux")

        warn.assert_not_called()


class OnnxBackendTestEnvironmentTest(unittest.TestCase):
    def test_cpu_and_cuda_require_released_opsets_by_default(self):
        with mock.patch.dict(build.os.environ, {}, clear=True):
            for use_cuda in (False, True):
                with self.subTest(use_cuda=use_cuda):
                    self.assertEqual(
                        build.get_onnx_backend_test_environment(use_cuda),
                        _RELEASED_OPSET_ENVIRONMENT,
                    )

    def test_cpu_and_cuda_require_released_opsets_regardless_of_parent_policy(self):
        for parent_policy in ("0", "1"):
            with mock.patch.dict(build.os.environ, {"ALLOW_RELEASED_ONNX_OPSET_ONLY": parent_policy}, clear=True):
                for use_cuda in (False, True):
                    with self.subTest(use_cuda=use_cuda, parent_policy=parent_policy):
                        self.assertEqual(
                            build.get_onnx_backend_test_environment(use_cuda),
                            _RELEASED_OPSET_ENVIRONMENT,
                        )


if __name__ == "__main__":
    unittest.main()
