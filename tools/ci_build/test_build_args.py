#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_args


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


if __name__ == "__main__":
    unittest.main()
