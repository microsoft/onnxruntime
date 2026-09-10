#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import contextlib
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_args


class TelemetryBuildArgsTest(unittest.TestCase):
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

    def test_no_telemetry_disables_supported_target(self):
        args = self._parse("--no_telemetry", platform_name="linux")
        self.assertFalse(args.use_telemetry)

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


class AppleAccelerateBuildArgsTest(unittest.TestCase):
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

    def test_accelerate_accepted_arm64(self):
        args = self._parse("--use_apple_accelerate", platform_name="macos", machine="arm64")
        self.assertTrue(args.use_apple_accelerate)

    def test_accelerate_accepted_arm64e(self):
        args = self._parse("--use_apple_accelerate", "--osx_arch", "arm64e", platform_name="macos", machine="arm64")
        self.assertTrue(args.use_apple_accelerate)

    def test_accelerate_accepted_macos_framework_cross_compile(self):
        args = self._parse(
            "--use_apple_accelerate",
            "--osx_arch",
            "arm64",
            "--macos",
            "MacOSX",
            "--build_apple_framework",
            platform_name="macos",
            machine="x86_64",
        )
        self.assertTrue(args.use_apple_accelerate)

    def _assert_parser_error(self, args, expected_msg_fragment, **kwargs):
        """Assert parse_arguments calls parser.error with a message containing expected_msg_fragment."""
        with self.assertRaises(SystemExit) as cm:
            self._parse(*args, **kwargs)
        self.assertEqual(cm.exception.code, 2)
        # parser.error prints to stderr; verify message via mock
        # We rely on argparse exit code 2 + the fragment appearing in the error path.
        # For a stronger check, capture stderr:
        f = io.StringIO()
        with redirect_stderr(f), contextlib.suppress(SystemExit):
            self._parse(*args, **kwargs)
        self.assertIn(expected_msg_fragment, f.getvalue())

    def test_accelerate_rejected_on_linux(self):
        self._assert_parser_error(
            ("--use_apple_accelerate",),
            "only supported on macOS",
            platform_name="linux",
        )

    def test_accelerate_rejected_on_windows(self):
        self._assert_parser_error(
            ("--use_apple_accelerate",),
            "only supported on macOS",
            platform_name="windows",
        )

    def test_accelerate_rejected_x86_64(self):
        self._assert_parser_error(
            ("--use_apple_accelerate", "--osx_arch", "x86_64"),
            "requires an arm64 target architecture",
            platform_name="macos",
            machine="x86_64",
        )

    def test_accelerate_rejected_ios(self):
        self._assert_parser_error(
            ("--use_apple_accelerate", "--ios"),
            "not supported for iOS builds",
            platform_name="macos",
            machine="arm64",
        )

    def test_accelerate_rejected_tvos(self):
        self._assert_parser_error(
            ("--use_apple_accelerate", "--tvos"),
            "not supported for tvOS builds",
            platform_name="macos",
            machine="arm64",
        )

    def test_accelerate_rejected_visionos(self):
        self._assert_parser_error(
            ("--use_apple_accelerate", "--visionos"),
            "not supported for visionOS builds",
            platform_name="macos",
            machine="arm64",
        )

    def test_accelerate_rejected_catalyst(self):
        self._assert_parser_error(
            ("--use_apple_accelerate", "--macos", "Catalyst", "--build_apple_framework"),
            "not supported for Mac Catalyst builds",
            platform_name="macos",
            machine="arm64",
        )

    def test_accelerate_rejected_android(self):
        self._assert_parser_error(
            ("--use_apple_accelerate", "--android"),
            "not supported for Android builds",
            platform_name="macos",
            machine="arm64",
        )

    def test_accelerate_rejected_build_wasm(self):
        self._assert_parser_error(
            ("--use_apple_accelerate", "--build_wasm"),
            "not supported for WebAssembly builds",
            platform_name="macos",
            machine="arm64",
        )

    def test_accelerate_rejected_build_wasm_static_lib(self):
        self._assert_parser_error(
            ("--use_apple_accelerate", "--build_wasm_static_lib"),
            "not supported for WebAssembly builds",
            platform_name="macos",
            machine="arm64",
        )

    def test_accelerate_rejected_rv64(self):
        self._assert_parser_error(
            ("--use_apple_accelerate", "--rv64"),
            "not supported for RISC-V (rv64) builds",
            platform_name="macos",
            machine="arm64",
        )


if __name__ == "__main__":
    unittest.main()
