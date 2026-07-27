# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
REPO_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR / "python"))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "github" / "android"))

import build  # noqa: E402
import build_aar_package  # noqa: E402
from build_args import parse_arguments, target_supports_telemetry  # noqa: E402


class TelemetryBuildArgsTest(unittest.TestCase):
    def _parse(self, *build_args):
        with (
            mock.patch.object(
                sys, "argv", ["build.py", "--build_dir", "build", "--skip_tests", "--update", *build_args]
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return parse_arguments()

    def _generated_telemetry_value(self, *build_args):
        args = self._parse(*build_args)
        cmake_commands = []
        with (
            tempfile.TemporaryDirectory() as build_dir,
            mock.patch.object(
                build, "run_subprocess", side_effect=lambda command, **kwargs: cmake_commands.append(command)
            ),
        ):
            build.generate_build_tree(
                cmake_path="cmake",
                source_dir=str(REPO_DIR),
                build_dir=build_dir,
                cuda_home="",
                cudnn_home="",
                nccl_home="",
                tensorrt_home="",
                tensorrt_rtx_home="",
                migraphx_home="",
                acl_home="",
                acl_libs="",
                qnn_home="",
                snpe_root="",
                cann_home="",
                path_to_protoc_exe="",
                configs=["Release"],
                cmake_extra_defines=[],
                args=args,
                cmake_extra_args=[],
            )

        self.assertEqual(len(cmake_commands), 1)
        return next(
            argument.rsplit("=", 1)[1]
            for argument in cmake_commands[0]
            if argument.startswith("-Donnxruntime_USE_TELEMETRY=")
        )

    def test_telemetry_enabled_by_default(self):
        self.assertTrue(self._parse().use_telemetry)
        self.assertEqual(self._generated_telemetry_value(), "ON")

    def test_no_telemetry_disables_telemetry(self):
        self.assertFalse(self._parse("--no_telemetry").use_telemetry)
        self.assertEqual(self._generated_telemetry_value("--no_telemetry"), "OFF")

    def test_webassembly_disables_telemetry(self):
        self.assertFalse(self._parse("--build_wasm").use_telemetry)
        self.assertFalse(self._parse("--build_wasm_static_lib").use_telemetry)
        self.assertEqual(self._generated_telemetry_value("--build_wasm"), "OFF")

    def test_unsupported_apple_targets_disable_telemetry(self):
        for target in (
            mock.Mock(build_wasm=False, disable_exceptions=False, visionos=False, tvos=False, macos=None),
            mock.Mock(build_wasm=False, disable_exceptions=False, visionos=True, tvos=False, macos=None),
            mock.Mock(build_wasm=False, disable_exceptions=False, visionos=False, tvos=True, macos=None),
            mock.Mock(build_wasm=False, disable_exceptions=False, visionos=False, tvos=False, macos="Catalyst"),
        ):
            with self.subTest(target=target):
                self.assertEqual(
                    target_supports_telemetry(target),
                    not (target.visionos or target.tvos or target.macos == "Catalyst"),
                )

    def test_exception_free_build_disables_telemetry(self):
        self.assertFalse(self._parse("--minimal_build", "--disable_exceptions").use_telemetry)
        self.assertEqual(
            self._generated_telemetry_value("--minimal_build", "--disable_exceptions"),
            "OFF",
        )

    def test_build_wrappers_use_platform_defaults(self):
        self.assertIn("--no_telemetry", (REPO_DIR / "build.bat").read_text(encoding="utf-8"))
        arm64x_build_bat = (REPO_DIR / "build_arm64x.bat").read_text(encoding="utf-8")
        self.assertEqual(arm64x_build_bat.count("--no_telemetry"), 2)
        build_sh = (REPO_DIR / "build.sh").read_text(encoding="utf-8")
        self.assertNotIn("--use_telemetry", build_sh)
        self.assertNotIn("--no_telemetry", build_sh)

    def test_android_aar_telemetry_defaults_on_and_can_be_disabled(self):
        for build_params, expected in (
            (["--android"], True),
            (["--android", "--no_telemetry"], False),
            (["--android", "--minimal_build", "--disable_exceptions"], False),
        ):
            with self.subTest(build_params=build_params), tempfile.TemporaryDirectory() as temp_dir:
                settings_file = pathlib.Path(temp_dir) / "build-settings.json"
                settings_file.write_text(
                    json.dumps({"build_params": build_params}),
                    encoding="utf-8",
                )
                settings = build_aar_package._parse_build_settings(mock.Mock(build_settings_file=settings_file))
                self.assertEqual(settings["use_telemetry"], expected)


if __name__ == "__main__":
    unittest.main()
