# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import contextlib
import io
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase, main, mock

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
original_sys_path = sys.path.copy()
try:
    sys.path[:0] = [
        str(TOOLS_DIR / "python"),
        str(SCRIPT_DIR),
        str(SCRIPT_DIR / "github" / "android"),
    ]
    import build_aar_package
    from build_args import parse_arguments, target_supports_telemetry
finally:
    sys.path[:] = original_sys_path


class TelemetryBuildArgsTest(TestCase):
    @staticmethod
    def _target(**overrides):
        values = {
            "android": False,
            "build_wasm": False,
            "disable_exceptions": False,
            "macos": None,
            "rv64": False,
            "tvos": False,
            "visionos": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _parse(self, *build_args):
        with (
            mock.patch.object(
                sys, "argv", ["build.py", "--build_dir", "build", "--skip_tests", "--update", *build_args]
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return parse_arguments()

    def test_telemetry_enabled_by_default(self):
        with mock.patch("build_args.target_supports_telemetry", return_value=True):
            self.assertTrue(self._parse().use_telemetry)

    def test_no_telemetry_disables_telemetry(self):
        self.assertFalse(self._parse("--no_telemetry").use_telemetry)

    def test_unsupported_build_modes_disable_telemetry(self):
        for build_args in (
            ("--build_wasm",),
            ("--build_wasm_static_lib",),
            ("--minimal_build", "--disable_exceptions"),
            ("--rv64",),
        ):
            with self.subTest(build_args=build_args):
                self.assertFalse(self._parse(*build_args).use_telemetry)

    def test_unsupported_apple_targets_disable_telemetry(self):
        for target, expected in (
            (self._target(), True),
            (self._target(visionos=True), False),
            (self._target(tvos=True), False),
            (self._target(macos="Catalyst"), False),
        ):
            with self.subTest(target=target):
                self.assertEqual(target_supports_telemetry(target), expected)

    def test_unsupported_hosts_disable_telemetry(self):
        with (
            mock.patch("build_args.is_windows", return_value=False),
            mock.patch("build_args.is_macOS", return_value=False),
            mock.patch("build_args.is_linux", return_value=False),
        ):
            self.assertFalse(target_supports_telemetry(self._target()))

    def test_linux_telemetry_architecture_allowlist(self):
        with (
            mock.patch("build_args.is_windows", return_value=False),
            mock.patch("build_args.is_macOS", return_value=False),
            mock.patch("build_args.is_linux", return_value=True),
        ):
            with mock.patch("build_args.platform.machine", return_value="x86_64"):
                self.assertTrue(target_supports_telemetry(self._target()))
            with mock.patch("build_args.platform.machine", return_value="riscv64"):
                self.assertFalse(target_supports_telemetry(self._target()))

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
    main()
