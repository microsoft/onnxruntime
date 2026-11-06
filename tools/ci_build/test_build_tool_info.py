# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_tool_info
from build_tool_info import get_tool_info

TOOL_INFO_SCRIPT = Path(build_tool_info.__file__)
HOST_SHA512_CASES = (
    ("cmake", "linux", "x86_64", "linux", "x64"),
    ("cmake", "linux", "aarch64", "linux", "arm64"),
    ("cmake", "win32", "AMD64", "windows", "x64"),
    ("cmake", "win32", "ARM64", "windows", "arm64"),
    ("cmake", "darwin", "x64", "macos", "universal"),
    ("cmake", "darwin", "arm64", "macos", "universal"),
    ("ccache", "linux", "x64", "linux", "x64"),
)
UNSUPPORTED_CASES = (
    ("cmake", "freebsd", "x64"),
    ("cmake", "linux", "s390x"),
    ("cmake", "darwin", "ppc64"),
    ("ccache", "linux", "arm64"),
    ("ccache", "darwin", "arm64"),
    ("ccache", "win32", "x64"),
)


class TestBuildToolInfo(unittest.TestCase):
    def test_select_host_sha512(self):
        for tool, host_os, host_arch, os_key, arch_key in HOST_SHA512_CASES:
            with self.subTest(tool=tool, host_os=host_os, host_arch=host_arch):
                info = get_tool_info(tool, host_os=host_os, host_arch=host_arch)
                self.assertEqual(info.sha512, build_tool_info._BUILD_TOOL_INFO[tool]["sha512"][os_key][arch_key])

        # vcpkg tool info is host-independent, including for build.py's fallback clone.
        self.assertEqual(
            get_tool_info("vcpkg", host_os="freebsd", host_arch="mips"),
            get_tool_info("vcpkg"),
        )

    def test_reject_unsupported_hosts(self):
        for tool, host_os, host_arch in UNSUPPORTED_CASES:
            with (
                self.subTest(tool=tool, host_os=host_os, host_arch=host_arch),
                self.assertRaisesRegex(ValueError, f"No {tool} SHA-512 configured"),
            ):
                get_tool_info(tool, host_os=host_os, host_arch=host_arch)

    def test_reject_missing_or_invalid_metadata(self):
        metadata = copy.deepcopy(build_tool_info._BUILD_TOOL_INFO)
        del metadata["cmake"]["sha512"]["windows"]["arm64"]
        with (
            patch.object(build_tool_info, "_BUILD_TOOL_INFO", metadata),
            self.assertRaisesRegex(ValueError, "No cmake SHA-512 configured"),
        ):
            get_tool_info("cmake", host_os="win32", host_arch="arm64")

        for field, value in (("sha512", "invalid"), ("version", "invalid\nversion"), ("version", "invalid\n")):
            with self.subTest(field=field):
                metadata = copy.deepcopy(build_tool_info._BUILD_TOOL_INFO)
                metadata["vcpkg"][field] = value
                with (
                    patch.object(build_tool_info, "_BUILD_TOOL_INFO", metadata),
                    self.assertRaisesRegex(ValueError, "Invalid vcpkg"),
                ):
                    get_tool_info("vcpkg")

        with self.assertRaises(KeyError):
            get_tool_info("unknown")


class TestBuildToolInfoCLI(unittest.TestCase):
    def run_cli(self, tools, metadata):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "build_tool_info.py"
            shutil.copyfile(TOOL_INFO_SCRIPT, script)
            info_path = script.with_suffix(".json")
            info_path.write_text(json.dumps(metadata), encoding="utf-8")
            command = [sys.executable, "-I", "-S", str(script)]
            if tools is not None:
                command.extend(("--tools", *tools))
            result = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            outputs = dict(line.split("=", 1) for line in result.stdout.splitlines())
            return result, outputs

    def test_cli_outputs_only_requested_tools(self):
        for tools in (["vcpkg"], ["cmake", "vcpkg"]):
            with self.subTest(tools=tools):
                result, outputs = self.run_cli(tools, build_tool_info._BUILD_TOOL_INFO)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected_keys = set()
                for tool in tools:
                    info = get_tool_info(tool)
                    self.assertEqual(outputs[f"{tool}-version"], info.version)
                    self.assertEqual(outputs[f"{tool}-sha512"], info.sha512)
                    expected_keys.update((f"{tool}-version", f"{tool}-sha512"))
                self.assertEqual(set(outputs), expected_keys)

    def test_cli_rejects_missing_or_unknown_tools(self):
        cases = (
            (None, "required: --tools"),
            ([], "expected at least one argument"),
            ([""], "invalid choice"),
            ([" "], "invalid choice"),
            (["cmake", "unknown"], "invalid choice"),
            (["cmake,vcpkg"], "invalid choice"),
        )
        for tools, expected_error in cases:
            with self.subTest(tools=tools):
                result, outputs = self.run_cli(tools, build_tool_info._BUILD_TOOL_INFO)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)
                self.assertEqual(outputs, {})

    def test_cli_rejects_invalid_metadata_without_partial_outputs(self):
        for field, value in (("sha512", "invalid"), ("version", "invalid\nversion"), ("version", "invalid\n")):
            with self.subTest(field=field):
                metadata = copy.deepcopy(build_tool_info._BUILD_TOOL_INFO)
                metadata["vcpkg"][field] = value
                result, outputs = self.run_cli(["cmake", "vcpkg"], metadata)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Invalid vcpkg", result.stderr)
                self.assertEqual(outputs, {})


if __name__ == "__main__":
    unittest.main()
