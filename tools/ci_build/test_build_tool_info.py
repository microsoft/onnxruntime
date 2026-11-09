# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_tool_info

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
                info = build_tool_info.get_tool_info(tool, host_os=host_os, host_arch=host_arch)
                self.assertEqual(info.sha512, build_tool_info._BUILD_TOOL_INFO[tool]["sha512"][os_key][arch_key])

        # vcpkg tool info is host-independent, including for build.py's fallback clone.
        self.assertEqual(
            build_tool_info.get_tool_info("vcpkg", host_os="freebsd", host_arch="mips"),
            build_tool_info.get_tool_info("vcpkg"),
        )

    def test_reject_unsupported_hosts(self):
        for tool, host_os, host_arch in UNSUPPORTED_CASES:
            with (
                self.subTest(tool=tool, host_os=host_os, host_arch=host_arch),
                self.assertRaisesRegex(ValueError, f"No {tool} SHA-512 configured"),
            ):
                build_tool_info.get_tool_info(tool, host_os=host_os, host_arch=host_arch)

    def test_reject_missing_or_invalid_metadata(self):
        metadata = copy.deepcopy(build_tool_info._BUILD_TOOL_INFO)
        del metadata["cmake"]["sha512"]["windows"]["arm64"]
        with (
            patch.object(build_tool_info, "_BUILD_TOOL_INFO", metadata),
            self.assertRaisesRegex(ValueError, "No cmake SHA-512 configured"),
        ):
            build_tool_info.get_tool_info("cmake", host_os="win32", host_arch="arm64")

        for field, value in (("sha512", "invalid"), ("version", "invalid\nversion"), ("version", "invalid\n")):
            with self.subTest(field=field):
                metadata = copy.deepcopy(build_tool_info._BUILD_TOOL_INFO)
                metadata["vcpkg"][field] = value
                with (
                    patch.object(build_tool_info, "_BUILD_TOOL_INFO", metadata),
                    self.assertRaisesRegex(ValueError, "Invalid vcpkg"),
                ):
                    build_tool_info.get_tool_info("vcpkg")

        with self.assertRaises(KeyError):
            build_tool_info.get_tool_info("unknown")


if __name__ == "__main__":
    unittest.main()
