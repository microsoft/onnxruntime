# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Read the common build tool versions and SHA-512 digests."""

from __future__ import annotations

import json
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

with Path(__file__).with_suffix(".json").open(encoding="utf-8") as info_file:
    _BUILD_TOOL_INFO = json.load(info_file)


@dataclass(frozen=True)
class ToolInfo:
    version: str
    sha512: str


def get_tool_info(tool: str, *, host_os: str | None = None, host_arch: str | None = None) -> ToolInfo:
    """Return tool info for the specified host OS and architecture.

    Args:
        tool: Tool name in build_tool_info.json.
        host_os: Host OS in sys.platform format, or None to use sys.platform.
        host_arch: Host architecture in platform.machine() format, or None to
            use platform.machine().

    Returns:
        The tool's version and SHA-512 digest for the specified host.
    """
    info = _BUILD_TOOL_INFO[tool]

    version = info["version"]
    sha512 = info["sha512"]

    if isinstance(sha512, dict):
        host_os = host_os if host_os is not None else sys.platform
        host_arch = host_arch if host_arch is not None else platform.machine()

        os_key = {
            "win32": "windows",
            "linux": "linux",
            "darwin": "macos",
        }.get(host_os)

        arch_key = {
            "amd64": "x64",
            "x86_64": "x64",
            "x64": "x64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }.get(host_arch.lower())

        os_hashes = sha512.get(os_key, {})
        if arch_key is None or not (sha512 := os_hashes.get("universal", os_hashes.get(arch_key))):
            raise ValueError(f"No {tool} SHA-512 configured for host OS '{host_os}' and architecture '{host_arch}'")

    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9._+-]+", version):
        raise ValueError(f"Invalid {tool} version")

    if not isinstance(sha512, str) or not re.fullmatch(r"[0-9a-fA-F]{128}", sha512):
        raise ValueError(f"Invalid {tool} SHA-512")

    return ToolInfo(version, sha512)
