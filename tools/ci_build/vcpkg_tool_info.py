"""Accessors for the vcpkg tool release supported by ONNX Runtime."""

from __future__ import annotations

import json
from pathlib import Path

with Path(__file__).with_suffix(".json").open(encoding="utf-8") as info_file:
    _VCPKG_TOOL_INFO: dict[str, str] = json.load(info_file)


def get_vcpkg_release_tag() -> str:
    """Returns the Git tag of the vcpkg release supported by ONNX Runtime."""
    return _VCPKG_TOOL_INFO["release_tag"]


def get_vcpkg_sha512() -> str:
    """Returns the SHA-512 digest of the vcpkg release supported by ONNX Runtime."""
    return _VCPKG_TOOL_INFO["sha512"]
