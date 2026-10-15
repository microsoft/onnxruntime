# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shared utilities for the WebGPU plugin EP packaging scripts. Not a public API."""

from __future__ import annotations

import re
from pathlib import Path

# Matches "@var@" template variables.
_TEMPLATE_VARIABLE_PATTERN = re.compile(r"@(\w+)@")

PLUGIN_BINARIES = {
    "Windows": (Path("onnxruntime_providers_webgpu.dll"), Path("dxcompiler.dll")),
    "Linux": (Path("libonnxruntime_providers_webgpu.so"),),
    "Darwin": (Path("libonnxruntime_providers_webgpu.dylib"),),
}

WINDOWS_AGILITY_SDK_BINARIES = (Path("D3D12/D3D12Core.dll"), Path("D3D12/d3d12SDKLayers.dll"))


def get_plugin_binaries(target_platform: str, *, include_agility_sdk_files: bool = False) -> tuple[Path, ...]:
    """Return the required plugin binaries for a target platform.

    `target_platform` uses `platform.system()` names such as "Windows", "Linux", and "Darwin".
    """
    try:
        binaries = list(PLUGIN_BINARIES[target_platform])
    except KeyError as e:
        raise ValueError(f"Unsupported WebGPU packaging platform: {target_platform}") from e

    if target_platform == "Windows" and include_agility_sdk_files:
        binaries.extend(WINDOWS_AGILITY_SDK_BINARIES)

    return tuple(binaries)


def get_windows_agility_sdk_state(binary_dir: Path) -> tuple[bool, tuple[Path, ...]]:
    """Return whether the full Agility SDK pair is present and which files are missing.

    If neither file is present, returns ``(False, ())``.
    If both files are present, returns ``(True, ())``.
    If only a subset is present, returns ``(False, missing_files)`` so callers can fail loudly.
    """
    present = tuple((binary_dir / relative_path).is_file() for relative_path in WINDOWS_AGILITY_SDK_BINARIES)
    if all(present):
        return True, ()
    if any(present):
        missing = tuple(
            relative_path for relative_path, is_present in zip(WINDOWS_AGILITY_SDK_BINARIES, present) if not is_present
        )
        return False, missing
    return False, ()


def gen_file_from_template(
    template_file: Path, output_file: Path, variable_substitutions: dict[str, str], strict: bool = True
) -> None:
    """Generate a file from a template by substituting "@var@" markers with provided values.

    If `strict` is True, raises ValueError when the set of "@var@" names found in the template
    does not match the keys of `variable_substitutions`.

    Note: substituted values are inserted verbatim with no awareness of the target file's syntax.
    The caller is responsible for any quoting/escaping required by the target format.
    """
    content = template_file.read_text(encoding="utf-8")

    variables_in_file: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        variables_in_file.add(name)
        return variable_substitutions.get(name, match.group(0))

    content = _TEMPLATE_VARIABLE_PATTERN.sub(replace, content)

    if strict and variables_in_file != variable_substitutions.keys():
        provided = set(variable_substitutions.keys())
        raise ValueError(
            f"Template variables and substitution keys do not match for {template_file}. "
            f"Only in template: {sorted(variables_in_file - provided)}. "
            f"Only in substitutions: {sorted(provided - variables_in_file)}."
        )

    output_file.write_text(content, encoding="utf-8")
