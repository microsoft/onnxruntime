# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Shared utilities for the WebGPU plugin EP packaging scripts. Not a public API."""

from __future__ import annotations

import re
from pathlib import Path

# Matches "@var@" template variables.
_TEMPLATE_VARIABLE_PATTERN = re.compile(r"@(\w+)@")


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
