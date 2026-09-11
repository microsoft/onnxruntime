# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Code generators that emit C++ from PASS2 results.

The single :class:`StaticCodeGenerator` covers both ``static-cpp`` and
``static-cpp-literal`` (the difference is whether a string table is
used).
"""

from __future__ import annotations

from ..errors import WgslTemplateGenerateError
from .static_cpp import (
    CodeSegment,
    CodeSegmentArg,
    StaticCodeGenerator,
)


def resolve_code_generator(name: str) -> StaticCodeGenerator:
    """Resolve a generator name to a fresh code-generator instance.

    Only the static C++ generators are supported; the dynamic generator
    is provided by the npm path and is not implemented here.
    """
    if name == "static-cpp":
        return StaticCodeGenerator(use_string_table=True)
    if name == "static-cpp-literal":
        return StaticCodeGenerator(use_string_table=False)
    raise WgslTemplateGenerateError(
        f"Unknown code generator: {name}",
        "generator-not-found",
    )


__all__ = [
    "CodeSegment",
    "CodeSegmentArg",
    "StaticCodeGenerator",
    "resolve_code_generator",
]
