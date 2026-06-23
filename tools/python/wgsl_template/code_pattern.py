# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Built-in code patterns and indices-helper rewrite table.

A :class:`CodePattern` is a single rewrite rule consumed by the PASS2
generator. ``DEFAULT_PATTERNS`` are always active; the named entries
in :func:`lookup_pattern` are opt-in via ``#use NAME`` directives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern

# ----------------------------------------------------------------------
# Pattern data class
# ----------------------------------------------------------------------


@dataclass
class CodePattern:
    """A single rewrite rule for the PASS2 generator.

    ``type`` selects how the matcher uses this pattern:
      * "control"  — punctuation tokens: ``(``, ``)``, ``,``, ``{``,
                     ``}``, ``$MAIN {``.
      * "param"    — parameter name introduced by ``#param``.
      * "variable" — bare shader variable identifier.
      * "function" — function call: name followed by ``(``.
      * "method"   — method call on a shader variable: ``var.name(...)``.
      * "property" — property access on a shader variable: ``var.prop``.
    """

    type: str
    pattern: str | Pattern[str]

    # If present, replaces matched group(s). For function/method/property
    # patterns: index 0 replaces the receiver capture (group 1), index 1
    # replaces the name capture (group 2). A ``None`` element means
    # "leave that group as-is".
    replace: str | list[str | None] | None = None

    variable_type: str | None = None  # "shader-variable"
    param_type: str | None = None  # "int"
    arg_types: list[str] = field(default_factory=list)  # "expression"|"string"|"auto"


# ----------------------------------------------------------------------
# Default control-token patterns (always active)
# ----------------------------------------------------------------------


DEFAULT_PATTERNS: list[CodePattern] = [
    CodePattern(type="control", pattern=re.compile(r"\(")),
    CodePattern(type="control", pattern=re.compile(r"\)")),
    CodePattern(type="control", pattern=re.compile(r",")),
    CodePattern(type="control", pattern=re.compile(r"\{")),
    CodePattern(type="control", pattern=re.compile(r"\}")),
    CodePattern(
        type="control",
        pattern=re.compile(r"(?<![a-zA-Z0-9_])(\$MAIN\s*\{)"),
    ),
]


# ----------------------------------------------------------------------
# Built-in patterns (opt-in via #use NAME)
# ----------------------------------------------------------------------


_BUILT_IN_PATTERNS: list[tuple[str, CodePattern]] = [
    (
        "guardAgainstOutOfBoundsWorkgroupSizes",
        CodePattern(
            type="function",
            pattern=re.compile(r"\b(guardAgainstOutOfBoundsWorkgroupSizes)\s*\("),
            replace=["shader_helper.GuardAgainstOutOfBoundsWorkgroupSizes"],
            arg_types=["string"],
        ),
    ),
    (
        "getElementAt",
        CodePattern(
            type="function",
            pattern=re.compile(r"\b(getElementAt)\s*\("),
            replace=["GetElementAt"],
            arg_types=["string", "auto", "expression", "expression"],
        ),
    ),
]


# ----------------------------------------------------------------------
# Indices-helper patterns
# ----------------------------------------------------------------------


_INDICES_HELPER_PATTERNS: list[tuple[str, CodePattern]] = [
    (
        ".rank",
        CodePattern(
            type="property",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(rank)\b"),
            replace=[None, "Rank()"],
        ),
    ),
    (
        ".numComponents",
        CodePattern(
            type="property",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(numComponents)\b"),
            replace=[None, "NumComponents()"],
        ),
    ),
    (
        ".offsetToIndices",
        CodePattern(
            type="method",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(offsetToIndices)\s*\("),
            replace=[None, "OffsetToIndices"],
            arg_types=["string"],
        ),
    ),
    (
        ".indicesToOffset",
        CodePattern(
            type="method",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(indicesToOffset)\s*\("),
            replace=[None, "IndicesToOffset"],
            arg_types=["string"],
        ),
    ),
    (
        ".indicesSet",
        CodePattern(
            type="method",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(indicesSet)\s*\("),
            replace=[None, "IndicesSet"],
            arg_types=["string", "auto", "auto"],
        ),
    ),
    (
        ".indicesGet",
        CodePattern(
            type="method",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(indicesGet)\s*\("),
            replace=[None, "IndicesGet"],
            arg_types=["string", "auto"],
        ),
    ),
    (
        ".set",
        CodePattern(
            type="method",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(set)\s*\("),
            replace=[None, "Set"],
        ),
    ),
    (
        ".setByOffset",
        CodePattern(
            type="method",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(setByOffset)\s*\("),
            replace=[None, "SetByOffset"],
        ),
    ),
    (
        ".setByIndices",
        CodePattern(
            type="method",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(setByIndices)\s*\("),
            replace=[None, "SetByIndices"],
            arg_types=["string", "string"],
        ),
    ),
    (
        ".get",
        CodePattern(
            type="method",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(get)\s*\("),
            replace=[None, "Get"],
        ),
    ),
    (
        ".getByOffset",
        CodePattern(
            type="method",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(getByOffset)\s*\("),
            replace=[None, "GetByOffset"],
        ),
    ),
    (
        ".getByIndices",
        CodePattern(
            type="method",
            pattern=re.compile(r"\b([_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(getByIndices)\s*\("),
            replace=[None, "GetByIndices"],
            arg_types=["string"],
        ),
    ),
]


_PATTERN_MAP: dict[str, CodePattern] = dict(_BUILT_IN_PATTERNS + _INDICES_HELPER_PATTERNS)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def lookup_pattern(name: str) -> CodePattern | None:
    """Look up a built-in pattern by ``#use`` name. Returns ``None`` if
    the name isn't recognized."""
    return _PATTERN_MAP.get(name)


def create_param_pattern(name: str) -> CodePattern:
    """Create a runtime ``#param`` pattern that matches the identifier
    as a whole word. Raises ``ValueError`` if the name is not a valid
    identifier."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f'Invalid parameter identifier: "{name}". Parameter names '
            f"must start with a letter or underscore and contain only "
            f"letters, digits, and underscores."
        )
    return CodePattern(type="param", pattern=re.compile(rf"\b{re.escape(name)}\b"))
