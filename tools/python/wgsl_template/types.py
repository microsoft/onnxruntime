# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Shared data structures for the WGSL template engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeReference:
    """Where a parsed line came from in the original source."""

    file_path: str
    line_number: int  # 1-based


@dataclass
class ParsedLine:
    """A line after parsing, plus its origin."""

    line: str
    code_reference: CodeReference


@dataclass
class TemplatePass0:
    """Result of PASS0 (load): raw lines straight from disk."""

    file_path: str  # absolute resolved path on disk
    raw: list[str]  # split on /\r?\n/


@dataclass
class TemplatePass1:
    """Result of PASS1 (parse): pre-processed but not yet code-generated."""

    file_path: str
    raw: list[str]
    pass1: list[ParsedLine]


@dataclass
class GenerateResult:
    """Result of running PASS2 over a single template."""

    code: str
    params: dict[str, str] = field(default_factory=dict)  # name -> param-type
    variables: dict[str, str] = field(default_factory=dict)  # name -> variable-type
    has_main_function: bool = False


@dataclass
class TemplatePass2:
    """Result of PASS2: code-generated string + metadata."""

    file_path: str
    generate_result: GenerateResult


@dataclass
class TemplateRepository:
    """A collection of templates loaded from disk.

    ``templates`` may hold ``TemplatePass0``, ``TemplatePass1``, or
    ``TemplatePass2`` depending on which pipeline stage produced it.
    """

    base_path: str
    templates: dict[str, object]


# ----------------------------------------------------------------------
# Source directory specification (loader input)
# ----------------------------------------------------------------------


@dataclass
class SourceDir:
    """A source directory with an optional alias prefix.

    The CLI doesn't expose aliases, but the loader API still supports
    them so directory-aliased test fixtures keep working.
    """

    path: str
    alias: str | None = None
