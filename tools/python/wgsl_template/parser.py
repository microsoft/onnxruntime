# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""PASS1: comment stripping, #include expansion, #define substitution.

Three sub-passes run in order on each top-level template:

1. :func:`_strip_comments` — replace ``//`` and ``/* */`` regions with
   empty text while preserving line count (every line is kept, even if
   it becomes empty). Not string-aware.
2. :func:`_expand_includes` — flatten ``#include "name"`` directives,
   detect cycles (incl. self-include), reject missing files.
3. :func:`_apply_macros` — process ``#define`` directives. Expansion
   is eager at definition time, whole-word match, and rejects
   self-reference / mutual circular / duplicate / empty value.
"""

from __future__ import annotations

import re

from .errors import WgslTemplateParseError
from .types import (
    CodeReference,
    ParsedLine,
    TemplatePass0,
    TemplatePass1,
    TemplateRepository,
)

_INCLUDE_RE = re.compile(r"^\s*#include\s+(.+)$")
_DEFINE_FULL_RE = re.compile(r"^\s*#define\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(.+)$")
_DEFINE_EMPTY_RE = re.compile(r"^\s*#define\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$")
_DEFINE_INVALID_NAME_RE = re.compile(r"^\s*#define\s+(\S+)(?:\s+(.+))?$")


# ----------------------------------------------------------------------
# Sub-pass 1: comment stripping
# ----------------------------------------------------------------------


def _strip_comments(file_path: str, raw: list[str]) -> list[ParsedLine]:
    """Strip ``//`` and ``/* */`` comments, returning one
    :class:`ParsedLine` per input line.

    Each line's :class:`CodeReference` line number is bound to its index
    in ``raw`` (the original source), so errors and ``--preserve-code-ref``
    output point at the true source line regardless of comment removal.

    NOT string-aware. WGSL has no string type, but template arguments
    like ``getElementAt("a/*b", "c")`` would be mis-stripped.
    """

    out: list[ParsedLine] = []
    in_multi = False

    for source_index, line in enumerate(raw):
        processed: list[str] = []
        i = 0
        n = len(line)

        while i < n:
            if in_multi:
                # Look for end of multi-line comment.
                if i < n - 1 and line[i] == "*" and line[i + 1] == "/":
                    in_multi = False
                    i += 2
                else:
                    i += 1
            else:
                # Single-line comment: rest of line is a comment.
                if i < n - 1 and line[i] == "/" and line[i + 1] == "/":
                    break
                # Multi-line comment start.
                if i < n - 1 and line[i] == "/" and line[i + 1] == "*":
                    in_multi = True
                    i += 2
                else:
                    processed.append(line[i])
                    i += 1

        # Always append (even if empty) to preserve line count, and tag
        # each line with its true source line number (1-based).
        out.append(
            ParsedLine(
                line="".join(processed).rstrip(),
                code_reference=CodeReference(
                    file_path=file_path,
                    line_number=source_index + 1,
                ),
            )
        )

    if in_multi:
        raise WgslTemplateParseError(
            "Unterminated multi-line comment detected in template",
            "comment-removal",
            file_path=file_path,
        )

    return out


# ----------------------------------------------------------------------
# Sub-pass 2: #include expansion
# ----------------------------------------------------------------------


class _ParseEntry:
    __slots__ = ("include_processed", "lines")

    def __init__(self, lines: list[ParsedLine]) -> None:
        self.lines: list[ParsedLine] = lines
        self.include_processed: bool = False


def _expand_includes(
    parse_state: dict[str, _ParseEntry],
    include_stack: list[str],
) -> None:
    """Recursively expand ``#include`` directives in place.

    Mutates ``parse_state[include_stack[-1]].lines`` to contain the
    flattened content. Sets ``include_processed`` so repeat includes
    are no-ops.
    """

    current_file = include_stack[-1]
    current = parse_state.get(current_file)
    if current is None:
        raise WgslTemplateParseError(
            f'File "{current_file}" not found in parse state',
            "include-resolution",
            file_path=current_file,
        )

    if current.include_processed:
        return

    flattened: list[ParsedLine] = []
    for line_index, parsed_line in enumerate(current.lines):
        line = parsed_line.line
        m = _INCLUDE_RE.match(line)
        if m is None:
            flattened.append(parsed_line)
            continue

        include_param = m.group(1).strip()
        if not (include_param.startswith('"') and include_param.endswith('"')):
            raise WgslTemplateParseError(
                f"Invalid #include directive in file {current_file} at line "
                f"{line_index + 1}: file path must be enclosed in double quotes",
                "syntax-error",
                file_path=current_file,
                line_number=line_index + 1,
            )

        include_path = include_param[1:-1]

        if include_path in include_stack:
            raise WgslTemplateParseError(
                f"Circular #include detected in file {current_file} at line "
                f"{line_index + 1}: {include_path} is already included",
                "include-circular",
                file_path=current_file,
                line_number=line_index + 1,
            )
        if include_path not in parse_state:
            raise WgslTemplateParseError(
                f'File "{include_path}" not found in parse state for '
                f'#include directive in file "{current_file}" at line '
                f"{line_index + 1}",
                "include-not-found",
                file_path=current_file,
                line_number=line_index + 1,
            )

        include_stack.append(include_path)
        _expand_includes(parse_state, include_stack)
        flattened.extend(parse_state[include_path].lines)
        include_stack.pop()

    current.include_processed = True
    current.lines = flattened


# ----------------------------------------------------------------------
# Sub-pass 3: #define substitution
# ----------------------------------------------------------------------


def _apply_macros(lines: list[ParsedLine], file_name: str) -> list[ParsedLine]:
    """Process ``#define`` directives and apply substitutions.

    Rules:
      * File-scoped per top-level template (after include flattening).
      * Eager expansion at definition time: ``#define X Y`` stores the
        already-expanded ``Y``. Forward references do not work.
      * Whole-word match (``\\bNAME\\b``).
      * Reject self-reference, mutual circular dependencies,
        duplicate definitions, empty / whitespace-only values.
    """

    macros: dict[str, str] = {}
    out: list[ParsedLine] = []

    for line_index, parsed_line in enumerate(lines):
        line = parsed_line.line

        if line.lstrip().startswith("#define "):
            full = _DEFINE_FULL_RE.match(line)
            if full is None:
                # Try to give a more specific error message.
                empty = _DEFINE_EMPTY_RE.match(line)
                if empty is not None:
                    raise WgslTemplateParseError(
                        f"Invalid macro definition in file {file_name} at "
                        f'line {line_index + 1}: macro "{empty.group(1)}" '
                        f"has no value",
                        "syntax-error",
                        file_path=file_name,
                        line_number=line_index + 1,
                    )

                invalid = _DEFINE_INVALID_NAME_RE.match(line)
                if invalid is not None:
                    raise WgslTemplateParseError(
                        f"Invalid macro definition in file {file_name} at "
                        f"line {line_index + 1}: invalid macro name "
                        f'"{invalid.group(1)}" (must start with letter or '
                        f"underscore, contain only letters, numbers, and "
                        f"underscores)",
                        "syntax-error",
                        file_path=file_name,
                        line_number=line_index + 1,
                    )

                raise WgslTemplateParseError(
                    f"Invalid macro definition in file {file_name} at line "
                    f"{line_index + 1}: malformed #define directive",
                    "syntax-error",
                    file_path=file_name,
                    line_number=line_index + 1,
                )

            macro_name = full.group(1)
            macro_value = full.group(2).strip()

            if macro_value == "":
                raise WgslTemplateParseError(
                    f"Invalid macro definition in file {file_name} at line "
                    f'{line_index + 1}: macro "{macro_name}" has no value',
                    "syntax-error",
                    file_path=file_name,
                    line_number=line_index + 1,
                )

            if macro_name in macros:
                raise WgslTemplateParseError(
                    f"Duplicate macro definition in file {file_name} at line "
                    f'{line_index + 1}: "{macro_name}" is already defined',
                    "define-expansion",
                    file_path=file_name,
                    line_number=line_index + 1,
                )

            # Direct self-reference check.
            self_ref = re.compile(rf"\b{re.escape(macro_name)}\b")
            if self_ref.search(macro_value):
                raise WgslTemplateParseError(
                    f"Circular macro reference in file {file_name} at line "
                    f'{line_index + 1}: macro "{macro_name}" references '
                    f"itself",
                    "define-expansion",
                    file_path=file_name,
                    line_number=line_index + 1,
                )

            # Apply existing macros to the new value (eager expansion).
            expanded_value = macro_value
            for existing_name, existing_value in macros.items():
                regex = re.compile(rf"\b{re.escape(existing_name)}\b")
                if regex.search(expanded_value):
                    # Mutual circular check: existing macro references
                    # the new one.
                    circ = re.compile(rf"\b{re.escape(macro_name)}\b")
                    if circ.search(existing_value):
                        raise WgslTemplateParseError(
                            f"Circular macro reference in file {file_name} "
                            f"at line {line_index + 1}: macro "
                            f'"{macro_name}" creates circular dependency '
                            f'with "{existing_name}"',
                            "define-expansion",
                            file_path=file_name,
                            line_number=line_index + 1,
                        )
                    expanded_value = regex.sub(existing_value, expanded_value)

            macros[macro_name] = expanded_value
            line = ""  # Clear the #define line in the output.
        else:
            # Apply macro substitutions to the current line.
            for macro_name, macro_value in macros.items():
                regex = re.compile(rf"\b{re.escape(macro_name)}\b")
                line = regex.sub(macro_value, line)

        out.append(ParsedLine(line=line, code_reference=parsed_line.code_reference))

    return out


# ----------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------


def parse(repo: TemplateRepository) -> TemplateRepository:
    """Run PASS1 on every template in the repository."""

    # STEP 1 — strip comments. _strip_comments returns ParsedLine
    # objects already tagged with their true source line numbers.
    parse_state: dict[str, _ParseEntry] = {}
    for template_key, template in repo.templates.items():
        assert isinstance(template, TemplatePass0)
        parsed_lines = _strip_comments(template_key, template.raw)
        parse_state[template_key] = _ParseEntry(parsed_lines)

    # STEP 2 — expand #include directives in every top-level template.
    pass1_templates: dict[str, object] = {}
    for template_key, template in repo.templates.items():
        assert isinstance(template, TemplatePass0)
        _expand_includes(parse_state, [template_key])
        pass1_templates[template_key] = TemplatePass1(
            file_path=template.file_path,
            raw=template.raw,
            pass1=parse_state[template_key].lines,
        )

    # STEP 3 — apply #define substitutions on the post-include result.
    for template_key in list(pass1_templates):
        template = pass1_templates[template_key]
        assert isinstance(template, TemplatePass1)
        processed = _apply_macros(list(template.pass1), template_key)
        pass1_templates[template_key] = TemplatePass1(
            file_path=template.file_path,
            raw=template.raw,
            pass1=processed,
        )

    return TemplateRepository(
        base_path=repo.base_path,
        templates=pass1_templates,
    )
