# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""PASS2: pattern matching, directives, and code generation.

The state machine in :func:`_generate_impl` tracks parentheses,
brackets, a function-call stack, the ``$MAIN`` context, and a separate
output bucket for ``#if``/``#elif`` condition expressions. ``#if``
expressions go through the same pattern matcher as the body but their
output is buffered and flushed between ``if (`` and ``) {\n``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .code_generator.static_cpp import (
    CodeSegment,
    CodeSegmentArg,
    StaticCodeGenerator,
)
from .code_pattern import (
    DEFAULT_PATTERNS,
    CodePattern,
    create_param_pattern,
    lookup_pattern,
)
from .errors import WgslTemplateGenerateError
from .types import (
    GenerateResult,
    ParsedLine,
    TemplatePass1,
    TemplatePass2,
    TemplateRepository,
)

# ----------------------------------------------------------------------
# Generator state
# ----------------------------------------------------------------------


@dataclass
class _FunctionCallState:
    name: str
    parentheses_state: int
    params: list[list[CodeSegment]] = field(default_factory=list)
    current_param: list[CodeSegment] = field(default_factory=list)
    caller: str | None = None
    arg_types: list[str] = field(default_factory=list)


# preprocessIfStack entry: (type, init_paren, init_bracket, prev_paren, prev_bracket)
_IfStackEntry = list[object]


@dataclass
class _GeneratorState:
    repo: TemplateRepository
    pass1: list[ParsedLine]
    code_generator: StaticCodeGenerator
    file_path: str

    current_line: int = 0
    current_column: int = 0
    preprocess_if_stack: list[_IfStackEntry] = field(default_factory=list)
    patterns: list[CodePattern] = field(default_factory=lambda: list(DEFAULT_PATTERNS))
    current_function_call: list[_FunctionCallState] = field(default_factory=list)
    current_parentheses_state: int = 0
    main_function: str = "not-started"  # "not-started" | "started" | "ended"
    current_bracket_state: int = 0
    result: list[CodeSegment] = field(default_factory=list)
    used_params: dict[str, str] = field(default_factory=dict)  # name -> param-type
    used_variables: dict[str, str] = field(default_factory=dict)  # name -> variable-type


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _merge_adjacent_segments(segments: list[CodeSegment]) -> list[CodeSegment]:
    """Merge runs of adjacent ``raw`` or ``code`` segments to keep the
    final ``emit`` output compact."""
    out: list[CodeSegment] = []
    for seg in segments:
        if out and out[-1].type == seg.type and seg.type in ("raw", "code"):
            out[-1].content += seg.content
        else:
            out.append(CodeSegment(type=seg.type, content=seg.content))
    return out


def _match_next_pattern(
    content: str, patterns: list[CodePattern]
) -> tuple[CodePattern, int, int, re.Match[str] | None] | None:
    """Find the pattern that starts earliest in ``content``.

    Returns ``(pattern, start_index, length, match)`` or ``None``.
    """
    earliest: tuple[CodePattern, int, int, re.Match] | None = None
    for pattern in patterns:
        regex = pattern.pattern
        if isinstance(regex, str):
            regex = re.compile(regex)
        m = regex.search(content)
        if m is None:
            continue
        idx = m.start()
        if earliest is None or idx < earliest[1]:
            earliest = (pattern, idx, len(m.group(0)), m)
    return earliest


# ----------------------------------------------------------------------
# Core generator
# ----------------------------------------------------------------------


def _generate_impl(state: _GeneratorState, preserve_code_reference: bool) -> None:
    pass1 = state.pass1
    cg = state.code_generator
    if_stack = state.preprocess_if_stack
    patterns = state.patterns
    fn_call = state.current_function_call

    # current_pre_processor_expression:
    #   None  - normal output flow
    #   list  - buffering segments for an #if/#elif condition
    pre_processor_expression: list[CodeSegment] | None = None

    def output(typ: str, content: str) -> None:
        nonlocal pre_processor_expression
        seg = CodeSegment(type=typ, content=content)
        if pre_processor_expression is not None:
            if typ == "raw":
                raise WgslTemplateGenerateError(
                    f"Raw content inside preprocessor expression at line "
                    f"{state.current_line + 1}, column {state.current_column}",
                    "code-generation-failed",
                    file_path=state.file_path,
                    line_number=state.current_line + 1,
                )
            elif typ == "code" and content == "\n":
                # End of expression — newline closes the #if line.
                pass
            else:
                seg.type = "raw"  # convert to raw inside the expression
                pre_processor_expression.append(seg)
            return

        if fn_call:
            if typ == "raw":
                raise WgslTemplateGenerateError(
                    f"Raw content inside function call at line {state.current_line + 1}, column {state.current_column}",
                    "code-generation-failed",
                    file_path=state.file_path,
                    line_number=state.current_line + 1,
                )
            fn_call[-1].current_param.append(seg)
        else:
            state.result.append(seg)

    def process_pattern_match(line: str, restline: str, next_match) -> None:
        """Handle one pattern match. Mutates state in place."""
        nonlocal pre_processor_expression
        pattern, start_idx, length, match = next_match

        matched = line[state.current_column + start_idx : state.current_column + start_idx + length]

        # Advance past the matched text.
        state.current_column += start_idx + length

        caller: str | None = None

        ptype = pattern.type

        if ptype == "control":
            if matched == "(":
                state.current_parentheses_state += 1
                output("code", "(")
            elif matched == ",":
                if fn_call and fn_call[-1].parentheses_state + 1 == state.current_parentheses_state:
                    call = fn_call[-1]
                    if not call.current_param:
                        raise WgslTemplateGenerateError(
                            f"Empty parameter at line {state.current_line + 1}, column {state.current_column}",
                            "parameter-missing",
                            file_path=state.file_path,
                            line_number=state.current_line + 1,
                        )
                    call.params.append(call.current_param)
                    call.current_param = []
                else:
                    output("code", ",")
            elif matched == ")":
                state.current_parentheses_state -= 1
                if state.current_parentheses_state < 0:
                    raise WgslTemplateGenerateError(
                        f"Unmatched closing parenthesis at line "
                        f"{state.current_line + 1}, column "
                        f"{state.current_column}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                if fn_call and fn_call[-1].parentheses_state == state.current_parentheses_state:
                    call = fn_call.pop()
                    if call.current_param:
                        call.params.append(call.current_param)
                    params = [_merge_adjacent_segments(p) for p in call.params]
                    # Trim leading/trailing whitespace-only or whitespace
                    # text from each parameter.
                    for param in params:
                        if param and param[-1].type == "code":
                            if param[-1].content.strip() == "":
                                param.pop()
                            else:
                                param[-1].content = param[-1].content.rstrip()
                        if param and param[0].type == "code":
                            if param[0].content.strip() == "":
                                param.pop(0)
                            else:
                                param[0].content = param[0].content.lstrip()

                    code_args = [
                        CodeSegmentArg(
                            type=(call.arg_types[i] if i < len(call.arg_types) else "auto"),
                            code=p,
                        )
                        for i, p in enumerate(params)
                    ]

                    if call.caller:
                        emitted = cg.method(call.caller, call.name, code_args)
                    else:
                        emitted = cg.function(call.name, code_args)
                    output("expression", emitted)
                else:
                    output("code", ")")
            elif matched == "{":
                state.current_bracket_state += 1
                output("code", "{")
            elif matched == "}":
                state.current_bracket_state -= 1
                if state.current_bracket_state < 0:
                    raise WgslTemplateGenerateError(
                        f"Unmatched closing bracket at line {state.current_line + 1}, column {state.current_column}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                if state.current_bracket_state == 0 and state.main_function == "started":
                    output("raw", "MainFunctionEnd();\n")
                    state.main_function = "ended"
                else:
                    output("code", "}")
            elif "$MAIN" in matched:
                if state.main_function != "not-started":
                    raise WgslTemplateGenerateError(
                        f"Multiple main function start ($MAIN) detected at "
                        f"line {state.current_line + 1}, column "
                        f"{state.current_column}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                if fn_call:
                    raise WgslTemplateGenerateError(
                        f"$MAIN directive inside function call at line "
                        f"{state.current_line + 1}, column "
                        f"{state.current_column}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                if state.current_parentheses_state != 0:
                    raise WgslTemplateGenerateError(
                        f"$MAIN directive inside parentheses at line "
                        f"{state.current_line + 1}, column "
                        f"{state.current_column}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                if state.current_bracket_state != 0:
                    raise WgslTemplateGenerateError(
                        f"$MAIN directive inside brackets at line "
                        f"{state.current_line + 1}, column "
                        f"{state.current_column}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                if pre_processor_expression is not None:
                    raise WgslTemplateGenerateError(
                        f"$MAIN directive inside preprocessor expression "
                        f"at line {state.current_line + 1}, column "
                        f"{state.current_column}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                state.main_function = "started"
                output("raw", "MainFunctionStart();\n")
                state.current_bracket_state = 1

        elif ptype == "param":
            state.used_params[matched] = pattern.param_type or "int"
            output("expression", cg.param(matched))

        elif ptype == "variable":
            variable_name = matched
            if isinstance(pattern.replace, list) and pattern.replace and pattern.replace[0]:
                variable_name = pattern.replace[0]
            state.used_variables[variable_name] = pattern.variable_type or "shader-variable"
            output("expression", cg.variable(variable_name))

        elif ptype in ("method", "function"):
            # Method-call patterns first extract their receiver, then
            # fall through to the same logic as a plain function call.
            if ptype == "method":
                if isinstance(pattern.replace, list) and pattern.replace and pattern.replace[0]:
                    caller = pattern.replace[0]
                else:
                    span = match.span(1)
                    caller = restline[span[0] : span[1]]
                state.used_variables[caller] = pattern.variable_type or "shader-variable"

            # Resolve function/method name.
            replace_index = 1 if caller else 0
            if (
                isinstance(pattern.replace, list)
                and len(pattern.replace) > replace_index
                and pattern.replace[replace_index]
            ):
                name = pattern.replace[replace_index]
            else:
                group_index = 2 if caller else 1
                span = match.span(group_index)
                name = restline[span[0] : span[1]]

            fn_call.append(
                _FunctionCallState(
                    name=name,
                    parentheses_state=state.current_parentheses_state,
                    caller=caller,
                    arg_types=list(pattern.arg_types) if pattern.arg_types else [],
                )
            )
            state.current_parentheses_state += 1

        elif ptype == "property":
            if isinstance(pattern.replace, list) and pattern.replace and pattern.replace[0]:
                caller = pattern.replace[0]
            else:
                span = match.span(1)
                caller = restline[span[0] : span[1]]
            state.used_variables[caller] = pattern.variable_type or "shader-variable"
            if isinstance(pattern.replace, list) and len(pattern.replace) > 1 and pattern.replace[1]:
                name = pattern.replace[1]
            else:
                span = match.span(2)
                name = restline[span[0] : span[1]]
            output("expression", cg.property(caller, name))

    def process_current_line(line: str) -> None:
        """Tokenize and emit ``line`` from ``current_column`` onward."""
        while state.current_column < len(line):
            restline = line[state.current_column :]
            nxt = _match_next_pattern(restline, patterns)
            if nxt is None:
                break
            start_idx = nxt[1]
            if start_idx > 0:
                output(
                    "code",
                    line[state.current_column : state.current_column + start_idx],
                )
            process_pattern_match(line, restline, nxt)
        if state.current_column < len(line):
            output("code", line[state.current_column :])
        output("code", "\n")

    previous_line_was_empty = True

    for i in range(len(pass1)):
        line = pass1[i].line
        state.current_line = i
        state.current_column = 0

        if preserve_code_reference:
            max_line_number = len(pass1)
            line_number_width = len(str(max_line_number))
            padded_line_number = str(state.current_line + 1).rjust(line_number_width)
            source_path = pass1[i].code_reference.file_path
            source_template = state.repo.templates[source_path]
            assert isinstance(source_template, TemplatePass1)
            source_line = source_template.raw[pass1[i].code_reference.line_number - 1]
            output("raw", f"// {padded_line_number} | {source_line}\n")

        if line == "":
            if i == len(pass1) - 1:
                continue
            if previous_line_was_empty:
                continue
            previous_line_was_empty = True
        else:
            previous_line_was_empty = False

        stripped = line.lstrip()
        if stripped.startswith("#"):
            if stripped.startswith("#use "):
                uses = [p for p in stripped[5:].split(" ") if p.strip()]
                for use in uses:
                    pat = lookup_pattern(use)
                    if pat is None:
                        raise WgslTemplateGenerateError(
                            f"Unknown use: {use} at line {state.current_line + 1}",
                            "code-pattern-not-found",
                            file_path=state.file_path,
                            line_number=state.current_line + 1,
                        )
                    patterns.append(pat)
            elif stripped.startswith("#param "):
                params = [p for p in stripped[7:].split(" ") if p.strip()]
                if not params:
                    raise WgslTemplateGenerateError(
                        f"No parameters specified in #param directive at line {state.current_line + 1}",
                        "parameter-missing",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                for param in params:
                    pat = create_param_pattern(param)
                    # Allow duplicates if same; error on type mismatch.
                    pat_str = pat.pattern.pattern if hasattr(pat.pattern, "pattern") else str(pat.pattern)
                    existing = next(
                        (
                            p
                            for p in patterns
                            if p.type == "param"
                            and (p.pattern.pattern if hasattr(p.pattern, "pattern") else str(p.pattern)) == pat_str
                        ),
                        None,
                    )
                    if existing is None:
                        patterns.append(pat)
                    else:
                        existing_type = existing.param_type or "int"
                        new_type = pat.param_type or "int"
                        if existing_type != new_type:
                            raise WgslTemplateGenerateError(
                                f"Duplicate param with different type: {param} at line {state.current_line + 1}",
                                "parameter-type-mismatch",
                                file_path=state.file_path,
                                line_number=state.current_line + 1,
                            )
            elif stripped.startswith("#if "):
                if fn_call:
                    raise WgslTemplateGenerateError(
                        f"Preprocessor directive inside function call at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                condition = stripped[4:].strip()
                if not condition:
                    raise WgslTemplateGenerateError(
                        f"Empty condition in #if directive at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )

                leading_ws = len(line) - len(stripped)
                state.current_column = leading_ws + 4
                if_stack.append(
                    [
                        "if",
                        state.current_parentheses_state,
                        state.current_bracket_state,
                        None,
                        None,
                    ]
                )
                output("raw", "if (")
                pre_processor_expression = []
                cached_paren = state.current_parentheses_state
                state.current_parentheses_state = 0
                process_current_line(line)
                if state.current_parentheses_state != 0:
                    raise WgslTemplateGenerateError(
                        f"Unmatched parentheses in preprocessor expression at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                state.current_parentheses_state = cached_paren
                state.result.extend(pre_processor_expression)
                pre_processor_expression = None
                output("raw", ") {\n")
            elif stripped.startswith("#elif "):
                if not if_stack or if_stack[-1][0] not in ("if", "elif"):
                    raise WgslTemplateGenerateError(
                        f"#elif mismatch at line {state.current_line + 1}",
                        "code-generation-failed",
                        line_number=state.current_line + 1,
                    )
                if fn_call:
                    raise WgslTemplateGenerateError(
                        f"Preprocessor directive inside function call at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )

                prev_paren = if_stack[-1][3]
                if prev_paren is not None and state.current_parentheses_state != prev_paren:
                    raise WgslTemplateGenerateError(
                        f"Parentheses state mismatch in #elif directive at "
                        f"line {state.current_line + 1}, expected "
                        f"{prev_paren}, got {state.current_parentheses_state}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                prev_bracket = if_stack[-1][4]
                if prev_bracket is not None and state.current_bracket_state != prev_bracket:
                    raise WgslTemplateGenerateError(
                        f"Bracket state mismatch in #elif directive at "
                        f"line {state.current_line + 1}, expected "
                        f"{prev_bracket}, got {state.current_bracket_state}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                if_stack[-1][3] = state.current_parentheses_state
                if_stack[-1][4] = state.current_bracket_state
                state.current_parentheses_state = if_stack[-1][1]
                state.current_bracket_state = if_stack[-1][2]

                condition = stripped[6:].strip()
                if not condition:
                    raise WgslTemplateGenerateError(
                        f"Empty condition in #elif directive at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )

                leading_ws = len(line) - len(stripped)
                state.current_column = leading_ws + 6
                if_stack[-1][0] = "elif"
                output("raw", "} else if (")
                pre_processor_expression = []
                cached_paren = state.current_parentheses_state
                state.current_parentheses_state = 0
                process_current_line(line)
                if state.current_parentheses_state != 0:
                    raise WgslTemplateGenerateError(
                        f"Unmatched parentheses in preprocessor expression at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                state.current_parentheses_state = cached_paren
                state.result.extend(pre_processor_expression)
                pre_processor_expression = None
                output("raw", ") {\n")
            elif stripped.startswith("#else"):
                if not if_stack or if_stack[-1][0] not in ("if", "elif"):
                    raise WgslTemplateGenerateError(
                        f"#else mismatch at line {state.current_line + 1}",
                        "code-generation-failed",
                        line_number=state.current_line + 1,
                    )
                if fn_call:
                    raise WgslTemplateGenerateError(
                        f"Preprocessor directive inside function call at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )

                prev_paren = if_stack[-1][3]
                if prev_paren is not None and state.current_parentheses_state != prev_paren:
                    raise WgslTemplateGenerateError(
                        f"Parentheses state mismatch in #elif directive at "
                        f"line {state.current_line + 1}, expected "
                        f"{prev_paren}, got {state.current_parentheses_state}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                prev_bracket = if_stack[-1][4]
                if prev_bracket is not None and state.current_bracket_state != prev_bracket:
                    raise WgslTemplateGenerateError(
                        f"Bracket state mismatch in #elif directive at "
                        f"line {state.current_line + 1}, expected "
                        f"{prev_bracket}, got {state.current_bracket_state}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                if_stack[-1][3] = state.current_parentheses_state
                if_stack[-1][4] = state.current_bracket_state
                state.current_parentheses_state = if_stack[-1][1]
                state.current_bracket_state = if_stack[-1][2]

                if stripped[5:].strip() != "":
                    raise WgslTemplateGenerateError(
                        f"Unexpected content after #else at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                if_stack[-1][0] = "else"
                output("raw", "} else {\n")
            elif stripped.startswith("#endif"):
                if not if_stack:
                    raise WgslTemplateGenerateError(
                        f"#endif mismatch at line {state.current_line + 1}",
                        "code-generation-failed",
                        line_number=state.current_line + 1,
                    )
                if fn_call:
                    raise WgslTemplateGenerateError(
                        f"Preprocessor directive inside function call at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )

                prev_paren = if_stack[-1][3]
                if prev_paren is not None and state.current_parentheses_state != prev_paren:
                    raise WgslTemplateGenerateError(
                        f"Parentheses state mismatch in #elif directive at "
                        f"line {state.current_line + 1}, expected "
                        f"{prev_paren}, got {state.current_parentheses_state}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                prev_bracket = if_stack[-1][4]
                if prev_bracket is not None and state.current_bracket_state != prev_bracket:
                    raise WgslTemplateGenerateError(
                        f"Bracket state mismatch in #elif directive at "
                        f"line {state.current_line + 1}, expected "
                        f"{prev_bracket}, got {state.current_bracket_state}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )

                if stripped[6:].strip() != "":
                    raise WgslTemplateGenerateError(
                        f"Unexpected content after #endif at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                output("raw", "}\n")
                if_stack.pop()
            else:
                if stripped in ("#use", "#param", "#if", "#elif"):
                    raise WgslTemplateGenerateError(
                        f"Missing content after preprocessor directive at line {state.current_line + 1}",
                        "code-generation-failed",
                        file_path=state.file_path,
                        line_number=state.current_line + 1,
                    )
                raise WgslTemplateGenerateError(
                    f"Unknown preprocessor directive: {stripped} at line {state.current_line + 1}",
                    "code-generation-failed",
                    file_path=state.file_path,
                    line_number=state.current_line + 1,
                )
            previous_line_was_empty = True
        else:
            process_current_line(line)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def generate(
    file_path: str,
    repo: TemplateRepository,
    code_generator: StaticCodeGenerator,
    *,
    preserve_code_reference: bool = False,
) -> GenerateResult:
    """Run PASS2 on a single template, returning the per-template result."""

    template = repo.templates.get(file_path)
    if template is None or not isinstance(template, TemplatePass1):
        raise WgslTemplateGenerateError(
            f"Template not found for file: {file_path}",
            "generator-not-found",
            file_path=file_path,
        )

    state = _GeneratorState(
        repo=repo,
        pass1=list(template.pass1),
        code_generator=code_generator,
        file_path=file_path,
    )

    _generate_impl(state, preserve_code_reference)

    if state.preprocess_if_stack:
        kinds = ", ".join(str(entry[0]) for entry in state.preprocess_if_stack)
        raise WgslTemplateGenerateError(
            f"Unclosed preprocessor directive: {kinds}",
            "code-generation-failed",
        )
    if state.current_function_call:
        names = ", ".join(c.name for c in state.current_function_call)
        raise WgslTemplateGenerateError(
            f"Unclosed function call: {names}",
            "code-generation-failed",
        )
    if state.current_parentheses_state != 0:
        raise WgslTemplateGenerateError(
            "Unmatched parentheses at the end of processing",
            "code-generation-failed",
        )

    if state.main_function == "started":
        raise WgslTemplateGenerateError(
            "Main function context started but not ended at the end of processing",
            "code-generation-failed",
        )
    if state.current_bracket_state != 0:
        raise WgslTemplateGenerateError(
            "Unmatched brackets at the end of processing",
            "code-generation-failed",
        )

    state.result = _merge_adjacent_segments(state.result)

    sorted_params = {k: state.used_params[k] for k in sorted(state.used_params)}
    sorted_variables = {k: state.used_variables[k] for k in sorted(state.used_variables)}

    return GenerateResult(
        code=code_generator.emit(state.result),
        params=sorted_params,
        variables=sorted_variables,
        has_main_function=(state.main_function == "ended"),
    )


def generate_directory(
    repo: TemplateRepository,
    code_generator: StaticCodeGenerator,
    *,
    preserve_code_reference: bool = False,
) -> TemplateRepository:
    """Run PASS2 on every template in the repository."""

    out_templates: dict[str, object] = {}
    for file_path in repo.templates:
        result = generate(
            file_path,
            repo,
            code_generator,
            preserve_code_reference=preserve_code_reference,
        )
        template = repo.templates[file_path]
        assert isinstance(template, TemplatePass1)
        out_templates[file_path] = TemplatePass2(
            file_path=template.file_path,
            generate_result=result,
        )

    return TemplateRepository(
        base_path=repo.base_path,
        templates=out_templates,
    )
