# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Static C++ code generator.

The single :class:`StaticCodeGenerator` class implements both
``static-cpp`` (uses a string table) and ``static-cpp-literal``
(inlines string literals); the only difference is the constructor
flag.
"""

from __future__ import annotations

import builtins
import hashlib
import json
from dataclasses import dataclass

from ..types import TemplatePass2, TemplateRepository

# ----------------------------------------------------------------------
# Code segment data model
# ----------------------------------------------------------------------


@dataclass
class CodeSegment:
    type: str  # "raw" | "code" | "expression"
    content: str


@dataclass
class CodeSegmentArg:
    type: str  # "expression" | "string" | "auto"
    code: list[CodeSegment]


# ----------------------------------------------------------------------
# String literal renderer (JSON-style escaping)
# ----------------------------------------------------------------------


def _js_stringify(s: str) -> str:
    """Render ``s`` as a double-quoted, JSON-escaped C++ string literal.

    Non-ASCII characters are escaped via ``\\uXXXX`` so the output is
    pure ASCII and stable across hosts.
    """
    return json.dumps(s, ensure_ascii=True)


# ----------------------------------------------------------------------
# Static code generator
# ----------------------------------------------------------------------


class StaticCodeGenerator:
    """Emits C++ from PASS2 :class:`CodeSegment` lists.

    With ``use_string_table=True`` (the ``static-cpp`` Release mode),
    every WGSL string literal is replaced with a short ``__str_N``
    identifier and dedup'd across the whole build. With
    ``use_string_table=False`` (Debug mode), strings appear inline as
    ``"..."`` literals.
    """

    def __init__(self, use_string_table: bool = True) -> None:
        self.use_string_table = use_string_table
        # Maps unique string content to its assigned __str_N id.
        # Insertion order is the assignment order.
        self._string_table: dict[str, int] | None = {} if use_string_table else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_string(self, s: str) -> str:
        if self._string_table is not None:
            existing = self._string_table.get(s)
            if existing is None:
                existing = len(self._string_table)
                self._string_table[s] = existing
            return f"__str_{existing}"
        return _js_stringify(s)

    def _render_arg(self, arg: CodeSegmentArg) -> str:
        if not arg.code:
            return ""

        if len(arg.code) == 1:
            seg = arg.code[0]
            if arg.type == "string" and seg.type == "expression":
                return f"wgsl_detail::pass_as_string({seg.content})"
            if arg.type != "expression" and seg.type == "code":
                return self._render_string(seg.content)
            return seg.content

        # Multi-segment argument.
        if arg.type != "expression":
            rendered = []
            for seg in arg.code:
                if seg.type == "code":
                    rendered.append(self._render_string(seg.content))
                else:
                    rendered.append(seg.content)
            return f"absl::StrCat({', '.join(rendered)})"
        return "".join(seg.content for seg in arg.code)

    # ------------------------------------------------------------------
    # CodeGenerator interface (called from the PASS2 generator)
    # ------------------------------------------------------------------

    def emit(self, code: list[CodeSegment]) -> str:
        out = []
        for seg in code:
            if seg.type == "raw":
                out.append(seg.content)
            elif seg.type == "code":
                out.append(f"ss << {self._render_string(seg.content)};\n")
            elif seg.type == "expression":
                out.append(f"ss << {seg.content};\n")
            else:  # pragma: no cover  defensive
                raise ValueError(f"Unknown segment type: {seg.type!r}")
        return "".join(out)

    def param(self, name: str) -> str:
        return f"__param_{name}"

    def variable(self, name: str) -> str:
        return f"__var_{name}"

    def property(self, obj: str, property_name: str) -> str:
        return f"__var_{obj}.{property_name}"

    def function(self, name: str, args: list[CodeSegmentArg]) -> str:
        rendered = ", ".join(self._render_arg(a) for a in args)
        return f"{name}({rendered})"

    def method(self, obj: str, method_name: str, args: list[CodeSegmentArg]) -> str:
        rendered = ", ".join(self._render_arg(a) for a in args)
        return f"__var_{obj}.{method_name}({rendered})"

    # ------------------------------------------------------------------
    # Read-only access to the string table (used by build() to emit
    # string_table.h).
    # ------------------------------------------------------------------

    @builtins.property
    def string_table(self) -> dict[str, int] | None:
        return self._string_table

    # ------------------------------------------------------------------
    # build() — produce the per-template files plus index.h /
    # index_impl.h / string_table.h.
    # ------------------------------------------------------------------

    def build(
        self,
        repo: TemplateRepository,
        *,
        template_ext: str,
        include_path_prefix: str = "",
    ) -> TemplateRepository:
        result_files: dict[str, str] = {}
        impl_hashes: dict[str, str] = {}

        # STEP 1 — per-template implementation files.
        for name, template in repo.templates.items():
            assert isinstance(template, TemplatePass2)
            if not name.endswith(template_ext):
                raise ValueError(f'Template name "{name}" does not end with the expected extension "{template_ext}"')
            base = name[: -len(template_ext)]
            content = self._build_template_implementation(name, template)
            result_files[f"generated/{base}.h"] = content
            impl_hashes[name] = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # STEP 2 — string_table.h if enabled.
        if self._string_table is not None:
            content = self._build_generate_string_table()
            result_files["string_table.h"] = content
            impl_hashes["string_table.h"] = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # STEP 3 — index_impl.h.
        result_files["index_impl.h"] = self._build_generate_index_impl(
            repo, impl_hashes, include_path_prefix, template_ext
        )

        # STEP 4 — index.h.
        result_files["index.h"] = self._build_generate_index(repo)

        return TemplateRepository(base_path=repo.base_path, templates=dict(result_files))

    # ---- internal builders -------------------------------------------

    def _build_generate_index(self, repo: TemplateRepository) -> str:
        out: list[str] = []
        out.append("// This file is auto-generated by wgsl-gen. Do not edit manually.")
        out.append("")
        out.append("#ifndef INCLUDED_BY_WGSL_GEN_HEADER")
        out.append('#error "This file is expected to be included by wgsl-gen header. Do not include it directly."')
        out.append("#endif")
        out.append("")
        for name, template in repo.templates.items():
            assert isinstance(template, TemplatePass2)
            out.append("//")
            out.append(f"// Template: {name}")
            out.append("//")
            out.append("")
            quoted = _js_stringify(name)
            out.append("template <>")
            out.append(f"struct TemplateParameter<{quoted}> {{")
            out.append("  using type = struct {")
            for param_name in template.generate_result.params:
                out.append(f"    int param_{param_name};")
            for var_name in template.generate_result.variables:
                out.append(f"    const ShaderVariableHelper* var_{var_name};")
            out.append("  };")
            out.append("};")
            out.append("")
            out.append("template <>")
            out.append(
                f"Status ApplyTemplate<{quoted}>(ShaderHelper& shader_helper, "
                f"TemplateParameter<{quoted}>::type params);"
            )
            out.append("")
        return "\n".join(out)

    def _build_generate_string_table(self) -> str:
        if self._string_table is None:
            raise RuntimeError("String table is not enabled")

        out: list[str] = []
        out.append("// This file is auto-generated by wgsl-gen. Do not edit manually.")
        out.append("")
        out.append("#pragma once")
        out.append("#ifndef INCLUDED_BY_WGSL_GEN_IMPL")
        out.append('#error "This file is expected to be included by wgsl-gen impl. Do not include it directly."')
        out.append("#endif")
        out.append("")
        out.append("// String table constants")

        # Sort by id to ensure consistent output.
        sorted_entries = sorted(self._string_table.items(), key=lambda kv: kv[1])
        for s, sid in sorted_entries:
            out.append(f"constexpr const char* __str_{sid} = {_js_stringify(s)};")

        out.append("")
        return "\n".join(out)

    def _build_generate_index_impl(
        self,
        repo: TemplateRepository,
        impl_hashes: dict[str, str],
        include_path_prefix: str,
        template_ext: str,
    ) -> str:
        out: list[str] = []
        out.append(
            "// This file is auto-generated by wgsl-gen. Do not edit manually.\n"
            "\n"
            "#pragma once\n"
            "#ifndef INCLUDED_BY_WGSL_GEN_IMPL\n"
            '#error "This file is expected to be included by wgsl-gen impl. '
            'Do not include it directly."\n'
            "#endif\n"
            "\n"
            "// Helper functions or macros\n"
            "\n"
            '#pragma push_macro("MainFunctionStart")\n'
            "#undef MainFunctionStart\n"
            "#define MainFunctionStart() { [[maybe_unused]] auto& ss = "
            "shader_helper.MainFunctionBody();\n"
            '#pragma push_macro("MainFunctionEnd")\n'
            "#undef MainFunctionEnd\n"
            "#define MainFunctionEnd() }\n"
            "\n"
            "// Helper templates\n"
            "\n"
            "namespace wgsl_detail {\n"
            "template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>\n"
            "std::string pass_as_string(T&& v) {\n"
            "  return std::to_string(std::forward<T>(v));\n"
            "}\n"
            "template <typename...>\n"
            "std::string_view pass_as_string(std::string_view sv) {\n"
            "  return sv;\n"
            "}\n"
            "template <typename T>\n"
            "std::string pass_as_string(T&& v) {\n"
            "  return std::forward<T>(v);\n"
            "}\n"
            "}  // namespace wgsl_detail\n"
        )

        if self._string_table is not None:
            sthash = impl_hashes.get("string_table.h", "")
            out.append(f'#include "{include_path_prefix}string_table.h"  // {sthash}')
        out.append("")
        out.append("// Include template implementations")
        out.append("")

        for name in repo.templates:
            if not name.endswith(template_ext):
                raise ValueError(f'Template name "{name}" does not end with the expected extension "{template_ext}"')
            base = name[: -len(template_ext)]
            h = impl_hashes.get(name, "")
            out.append(f'#include "{include_path_prefix}generated/{base}.h"  // {h}')

        out.append("")
        out.append('#pragma pop_macro("MainFunctionStart")')
        out.append('#pragma pop_macro("MainFunctionEnd")')

        return "\n".join(out)

    def _build_template_implementation(self, file_path: str, template: TemplatePass2) -> str:
        out: list[str] = []
        out.append("// This file is auto-generated by wgsl-gen. Do not edit manually.")
        out.append("")
        out.append("#pragma once")
        out.append("")
        out.append("// Template implementation")
        out.append(f"// Source: {file_path}")
        out.append("")

        gr = template.generate_result
        params_unused = len(gr.params) == 0 and len(gr.variables) == 0
        quoted = _js_stringify(file_path)

        out.append("template <>")
        params_arg = "" if params_unused else "params"
        out.append(
            f"Status ApplyTemplate<{quoted}>(ShaderHelper& shader_helper, "
            f"TemplateParameter<{quoted}>::type {params_arg}) {{"
        )
        out.append("  [[maybe_unused]] auto& ss = shader_helper.AdditionalImplementation();")
        out.append("")

        if gr.params:
            out.append("  // Extract parameters")
            for param_name in gr.params:
                out.append(f"  auto& {self.param(param_name)} = params.param_{param_name};")
            out.append("")

        if gr.variables:
            out.append("  // Extract variables")
            for var_name in gr.variables:
                out.append(f"  auto& {self.variable(var_name)} = *params.var_{var_name};")
            out.append("")

        out.append(gr.code)

        out.append("")
        out.append("  return Status::OK();")
        out.append("}")

        return "\n".join(out)
