# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Unit tests for the PASS2 generator.

Combines focused unit tests with a fixture-driven runner that pulls
``generator-*`` cases from the ``testcases/`` fixtures next to this
file. We only fixture-test cases where:

* The fixture directory contains a ``*.static-cpp-literal.gen``
  golden, OR
* The case expects an error (``expectsError`` in test-config.json)
  with at least one of its generators being one we support.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from wgsl_template.code_generator import resolve_code_generator  # noqa: E402
from wgsl_template.errors import WgslTemplateGenerateError  # noqa: E402
from wgsl_template.generator import generate  # noqa: E402
from wgsl_template.loader import load_from_directory  # noqa: E402
from wgsl_template.parser import parse  # noqa: E402

# Fixtures live next to this file. Resolve relative to __file__ so the
# suite runs on any platform and from any working directory.
_TESTCASES_DIR = _THIS_DIR / "testcases"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))


def _gen(text: str, *, generator: str = "static-cpp-literal", preserve: bool = False) -> str:
    """Run a single template through PASS0+PASS1+PASS2, returning the
    emitted C++ string."""

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / "test.wgsl.template", text)
        repo = load_from_directory(root)
        parsed = parse(repo)
        cg = resolve_code_generator(generator)
        return generate(
            "test.wgsl.template",
            parsed,
            cg,
            preserve_code_reference=preserve,
        ).code


# ----------------------------------------------------------------------
# Focused unit tests
# ----------------------------------------------------------------------


class GeneratorBasicTest(unittest.TestCase):
    def test_simple_passthrough(self) -> None:
        # No directives, no patterns — every line becomes a code segment.
        out = _gen("a\nb\n")
        self.assertEqual(out, 'ss << "a\\nb\\n";\n')

    def test_param_declaration(self) -> None:
        out = _gen("#param X\nlet a = X;\n")
        # X expands to __param_X expression.
        self.assertIn("__param_X", out)

    def test_main_block(self) -> None:
        out = _gen("$MAIN {\n  a;\n}\n")
        self.assertIn("MainFunctionStart();", out)
        self.assertIn("MainFunctionEnd();", out)

    def test_source_references_have_no_trailing_whitespace(self) -> None:
        for generator in ("static-cpp", "static-cpp-literal"):
            with self.subTest(generator=generator):
                out = _gen("\n \t\nlet value = 1; \t\n", generator=generator, preserve=True)
                references = [line for line in out.splitlines() if line.startswith("// ")]
                self.assertTrue(references)
                self.assertTrue(any(line.endswith("|") for line in references))
                self.assertTrue(any(line.endswith("let value = 1;") for line in references))
                self.assertTrue(all(line == line.rstrip() for line in out.splitlines()))


class GeneratorLoopTest(unittest.TestCase):
    def test_nested_loops_and_helpers(self) -> None:
        text = """#use loop .rank .indicesGet
#param N
$MAIN {
loop(i, 0, N) {
loop(j, i, output.rank, 2) {
  let v = output.indicesGet(coords, j);
  values[i + j] = v;
}
}
}
"""
        for mode in ("static-cpp", "static-cpp-literal"):
            with self.subTest(mode=mode):
                out = _gen(text, generator=mode)
                self.assertIn("static_cast<int64_t>(__param_N)", out)
                self.assertIn("static_cast<int64_t>(__loop_0_index)", out)
                self.assertIn("static_cast<int64_t>(__var_output->Rank())", out)
                self.assertRegex(out, r'IndicesGet\((?:"coords"|__str_\d+), __loop_1_index\)')
                self.assertEqual(out.count("for (int64_t"), 2)

    def test_shader_loops_passthrough_and_index_scope(self) -> None:
        out = _gen(
            "#use loop\nloop {\nbreak;\n}\nfor (var k = 0; k < 2; k++) {}\nloop(k, 0, 2) {\na[k] = k;\n}\nlet k = 9;\n"
        )
        self.assertIn("loop {", out)
        self.assertIn("for (var k", out)
        self.assertIn("let k = 9;", out)
        self.assertIn("ss << __loop_0_index;", out)

    def test_conditions_inside_and_outside_loop(self) -> None:
        out = _gen("#use loop\n#param N\n#if N\nloop(k, 0, N) {\n#if k\na[k];\n#else\nb[k];\n#endif\n}\n#endif\n")
        self.assertIn("if (__loop_0_index)", out)

    def test_invalid_loops(self) -> None:
        cases = (
            "loop(k, 0) {\n}\n",
            "loop(k, 0, uniforms.K) {\n}\n",
            "loop(k, 0, 1 2) {\n}\n",
            "loop(k, 0, 2147483648) {\n}\n",
            "#use .rank\nloop(output, 0, output.rank) {\n}\n",
            "loop(k, 0, 4) {\nx(\n}\n);\n",
            "loop(k, 0, 4) {\n#param k\n}\n",
            "loop(k, 0, 4) { a[k]; }\n",
            "loop(k, 0, 4) {\n",
            "loop(k, 0, 4) {\nbreak;\n}\n",
            "loop(k, 0, 4) {\nlet k = 1;\n}\n",
            "loop(k, 0, 4) {\nvar<function> k : i32 = 0;\n}\n",
            "loop(k, 0, 4) {\nvar < function > k : i32 = 0;\n}\n",
            "#use .rank\nloop(output, 0, 1) {\nlet n = output.rank;\n}\n",
            "#use .offsetToIndices\nloop(output, 0, 1) {\noutput.offsetToIndices(0);\n}\n",
            "loop(k, 0, 4) {\nloop(k, 0, 4) {\n}\n}\n",
            "#param k\nloop(k, 0, 4) {\n}\n",
            "#param N\n#if N\nloop(k, 0, 4) {\n#endif\n}\n",
            "#param N\nloop(k, 0, 4) {\n#if N\n}\n#endif\n",
        )
        for text in cases:
            with self.subTest(text=text), self.assertRaises(WgslTemplateGenerateError):
                _gen("#use loop\n" + text)

    def test_member_selectors_and_signed_bounds(self) -> None:
        out = _gen("#use loop\nloop(x, -2147483648, -2147483647) {\na[x] = vector.x + vector . x + vector.\n x;\n}\n")
        self.assertIn("(-2147483648LL)", out)
        self.assertIn("vector.x + vector . x + vector.\\n x", out)
        self.assertEqual(out.count("ss << __loop_0_index;"), 1)

    def test_split_declarations_reject_active_loop_indices(self) -> None:
        for declaration in (
            "let\nk",
            "var\nk",
            "const\nk",
            "var\n<function>\nk",
            "var<\nfunction\n>\nk",
            "let\n#if FLAG\nvalue\n#else\nk\n#endif",
            "#if FLAG\nlet\n#else\nlet\n#endif\nk",
        ):
            for mode in ("static-cpp", "static-cpp-literal"):
                with (
                    self.subTest(declaration=declaration, mode=mode),
                    self.assertRaisesRegex(WgslTemplateGenerateError, "Cannot redeclare a template loop index"),
                ):
                    _gen(f"#use loop\n#param FLAG\nloop(k, 0, 2) {{\n{declaration} = 1;\n}}\n", generator=mode)

    def test_split_declaration_with_distinct_name_preserves_index_expression(self) -> None:
        for mode in ("static-cpp", "static-cpp-literal"):
            with self.subTest(mode=mode):
                out = _gen("#use loop\nloop(k, 0, 2) {\nlet\nvalue = k;\n}\n", generator=mode)
                self.assertEqual(out.count("ss << __loop_0_index;"), 1)

    def test_member_selectors_across_conditional_branches(self) -> None:
        text = """#use loop
#param FLAG
loop(x, 0, 1) {
  {
    let v = vec2<f32>(1.0);
    let a = v.
#if FLAG
      x
#else
      y
#endif
    ;
    let b = v.
#if FLAG
      y
#elif x == 0
      x
#else
      x
#endif
    ;
#if FLAG
    let c = v.
#else
    let c = v.
#endif
    x;
    values[x] = x;
  }
}
"""
        for mode in ("static-cpp", "static-cpp-literal"):
            with self.subTest(mode=mode):
                out = _gen(text, generator=mode)
                # Only the array index and assignment expand; the condition stays a host expression.
                self.assertEqual(out.count("ss << __loop_0_index;"), 2)
                self.assertIn("else if (__loop_0_index == 0)", out)

    def test_member_selector_context_restored_for_nested_branches(self) -> None:
        text = """#use loop
#param FLAG
loop(x, 0, 1) {
let a = v.
#if FLAG
#if FLAG
x
#else
x
#endif
#else
x
#endif
;
#if FLAG
let b = v.
x;
#else
let b = x;
#endif
}
"""
        for mode in ("static-cpp", "static-cpp-literal"):
            with self.subTest(mode=mode):
                self.assertEqual(_gen(text, generator=mode).count("ss << __loop_0_index;"), 1)

    def test_inconsistent_member_selector_context_reports_error(self) -> None:
        for mode in ("static-cpp", "static-cpp-literal"):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(WgslTemplateGenerateError, "inconsistent member-selector context"),
            ):
                _gen("#use loop\n#param FLAG\nloop(x, 0, 1) {\n#if FLAG\nv.\n#endif\nx;\n}\n", generator=mode)

    @unittest.skipUnless(
        shutil.which("clang++") or shutil.which("g++") or shutil.which("cl"), "C++ compiler unavailable"
    )
    def test_execute_generated_cpp(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++") or shutil.which("cl")
        text = """#use loop .offsetToIndices .indicesToOffset .getByOffset .setByOffset
#param N STEP
loop(i, +(0), N - (1 - 1), STEP) {
loop(j, i, N) {
a[i] = j;
}
}
loop(x, -2, 0) {
{
let v = -x;
let w = 1-x;
let member = vector.x + vector . x + vector.
 x;
let conditional_member = vector.
#if N
 x
#else
 y
#endif
;
output.offsetToIndices(x);
output.indicesToOffset(x);
let loaded = output.getByOffset(x);
output.setByOffset(x, x);
output.offsetToIndices(1-x);
}
}
loop(k, -2147483648, -2147483647) {
min_value = k;
}
flat_begin;
loop(flat, 0, 2) {
  accumulator += flat;
}
flat_end;
scoped_begin;
loop(block, 0, 2) {
  {
    let temporary = block;
    accumulator += temporary;
  }
}
scoped_end;
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "test.wgsl.template", text)
            parsed = parse(load_from_directory(root))
            # Compile the production converter and wrappers; only their non-inline backends are stubbed.
            header = (_THIS_DIR.parents[3] / "onnxruntime/core/providers/webgpu/shader_variable.h").read_text(
                encoding="utf-8"
            )
            converter = header[header.index("namespace detail {") : header.index("}  // namespace detail")]
            converter += "}  // namespace detail\n"
            wrappers = []
            for method in ("GetByOffset", "SetByOffset"):
                match = re.search(
                    rf"template <[^\n]+>\ninline std::string ShaderVariableHelper::{method}\([^{{]+\{{\n.*?\n\}}",
                    header,
                    re.DOTALL,
                )
                self.assertIsNotNone(match)
                wrappers.append(match.group(0))
            variable_helper = (
                """
class ShaderVariableHelper {
 public:
  template <typename TOffset>
  std::string GetByOffset(TOffset&& offset, bool use_storage_type = false) const;
  template <typename TOffset, typename TValue>
  std::string SetByOffset(TOffset&& offset, TValue&& value, bool use_storage_type = false) const;
 private:
  std::string GetByOffsetImpl(std::string_view offset, bool) const {
    return "load(" + std::string(offset) + ")";
  }
  std::string SetByOffsetImpl(std::string_view offset, std::string_view value, bool) const {
    return "store(" + std::string(offset) + "," + std::string(value) + ");";
  }
};
"""
                + converter
                + "\n".join(wrappers)
            )
            for mode in ("static-cpp", "static-cpp-literal"):
                cg = resolve_code_generator(mode)
                code = generate("test.wgsl.template", parsed, cg).code
                helper_code = cg._build_generate_index_impl(parsed, {}, "", ".wgsl.template")
                helper_code = helper_code[helper_code.index("namespace wgsl_detail {") :]
                helper_code = helper_code[
                    : helper_code.index("}  // namespace wgsl_detail") + len("}  // namespace wgsl_detail")
                ]
                strings = "\n".join(
                    f"const char* __str_{index} = {json.dumps(value)};"
                    for value, index in (cg.string_table or {}).items()
                )
                source = (
                    """#include <cstdint>
#include <sstream>
#include <iostream>
#include <type_traits>
#include <utility>
#include <string_view>
#define ORT_RETURN_IF_NOT(cond, message) if (!(cond)) return "invalid";
namespace absl {
template <typename... T> std::string StrCat(T&&... values) {
  std::ostringstream ss;
  (ss << ... << std::forward<T>(values));
  return ss.str();
}
}
"""
                    + variable_helper
                    + """
struct Variable : ShaderVariableHelper {
  std::string OffsetToIndices(std::string value) { return "offset(" + value + ");"; }
  std::string IndicesToOffset(std::string value) { return "indices(" + value + ");"; }
};
"""
                    + helper_code
                    + """
std::string run(int __param_N, int __param_STEP) {
std::ostringstream ss;
Variable variable;
auto* __var_output = &variable;
"""
                    + strings
                    + "\n"
                    + code
                    + "\nreturn ss.str();\n}\n"
                    + r"""
int main() {
  const int64_t integral = -2;
  int64_t mutable_integral = 2;
  ShaderVariableHelper real_helper;
  if (real_helper.GetByOffset(integral) != "load(-2)" ||
      real_helper.GetByOffset(mutable_integral) != "load(2)" ||
      real_helper.GetByOffset(int64_t{3}) != "load(3)" ||
      real_helper.GetByOffset(std::string_view{"idx"}) != "load(idx)" ||
      real_helper.SetByOffset(integral, mutable_integral) != "store(-2,2);") return 5;
  if (wgsl_detail::pass_as_string(integral) != "-2" ||
      wgsl_detail::pass_as_string(int64_t{2}) != "2" ||
      wgsl_detail::pass_as_string(std::string_view{"text"}) != "text") return 3;
  if (run(0, 1).find("a[") != std::string::npos ||
      run(-1, 1).find("a[") != std::string::npos) return 1;
  if (run(0, 1).find("let conditional_member = vector.\n y\n;") == std::string::npos ||
      run(3, 1).find("let conditional_member = vector.\n x\n;") == std::string::npos) return 4;
  if (run(3, 0) != "invalid" || run(3, -1) != "invalid") return 2;
  std::cout << run(3, 2);
}
"""
                )
                cpp = root / "loop.cc"
                exe = root / "loop.exe"
                _write(cpp, source)
                command = (
                    [compiler, "/nologo", "/EHsc", "/std:c++20", str(cpp), f"/Fe:{exe}"]
                    if Path(compiler).stem.lower() == "cl"
                    else [compiler, "-std=c++20", str(cpp), "-o", str(exe)]
                )
                subprocess.run(command, cwd=root, check=True, capture_output=True)
                result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
                self.assertEqual(
                    re.findall(r"a\[\((\d+)\)\] = \((\d+)\);", result.stdout),
                    [("0", "0"), ("0", "1"), ("0", "2"), ("2", "2")],
                )
                self.assertIn("let v = -(-2);", result.stdout)
                self.assertIn("let w = 1-(-2);", result.stdout)
                self.assertIn("vector.x + vector . x + vector.\n x", result.stdout)
                self.assertIn("offset(-2);", result.stdout)
                self.assertIn("indices(-2);", result.stdout)
                self.assertIn("let loaded = load(-2);", result.stdout)
                self.assertIn("store(-2,-2);", result.stdout)
                self.assertIn("offset(1-(-2));", result.stdout)
                self.assertIn("min_value = (-2147483648);", result.stdout)
                flat = result.stdout.split("flat_begin;", 1)[1].split("flat_end;", 1)[0]
                self.assertEqual(re.sub(r"\s+", "", flat), "accumulator+=(0);accumulator+=(1);")
                scoped = result.stdout.split("scoped_begin;", 1)[1].split("scoped_end;", 1)[0]
                self.assertEqual(
                    re.sub(r"\s+", "", scoped),
                    "{lettemporary=(0);accumulator+=temporary;}{lettemporary=(1);accumulator+=temporary;}",
                )


class GeneratorIfTest(unittest.TestCase):
    def test_if_else(self) -> None:
        out = _gen("#param X\n#if X\nyes\n#else\nno\n#endif\n")
        self.assertIn("if (__param_X) {", out)
        self.assertIn("} else {", out)
        self.assertIn("}\n", out)

    def test_if_elif(self) -> None:
        out = _gen("#param X\n#if X\na\n#elif X\nb\n#endif\n")
        self.assertIn("if (__param_X) {", out)
        self.assertIn("} else if (__param_X) {", out)

    def test_orphan_endif(self) -> None:
        with self.assertRaises(WgslTemplateGenerateError):
            _gen("#endif\n")

    def test_orphan_else(self) -> None:
        with self.assertRaises(WgslTemplateGenerateError):
            _gen("#else\n")

    def test_unclosed_if(self) -> None:
        with self.assertRaises(WgslTemplateGenerateError):
            _gen("#param X\n#if X\nbody\n")


class GeneratorPropertyTest(unittest.TestCase):
    def test_rank_property(self) -> None:
        out = _gen("#use .rank\nlet n = output.rank;\n")
        self.assertIn("__var_output->Rank()", out)

    def test_method_call(self) -> None:
        out = _gen("#use .offsetToIndices\nlet i = output.offsetToIndices(j);\n")
        self.assertIn("__var_output->OffsetToIndices", out)


class GeneratorFunctionTest(unittest.TestCase):
    def test_get_element_at(self) -> None:
        out = _gen("#use getElementAt\nlet x = getElementAt(a, b, c);\n")
        self.assertIn("GetElementAt", out)


class GeneratorParamErrorsTest(unittest.TestCase):
    def test_invalid_param_name_with_dot(self) -> None:
        with self.assertRaises(ValueError):
            _gen("#param a.b\n")

    def test_invalid_param_starts_with_number(self) -> None:
        with self.assertRaises(ValueError):
            _gen("#param 9bad\n")


# ----------------------------------------------------------------------
# Fixture-driven tests against the generator-* testcases
# ----------------------------------------------------------------------


_SUPPORTED_GENERATORS = {"static-cpp", "static-cpp-literal"}

# No fixtures are currently skipped.
_FIXTURE_SKIPS: set[str] = set()


def _normalize_lines(text: str) -> list:
    """Split on newlines, drop blank lines, trim whitespace per line."""
    return [line.strip() for line in re.split(r"\r?\n", text) if line.strip() != ""]


def _build_fixture_suite() -> unittest.TestSuite:
    suite = unittest.TestSuite()
    if not _TESTCASES_DIR.is_dir():
        return suite

    for entry in sorted(os.listdir(_TESTCASES_DIR)):
        case_dir = _TESTCASES_DIR / entry
        if not case_dir.is_dir():
            continue
        if not entry.startswith("generator-"):
            continue
        if entry in _FIXTURE_SKIPS:
            continue
        config_path = case_dir / "test-config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if config.get("disabled"):
            continue
        if config.get("type") != "generator":
            continue

        entries = config.get("entries", {})
        applicable_entries = []
        for template_path, entry_cfg in entries.items():
            generators_cfg = entry_cfg.get("generators", {}) or {}
            for gen_name, gen_cfg in generators_cfg.items():
                if gen_name not in _SUPPORTED_GENERATORS:
                    continue
                # Only require .gen golden if no expectsError.
                expects_error = gen_cfg.get("expectsError") or config.get("expectsError")
                golden_path = case_dir / f"{template_path}.{gen_name}.gen"
                if not expects_error and not golden_path.is_file():
                    continue
                applicable_entries.append(
                    {
                        "template_path": template_path,
                        "generator": gen_name,
                        "expects_error": expects_error,
                        "expected_params": gen_cfg.get("expectedParams"),
                        "expected_variables": gen_cfg.get("expectedVariables"),
                        "golden_path": golden_path,
                    }
                )

        if not applicable_entries:
            continue

        case_class = _make_generator_case(case_dir, config, applicable_entries)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case_class))

    return suite


def _make_generator_case(case_dir: Path, config: dict, applicable_entries: list) -> type:
    case_name = case_dir.name
    preserve_code_reference = bool(config.get("preserveCodeReference"))

    class _Case(unittest.TestCase):
        def __init__(self, methodName: str = "runTest") -> None:
            super().__init__(methodName)
            self._dir = case_dir
            self._entries = applicable_entries
            self._preserve = preserve_code_reference

        def __str__(self) -> str:
            return f"fixture[{case_name}]"

        def runTest(self) -> None:  # noqa: N802
            repo = load_from_directory(self._dir)
            parsed = parse(repo)
            for entry in self._entries:
                self._run_entry(parsed, entry)

        def _run_entry(self, parsed, entry: dict) -> None:
            cg = resolve_code_generator(entry["generator"])
            try:
                result = generate(
                    entry["template_path"],
                    parsed,
                    cg,
                    preserve_code_reference=self._preserve,
                )
            except Exception as e:
                if entry["expects_error"]:
                    if isinstance(entry["expects_error"], str):
                        self.assertIn(entry["expects_error"], str(e))
                    return
                raise

            if entry["expects_error"]:
                self.fail(
                    f"{entry['template_path']} ({entry['generator']}): "
                    f"expected error containing "
                    f"{entry['expects_error']!r} but generation succeeded"
                )

            golden_text = entry["golden_path"].read_text(encoding="utf-8")
            actual_lines = _normalize_lines(result.code)
            expected_lines = _normalize_lines(golden_text)
            self.assertEqual(
                len(actual_lines),
                len(expected_lines),
                f"{entry['template_path']} ({entry['generator']}): "
                f"line count mismatch; expected {len(expected_lines)} "
                f"got {len(actual_lines)}\n"
                f"--- expected ---\n" + "\n".join(expected_lines) + "\n"
                "--- actual ---\n" + "\n".join(actual_lines),
            )
            for i, (a, b) in enumerate(zip(actual_lines, expected_lines, strict=True)):
                self.assertEqual(
                    a,
                    b,
                    f"{entry['template_path']} ({entry['generator']}): "
                    f"line {i + 1} mismatch\n"
                    f"  expected: {b!r}\n"
                    f"  actual:   {a!r}",
                )

            if entry["expected_params"] is not None:
                self.assertEqual(
                    sorted(result.params.keys()),
                    sorted(entry["expected_params"]),
                )
            if entry["expected_variables"] is not None:
                self.assertEqual(
                    sorted(result.variables.keys()),
                    sorted(entry["expected_variables"]),
                )

    _Case.__name__ = f"Fixture_{case_name.replace('-', '_')}"
    _Case.__qualname__ = _Case.__name__
    return _Case


def load_tests(loader, standard_tests, pattern):
    standard_tests.addTests(_build_fixture_suite())
    return standard_tests


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
