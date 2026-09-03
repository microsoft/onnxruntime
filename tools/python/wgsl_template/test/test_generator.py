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
        self.assertIn("__var_output.Rank()", out)

    def test_method_call(self) -> None:
        out = _gen("#use .offsetToIndices\nlet i = output.offsetToIndices(j);\n")
        self.assertIn("__var_output.OffsetToIndices", out)


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
