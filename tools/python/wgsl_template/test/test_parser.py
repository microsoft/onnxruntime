# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Unit tests for the PASS1 parser.

Mixes inline temp-dir tests for tightly scoped behaviors with a
fixture-driven runner that pulls ``parser-*`` and ``loader-*`` cases
from the ``testcases/`` fixtures next to this file.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from wgsl_template.errors import WgslTemplateParseError  # noqa: E402
from wgsl_template.loader import load_from_directory  # noqa: E402
from wgsl_template.parser import parse  # noqa: E402
from wgsl_template.types import TemplatePass1  # noqa: E402

# Fixtures live next to this file. Resolve relative to __file__ so the
# suite runs on any platform and from any working directory.
_TESTCASES_DIR = _THIS_DIR / "testcases"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))


def _parse_text(text: str) -> list[str]:
    """Helper: feed a single template through PASS0+PASS1, return the
    resulting line strings."""

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / "test.wgsl.template", text)
        repo = load_from_directory(root)
        parsed = parse(repo)
        tpl = parsed.templates["test.wgsl.template"]
        assert isinstance(tpl, TemplatePass1)
        return [line.line for line in tpl.pass1]


# ----------------------------------------------------------------------
# Comment stripping (focused unit tests)
# ----------------------------------------------------------------------


class ParserCommentsTest(unittest.TestCase):
    def test_strips_single_line_comments(self) -> None:
        self.assertEqual(
            _parse_text("a // comment\nb\n"),
            ["a", "b", ""],
        )

    def test_strips_multiline_comment_inline(self) -> None:
        self.assertEqual(
            _parse_text("a /* x */ b\n"),
            ["a  b", ""],
        )

    def test_strips_multiline_comment_spanning_lines(self) -> None:
        # Leading whitespace on the post-`*/` portion is preserved
        # (only trailing whitespace gets trimmed).
        self.assertEqual(
            _parse_text("a /* x\ny\nz */ b\n"),
            ["a", "", " b", ""],
        )

    def test_preserves_line_count(self) -> None:
        # Three input lines should still be three output lines.
        self.assertEqual(
            _parse_text("// a\n// b\n// c\n"),
            ["", "", "", ""],
        )

    def test_unterminated_multiline_raises(self) -> None:
        with self.assertRaises(WgslTemplateParseError):
            _parse_text("a /* never ends\n")


# ----------------------------------------------------------------------
# #include expansion (focused unit tests)
# ----------------------------------------------------------------------


class ParserIncludeTest(unittest.TestCase):
    def _run(self, files: dict, top: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, content in files.items():
                _write(root / name, content)
            repo = load_from_directory(root)
            parsed = parse(repo)
            tpl = parsed.templates[top]
            assert isinstance(tpl, TemplatePass1)
            return [line.line for line in tpl.pass1]

    def test_simple_include(self) -> None:
        result = self._run(
            {
                "main.wgsl.template": '#include "utils.wgsl.template"\nmain\n',
                "utils.wgsl.template": "u1\nu2\n",
            },
            "main.wgsl.template",
        )
        # Trailing empty line from the LF terminator is preserved.
        self.assertEqual(result, ["u1", "u2", "", "main", ""])

    def test_circular_include_raises(self) -> None:
        with self.assertRaises(WgslTemplateParseError) as ctx:
            self._run(
                {
                    "a.wgsl.template": '#include "b.wgsl.template"\n',
                    "b.wgsl.template": '#include "a.wgsl.template"\n',
                },
                "a.wgsl.template",
            )
        self.assertIn("Circular #include", str(ctx.exception))

    def test_self_include_raises(self) -> None:
        with self.assertRaises(WgslTemplateParseError) as ctx:
            self._run(
                {"a.wgsl.template": '#include "a.wgsl.template"\n'},
                "a.wgsl.template",
            )
        self.assertIn("Circular #include", str(ctx.exception))

    def test_missing_include_raises(self) -> None:
        with self.assertRaises(WgslTemplateParseError) as ctx:
            self._run(
                {"a.wgsl.template": '#include "missing.wgsl.template"\n'},
                "a.wgsl.template",
            )
        self.assertIn("not found", str(ctx.exception))

    def test_unquoted_include_raises(self) -> None:
        with self.assertRaises(WgslTemplateParseError) as ctx:
            self._run(
                {"a.wgsl.template": "#include foo.wgsl.template\n"},
                "a.wgsl.template",
            )
        self.assertIn("double quotes", str(ctx.exception))


# ----------------------------------------------------------------------
# #define substitution (focused unit tests)
# ----------------------------------------------------------------------


class ParserMacroTest(unittest.TestCase):
    def test_basic_substitution(self) -> None:
        self.assertEqual(
            _parse_text("#define X 42\nlet a = X;\n"),
            ["", "let a = 42;", ""],
        )

    def test_whole_word_match(self) -> None:
        # Should NOT replace inside identifiers.
        self.assertEqual(
            _parse_text("#define X 42\nlet Xy = 1;\nlet X = 2;\n"),
            ["", "let Xy = 1;", "let 42 = 2;", ""],
        )

    def test_eager_expansion(self) -> None:
        # Y is defined first as 10. Then X uses Y, gets stored as 10.
        # Subsequent code uses X, which expands to 10.
        self.assertEqual(
            _parse_text("#define Y 10\n#define X Y\nlet a = X;\n"),
            ["", "", "let a = 10;", ""],
        )

    def test_self_reference_raises(self) -> None:
        with self.assertRaises(WgslTemplateParseError) as ctx:
            _parse_text("#define X X+1\n")
        self.assertIn("references itself", str(ctx.exception))

    def test_duplicate_raises(self) -> None:
        with self.assertRaises(WgslTemplateParseError) as ctx:
            _parse_text("#define X 1\n#define X 2\n")
        self.assertIn("Duplicate macro", str(ctx.exception))

    def test_empty_value_raises(self) -> None:
        with self.assertRaises(WgslTemplateParseError) as ctx:
            _parse_text("#define X\n")
        self.assertIn("no value", str(ctx.exception))

    def test_invalid_name_raises(self) -> None:
        with self.assertRaises(WgslTemplateParseError) as ctx:
            _parse_text("#define 9bad value\n")
        self.assertIn("invalid macro name", str(ctx.exception))


# ----------------------------------------------------------------------
# Fixture-driven tests against the parser-* and loader-* testcases
# ----------------------------------------------------------------------


def _build_fixture_suite() -> unittest.TestSuite:
    """Discover ``parser-*`` and ``loader-*`` fixture directories with
    ``.pass1`` golden files and build a unittest suite.

    Returns an empty suite if the fixtures aren't present, so the run
    stays green even in a trimmed checkout.
    """

    suite = unittest.TestSuite()
    if not _TESTCASES_DIR.is_dir():
        return suite

    for entry in sorted(os.listdir(_TESTCASES_DIR)):
        case_dir = _TESTCASES_DIR / entry
        if not case_dir.is_dir():
            continue
        if not entry.startswith(("parser-", "loader-")):
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
        # Skip loader-directories cases — they don't exercise PASS1
        # and need a different harness.
        if config.get("type") != "parser":
            continue

        case_class = _make_case(case_dir, config)
        loader = unittest.defaultTestLoader
        suite.addTests(loader.loadTestsFromTestCase(case_class))

    return suite


def _make_case(case_dir: Path, config: dict) -> type:
    """Build a one-off TestCase class for a single fixture directory."""

    case_name = case_dir.name
    expects_error = config.get("expectsError")

    class _Case(unittest.TestCase):
        # Shorter id so `-v` output is readable.
        def __init__(self, methodName: str = "runTest") -> None:
            super().__init__(methodName)
            self._dir = case_dir
            self._expects_error = expects_error

        def __str__(self) -> str:
            return f"fixture[{case_name}]"

        def runTest(self) -> None:  # noqa: N802
            try:
                repo = load_from_directory(self._dir)
                parsed = parse(repo)
            except Exception as e:
                if self._expects_error:
                    if isinstance(self._expects_error, str):
                        self.assertIn(self._expects_error, str(e))
                    return
                raise

            if self._expects_error:
                self.fail(f"Expected error containing {self._expects_error!r} but parse() succeeded")

            # Compare each template's pass1 lines against its .pass1 golden.
            for template_key, template in parsed.templates.items():
                assert isinstance(template, TemplatePass1)
                golden_path = self._dir / f"{template_key}.pass1"
                if not golden_path.is_file():
                    self.fail(f"Missing golden file for {template_key} at {golden_path}")
                expected_text = golden_path.read_text(encoding="utf-8")
                # Normalize line endings to match the loader's split.
                expected_lines = expected_text.replace("\r\n", "\n").split("\n")
                actual_lines = [line.line for line in template.pass1]
                self.assertEqual(
                    len(actual_lines),
                    len(expected_lines),
                    f"Line count mismatch for {template_key}: expected "
                    f"{len(expected_lines)} got {len(actual_lines)}\n"
                    f"actual: {actual_lines!r}\n"
                    f"expected: {expected_lines!r}",
                )
                for i, (a, b) in enumerate(zip(actual_lines, expected_lines, strict=True)):
                    self.assertEqual(
                        a,
                        b,
                        f"Line {i + 1} mismatch in {template_key}: actual={a!r} expected={b!r}",
                    )

    _Case.__name__ = f"Fixture_{case_name.replace('-', '_')}"
    _Case.__qualname__ = _Case.__name__
    return _Case


def load_tests(loader, standard_tests, pattern):
    standard_tests.addTests(_build_fixture_suite())
    return standard_tests


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
