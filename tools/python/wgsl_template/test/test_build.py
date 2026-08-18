# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for the top-level build() orchestrator.

Combines focused unit tests with a fixture-driven runner over the
``build-*`` cases in ``testcases/``. Generated files are compared
against the ``expected/<generator>/`` golden tree using
``canonicalize`` (content-equivalence, not byte-for-byte), so
host-dependent ``__str_N`` numbering / ordering / sha256 markers don't
cause spurious failures. See ``golden_compare.py``.
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

from golden_compare import canonicalize, read_tree  # noqa: E402
from wgsl_template import build  # noqa: E402
from wgsl_template.errors import WgslTemplateError  # noqa: E402
from wgsl_template.types import SourceDir  # noqa: E402

_TESTCASES_DIR = _THIS_DIR / "testcases"

_SUPPORTED_GENERATORS = {"static-cpp", "static-cpp-literal"}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))


def _walk_files(root: Path) -> list[str]:
    """Return every file under ``root`` as a sorted POSIX path
    relative to ``root``."""
    out: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            full = Path(dirpath) / f
            out.append(full.relative_to(root).as_posix())
    out.sort()
    return out


# ----------------------------------------------------------------------
# Focused unit tests
# ----------------------------------------------------------------------


class BuildIdempotentWriteTest(unittest.TestCase):
    def test_unchanged_file_is_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            out = root / "out"
            _write(src / "a.wgsl.template", "a\n")

            # First build.
            build(
                source_dirs=[src],
                out_dir=out,
                generator="static-cpp-literal",
            )
            mtimes_before = {p: (out / p).stat().st_mtime_ns for p in _walk_files(out)}
            self.assertTrue(mtimes_before, "expected files in output dir")

            # Second build with identical sources — files must not be rewritten.
            build(
                source_dirs=[src],
                out_dir=out,
                generator="static-cpp-literal",
            )
            mtimes_after = {p: (out / p).stat().st_mtime_ns for p in _walk_files(out)}
            self.assertEqual(mtimes_before, mtimes_after)

    def test_changed_file_is_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            out = root / "out"
            _write(src / "a.wgsl.template", "a\n")

            build(source_dirs=[src], out_dir=out, generator="static-cpp-literal")
            content_before = (out / "generated/a.h").read_bytes()

            _write(src / "a.wgsl.template", "completely different\n")
            build(source_dirs=[src], out_dir=out, generator="static-cpp-literal")
            content_after = (out / "generated/a.h").read_bytes()

            self.assertNotEqual(content_before, content_after)


class BuildCleanFlagTest(unittest.TestCase):
    def test_clean_removes_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            out = root / "out"
            _write(src / "a.wgsl.template", "a\n")
            _write(out / "stale.h", "STALE\n")

            build(source_dirs=[src], out_dir=out, generator="static-cpp-literal", clean=True)
            self.assertFalse((out / "stale.h").exists())
            self.assertTrue((out / "generated/a.h").exists())


class BuildErrorsTest(unittest.TestCase):
    def test_empty_source_dirs_raises(self) -> None:
        with self.assertRaises(WgslTemplateError):
            build(source_dirs=[], out_dir="/tmp/anywhere")


# ----------------------------------------------------------------------
# Fixture-driven tests against the build-* cases
# ----------------------------------------------------------------------


def _build_fixture_suite() -> unittest.TestSuite:
    suite = unittest.TestSuite()
    if not _TESTCASES_DIR.is_dir():
        return suite

    for entry in sorted(os.listdir(_TESTCASES_DIR)):
        case_dir = _TESTCASES_DIR / entry
        if not case_dir.is_dir():
            continue
        if not entry.startswith("build-"):
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
        if config.get("type") != "build":
            continue

        # Filter generators down to ones we support.
        gen_cfg = config.get("generators") or {}
        applicable = {name: cfg for name, cfg in gen_cfg.items() if name in _SUPPORTED_GENERATORS}
        if not applicable:
            continue

        case_class = _make_build_case(case_dir, config, applicable)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case_class))
    return suite


def _make_build_case(case_dir: Path, config: dict, applicable: dict) -> type:
    case_name = case_dir.name
    template_ext = config.get("templateExt") or ".wgsl.template"
    source_dirs_cfg = config.get("sourceDirs")

    class _Case(unittest.TestCase):
        def __init__(self, methodName: str = "runTest") -> None:
            super().__init__(methodName)
            self._dir = case_dir
            self._template_ext = template_ext
            self._source_dirs_cfg = source_dirs_cfg
            self._applicable = applicable

        def __str__(self) -> str:
            return f"fixture[{case_name}]"

        def runTest(self) -> None:  # noqa: N802
            src_dir = self._dir / "src"
            expected_dir = self._dir / "expected"
            self.assertTrue(src_dir.is_dir(), f"missing src dir: {src_dir}")
            self.assertTrue(expected_dir.is_dir(), f"missing expected dir: {expected_dir}")

            for gen_name, gen_cfg in self._applicable.items():
                with tempfile.TemporaryDirectory() as tmp:
                    out_dir = Path(tmp)

                    if self._source_dirs_cfg:
                        sources: list = []
                        for d in self._source_dirs_cfg:
                            if isinstance(d, str):
                                sources.append(src_dir / d)
                            else:
                                sources.append(
                                    SourceDir(
                                        path=str(src_dir / d["path"]),
                                        alias=d.get("alias"),
                                    )
                                )
                    else:
                        sources = [src_dir]

                    expects_error = gen_cfg.get("expectsError")
                    try:
                        build(
                            source_dirs=sources,
                            out_dir=out_dir,
                            template_ext=self._template_ext,
                            generator=gen_name,
                        )
                    except Exception as e:
                        if expects_error:
                            if isinstance(expects_error, str):
                                self.assertIn(expects_error, str(e))
                            return
                        raise

                    if expects_error:
                        self.fail(f"{gen_name}: expected error containing {expects_error!r} but build succeeded")

                    expected_gen = expected_dir / gen_name
                    self.assertTrue(
                        expected_gen.is_dir(),
                        f"missing expected/{gen_name} dir for {case_name}",
                    )

                    actual_files = _walk_files(out_dir)
                    expected_files = _walk_files(expected_gen)

                    self.assertEqual(
                        actual_files,
                        expected_files,
                        f"{case_name} ({gen_name}): file set mismatch\n"
                        f"actual:   {actual_files}\n"
                        f"expected: {expected_files}",
                    )

                    # Compare in canonical form (see golden_compare).
                    actual_canon = canonicalize(read_tree(out_dir))
                    expected_canon = canonicalize(read_tree(expected_gen))

                    for rel in expected_files:
                        self.assertEqual(
                            actual_canon[rel],
                            expected_canon[rel],
                            f"{case_name} ({gen_name}): {rel} differs (after canonicalization)\n"
                            f"actual:\n{actual_canon[rel]!r}\n"
                            f"expected:\n{expected_canon[rel]!r}",
                        )

    _Case.__name__ = f"BuildFixture_{case_name.replace('-', '_')}"
    _Case.__qualname__ = _Case.__name__
    return _Case


def load_tests(loader, standard_tests, pattern):
    standard_tests.addTests(_build_fixture_suite())
    return standard_tests


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
