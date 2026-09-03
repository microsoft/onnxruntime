# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Unit tests for the PASS0 loader.

Uses temporary directories with inline fixtures.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Make the wgsl_template package importable regardless of cwd.
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from wgsl_template.errors import WgslTemplateLoadError  # noqa: E402
from wgsl_template.loader import (  # noqa: E402
    load_from_directories,
    load_from_directory,
)
from wgsl_template.types import SourceDir, TemplatePass0  # noqa: E402


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use binary write so callers control line endings exactly.
    path.write_bytes(content.encode("utf-8"))


class LoaderBasicTest(unittest.TestCase):
    def test_loads_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "basic.wgsl.template",
                "fn a() {\n    var x = 5;\n    var y = 10;\n}\n",
            )

            repo = load_from_directory(root)

            self.assertEqual(set(repo.templates), {"basic.wgsl.template"})
            tpl = repo.templates["basic.wgsl.template"]
            assert isinstance(tpl, TemplatePass0)
            # Trailing \n means split produces an empty trailing entry.
            self.assertEqual(
                tpl.raw,
                ["fn a() {", "    var x = 5;", "    var y = 10;", "}", ""],
            )

    def test_empty_directory_returns_empty_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = load_from_directory(tmp)
            self.assertEqual(repo.templates, {})

    def test_recurses_into_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "main.wgsl.template", "main\n")
            _write(root / "shaders" / "vertex.wgsl.template", "vertex\n")
            _write(root / "shaders" / "compute" / "matrix.wgsl.template", "matrix\n")

            repo = load_from_directory(root)

            self.assertEqual(
                set(repo.templates),
                {
                    "main.wgsl.template",
                    "shaders/vertex.wgsl.template",
                    "shaders/compute/matrix.wgsl.template",
                },
            )

    def test_template_keys_are_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["zeta.wgsl.template", "alpha.wgsl.template", "mu.wgsl.template"]:
                _write(root / name, "x\n")

            repo = load_from_directory(root)

            self.assertEqual(
                list(repo.templates),
                ["alpha.wgsl.template", "mu.wgsl.template", "zeta.wgsl.template"],
            )

    def test_skips_files_with_other_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "keep.wgsl.template", "keep\n")
            _write(root / "drop.txt", "drop\n")
            _write(root / "drop.wgsl", "drop\n")

            repo = load_from_directory(root)

            self.assertEqual(set(repo.templates), {"keep.wgsl.template"})

    def test_uses_posix_paths_in_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "a" / "b" / "c.wgsl.template", "x\n")

            repo = load_from_directory(root)

            self.assertEqual(set(repo.templates), {"a/b/c.wgsl.template"})


class LoaderLineEndingsTest(unittest.TestCase):
    def test_lf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "lf.wgsl.template", "line1\nline2\n")

            repo = load_from_directory(root)
            self.assertEqual(repo.templates["lf.wgsl.template"].raw, ["line1", "line2", ""])

    def test_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "crlf.wgsl.template", "line1\r\nline2\r\n")

            repo = load_from_directory(root)
            self.assertEqual(repo.templates["crlf.wgsl.template"].raw, ["line1", "line2", ""])

    def test_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "mixed.wgsl.template", "a\r\nb\nc\r\n")

            repo = load_from_directory(root)
            self.assertEqual(repo.templates["mixed.wgsl.template"].raw, ["a", "b", "c", ""])


class LoaderExtensionTest(unittest.TestCase):
    def test_custom_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "shader.wgsl.custom", "x\n")
            _write(root / "ignored.wgsl.template", "x\n")

            repo = load_from_directory(root, ext=".wgsl.custom")
            self.assertEqual(set(repo.templates), {"shader.wgsl.custom"})


class LoaderMultipleDirectoriesTest(unittest.TestCase):
    def test_loads_from_multiple_dirs_without_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d1, d2 = root / "d1", root / "d2"
            _write(d1 / "a.wgsl.template", "a\n")
            _write(d2 / "b.wgsl.template", "b\n")

            repo = load_from_directories([d1, d2])
            self.assertEqual(set(repo.templates), {"a.wgsl.template", "b.wgsl.template"})

    def test_loads_with_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d1, d2 = root / "templates", root / "shaders"
            _write(d1 / "base.wgsl.template", "base\n")
            _write(d2 / "vertex.wgsl.template", "vertex\n")

            repo = load_from_directories(
                [
                    SourceDir(path=str(d1), alias="@templates"),
                    SourceDir(path=str(d2), alias="@shaders"),
                ]
            )

            self.assertEqual(
                set(repo.templates),
                {
                    "@templates/base.wgsl.template",
                    "@shaders/vertex.wgsl.template",
                },
            )

    def test_template_name_conflict_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d1, d2 = root / "d1", root / "d2"
            _write(d1 / "shared.wgsl.template", "v1\n")
            _write(d2 / "shared.wgsl.template", "v2\n")

            with self.assertRaises(WgslTemplateLoadError) as ctx:
                load_from_directories([d1, d2])
            self.assertIn("Template name conflict", str(ctx.exception))


class LoaderErrorsTest(unittest.TestCase):
    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(WgslTemplateLoadError) as ctx:
            load_from_directory("/no/such/path/does/not/exist/abc123")
        self.assertIn("Cannot access directory", str(ctx.exception))

    def test_path_is_a_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "notadir"
            file_path.write_text("x")
            with self.assertRaises(WgslTemplateLoadError) as ctx:
                load_from_directory(file_path)
            self.assertIn("not a directory", str(ctx.exception))

    def test_no_directories_raises(self) -> None:
        with self.assertRaises(WgslTemplateLoadError):
            load_from_directories([])


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
