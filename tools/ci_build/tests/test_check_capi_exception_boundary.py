# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import contextlib
import io
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_capi_exception_boundary as checker


class CapiExceptionBoundaryCheckerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "boundary.cc"
        self.source.write_text("API_IMPL_BEGIN\nAPI_IMPL_END\n", encoding="utf-8")

    def invoke(self, root):
        output = io.StringIO()
        with (
            unittest.mock.patch.object(sys, "argv", ["checker", "--root", str(root), "--list"]),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
        ):
            result = checker.main()
        return result, output.getvalue()

    def test_repo_relative_root(self):
        repo_root = Path(checker.__file__).resolve().parents[2]
        relative_root = self.root.relative_to(repo_root)
        result, output = self.invoke(relative_root)
        self.assertEqual(result, 0, output)
        self.assertIn((relative_root / "boundary.cc").as_posix(), output)
        self.assertIn(f"under {relative_root.as_posix()}.", output)

    def test_absolute_root_outside_repo(self):
        with unittest.mock.patch.object(
            checker, "__file__", str(self.root / "other" / "tools" / "ci_build" / "checker.py")
        ):
            result, output = self.invoke(self.root)
        self.assertEqual(result, 0, output)
        self.assertIn(self.source.as_posix(), output)
        self.assertIn(f"under {self.root.as_posix()}.", output)

    def test_unbalanced_external_source_reports_failure(self):
        self.source.write_text("API_IMPL_BEGIN\n", encoding="utf-8")
        result, output = self.invoke(self.root)
        self.assertEqual(result, 1, output)
        self.assertIn("1 opening macro(s) vs 0 API_IMPL_END", output)

    def test_missing_root_reports_usage_error(self):
        result, output = self.invoke(self.root / "missing")
        self.assertEqual(result, 2, output)
        self.assertIn("root directory not found", output)

    def test_macro_definitions_and_line_comments_are_not_call_sites(self):
        self.assertEqual(
            checker.count_boundaries(
                "#define OPEN \\\nAPI_IMPL_BEGIN\n// API_IMPL_END\nAPI_IMPL_BEGIN\nAPI_IMPL_END\n"
            ),
            (1, 1),
        )


if __name__ == "__main__":
    unittest.main()
