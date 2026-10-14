# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_disabled_tests

mock = unittest.mock


class DisabledTestsCheckerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def invoke(self, *args):
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["check_disabled_tests", "--test-dir", str(self.root), *args]),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
        ):
            result = check_disabled_tests.main()
        return result, output.getvalue()

    def test_macros_suites_multiline_and_multiple_declarations(self):
        self.write(
            "cases.cc",
            "TEST(Suite, DISABLED_One) {}\n"
            "TEST_F(DISABLED_Suite, Two) {}\n"
            "TEST_P(\n Suite,\n DISABLED_Three) {}\n"
            "TYPED_TEST(Suite, DISABLED_Four) {} TYPED_TEST_P(DISABLED_Suite, Five) {}\n"
            "TEST(DISABLED_Suite, DISABLED_Six) {}\nTEST(Suite, Enabled) {}\n",
        )
        hits = check_disabled_tests.find_disabled_tests(self.root)
        self.assertEqual([line for _, line, _ in hits], [1, 2, 3, 6, 6, 7])

    def test_comments_strings_and_non_source_files_are_excluded(self):
        source = (
            "// TEST(S, DISABLED_Comment)\n"
            "/*\n TEST_F(S, DISABLED_Block)\n */\n"
            'const char* s = "TEST_P(S, DISABLED_String)";\n'
            'const char* r = R"tag(TEST(S, DISABLED_Raw) " quoted)tag";\n'
            "TEST(/* explanation */ S, DISABLED_Real) {}\n"
        )
        self.write("cases.cu", source)
        self.write("cases.txt", "TEST(S, DISABLED_NotCpp)")
        hits = check_disabled_tests.find_disabled_tests(self.root)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][1], 7)

    def test_exact_baseline_both_directions_and_update(self):
        self.write("cases.cpp", "TEST(S, DISABLED_Test) {}")
        self.assertEqual(self.invoke("--baseline", "1")[0], 0)
        self.assertEqual(self.invoke("--baseline", "0")[0], 1)
        self.assertEqual(self.invoke("--baseline", "2")[0], 1)
        result, output = self.invoke("--update-baseline")
        self.assertEqual(result, 0)
        self.assertIn("_DEFAULT_BASELINE = 1", output)

    def test_external_and_relative_test_directories(self):
        self.write("cases.cxx", "TEST(S, DISABLED_Test) {}")
        repo_root = Path(check_disabled_tests.__file__).resolve().parents[2]
        source = self.root / "cases.cxx"
        display_path = source.relative_to(repo_root) if source.is_relative_to(repo_root) else source
        result, output = self.invoke("--baseline", "1", "--list")
        self.assertEqual(result, 0)
        self.assertIn(f"{display_path.as_posix()}:1:", output)
        relative = subprocess.run(
            [sys.executable, check_disabled_tests.__file__, "--test-dir", ".", "--baseline", "1", "--list"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(relative.returncode, 0, relative.stderr)
        self.assertIn(f"{display_path.as_posix()}:1:", relative.stdout)

    def test_read_failure_is_not_success(self):
        self.write("cases.cc", "TEST(S, DISABLED_Test) {}")
        with mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result, output = self.invoke("--baseline", "0")
        self.assertEqual(result, 2)
        self.assertIn("cannot scan disabled tests: denied", output)

    def test_directory_failure_is_not_success(self):
        def failed_walk(_root, *, onerror):
            onerror(PermissionError("directory denied"))

        with mock.patch.object(check_disabled_tests.os, "walk", side_effect=failed_walk):
            self.assertEqual(self.invoke("--baseline", "0")[0], 2)


if __name__ == "__main__":
    unittest.main()
