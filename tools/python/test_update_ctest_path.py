#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

from tools.python.update_ctest_path import replace_ctest_paths, update_ctest_paths


class CTestPathReplacementTest(unittest.TestCase):
    def test_overlapping_roots_and_tools_are_replaced_without_cascading(self):
        replacements = {
            "C:/producer": "C:/producer/consumer",
            "C:/producer/build": "D:/build",
            "C:/producer/python/python.exe": "D:/Python/python.exe",
            "C:/producer/cmake/cmake.exe": "D:/CMake/cmake.exe",
        }
        paths = [*replacements, "C:/unrelated/test.exe"]
        text = "\n".join(paths + [path.replace("/", "\\\\") for path in paths])
        expected = "\n".join(
            [*replacements.values(), paths[-1], *replacements.values(), paths[-1].replace("/", "\\\\")]
        )
        self.assertEqual(replace_ctest_paths(text, replacements), expected)

    def test_paths_and_replacement_values_are_literal(self):
        self.assertEqual(
            replace_ctest_paths('add_test("C:/build[1]/test.exe")', {"C:/build[1]": r"D:/new\1"}),
            'add_test("D:/new\\1/test.exe")',
        )

    def test_header_based_updater_preserves_interface_and_relocates_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            ctest_file = Path(directory) / "CTestTestfile.cmake"
            ctest_file.write_text(
                "# Source directory: C:/producer/cmake\n"
                "# Build directory: C:/producer/build\n"
                'add_test(test "C:/producer/build/test.exe" "C:\\\\producer/data")\n',
                encoding="utf-8",
            )
            update_ctest_paths(ctest_file, Path("new-source"), Path("new-build"))
            self.assertEqual(
                ctest_file.read_text(encoding="utf-8"),
                "# Source directory: new-source/cmake\n"
                "# Build directory: new-build\n"
                'add_test(test "new-build/test.exe" "new-source/data")\n',
            )
            self.assertFalse(ctest_file.with_suffix(".cmake.tmp").exists())


if __name__ == "__main__":
    unittest.main()
