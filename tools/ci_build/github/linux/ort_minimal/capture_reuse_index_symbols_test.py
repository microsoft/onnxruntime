# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import pathlib
import unittest

from capture_reuse_index_symbols import compile_command, link_command, parse_link_command


class CaptureReuseIndexSymbolsTest(unittest.TestCase):
    def test_redirects_actual_cmake_linker_dependency_output(self):
        command = [
            "clang++",
            "-Os",
            "-flto=thin",
            "-Xlinker",
            "--dependency-file=CMakeFiles/onnxruntime.dir/link.d",
            "-o",
            "normal.so",
        ]
        updated = link_command(command, pathlib.Path("/fresh/control.so"), pathlib.Path("/fresh/link.map"))
        self.assertIn("--dependency-file=/fresh/link.d", updated)
        self.assertNotIn("--dependency-file=CMakeFiles/onnxruntime.dir/link.d", updated)
        self.assertIn("--dependency-file=CMakeFiles/onnxruntime.dir/link.d", command)

    def test_preserves_shell_escaped_origin_without_variable_evaluation(self):
        command = parse_link_command(r'clang++ -Xlinker "-rpath=\$ORIGIN" -o normal.so', pathlib.Path("."))
        self.assertEqual(command, ["clang++", "-Xlinker", "-rpath=$ORIGIN", "-o", "normal.so"])

    def test_link_does_not_change_inputs_or_optimization(self):
        original = [
            "clang++",
            "-Os",
            "-flto",
            "-s",
            "api.o",
            "-o",
            "normal.so",
            "libonnxruntime_framework.a",
            "common.a",
        ]
        updated = link_command(original, pathlib.Path("fresh.so"), pathlib.Path("fresh.map"))
        self.assertEqual(
            updated,
            [
                "clang++",
                "-Os",
                "-flto",
                "api.o",
                "-o",
                "fresh.so",
                "libonnxruntime_framework.a",
                "common.a",
                "-Wl,-Map=fresh.map",
            ],
        )
        self.assertIn("-s", original)
        self.assertEqual(original[original.index("-o") + 1], "normal.so")

    def test_archive_replacement_is_unique_and_only_changes_framework(self):
        original = ["clang++", "-o", "normal.so", "/build/libonnxruntime_framework.a", "common.a"]
        updated = link_command(original, pathlib.Path("fresh.so"), pathlib.Path("fresh.map"), pathlib.Path("base.a"))
        self.assertEqual(updated[-3:], ["base.a", "common.a", "-Wl,-Map=fresh.map"])
        with self.assertRaises(ValueError):
            link_command(
                ["clang++", "-o", "normal.so"],
                pathlib.Path("fresh.so"),
                pathlib.Path("fresh.map"),
                pathlib.Path("base.a"),
            )

    def test_compile_redirects_all_generated_files_and_preserves_flags(self):
        original = [
            "clang++",
            "-Os",
            "-flto",
            "-DANDROID",
            "-MD",
            "-MT",
            "old.o",
            "-MF",
            "old.d",
            "-o",
            "old.o",
            "-c",
            "/src/allocation_planner.cc",
        ]
        updated = compile_command(
            original,
            pathlib.Path("/fresh/allocation_planner.cc"),
            pathlib.Path("/fresh/allocation_planner.cc.o"),
            "/src/allocation_planner.cc",
        )
        self.assertEqual(updated[:5], original[:5])
        self.assertEqual(updated[updated.index("-MT") + 1], "/fresh/allocation_planner.cc.o")
        self.assertEqual(updated[updated.index("-MF") + 1], "/fresh/allocation_planner.cc.o.d")
        self.assertEqual(updated[updated.index("-o") + 1], "/fresh/allocation_planner.cc.o")
        self.assertEqual(updated[updated.index("-c") + 1], "/fresh/allocation_planner.cc")
        self.assertEqual(updated[-1], "-ffile-prefix-map=/fresh/allocation_planner.cc=/src/allocation_planner.cc")
        self.assertEqual(original[original.index("-MF") + 1], "old.d")


if __name__ == "__main__":
    unittest.main()
