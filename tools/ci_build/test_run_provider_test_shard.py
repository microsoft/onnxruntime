# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_provider_test_shard import run_shard


@unittest.skipUnless(sys.platform.startswith("linux"), "Provider sharding requires native Linux")
class TestProviderTestShard(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ort shard test ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.binary = self.root / "provider_test"
        self.binary.touch()
        self.report = self.root / "provider.shard_2.results.xml"

    def launch(self, command, *, cwd, env, check):
        self.assertFalse(check)
        self.assertEqual(env["GTEST_TOTAL_SHARDS"], "4")
        self.assertEqual(env["GTEST_SHARD_INDEX"], "2")
        self.assertEqual(command[1:], ["--gtest_filter=Example.*", f"--gtest_output=xml:{self.report}"])
        self.assertNotEqual(cwd, self.root)
        self.assertEqual((cwd / "testdata").resolve(), self.root / "testdata")
        self.assertEqual((cwd.parent / "models").resolve(), self.root.parent / "models")
        self.assertEqual((cwd / "libprovider.so").resolve(), self.root / "libprovider.so")
        self.assertTrue(Path(env["TMPDIR"]).is_dir())
        self.assertEqual(env["TMPDIR"], env["TMP"])
        self.assertEqual(env["TMPDIR"], env["TEMP"])
        self.assertFalse(self.report.exists(), "A stale report must not satisfy the nonempty check")
        Path(env["GTEST_SHARD_STATUS_FILE"]).touch()
        self.report.write_text(
            '<testsuites><testsuite><testcase name="selected" status="run"/>'
            '<testcase name="disabled" status="notrun"/></testsuite></testsuites>',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    def run_with(self, launch):
        (self.root / "libprovider.so").touch()
        with patch("run_provider_test_shard.subprocess.run", side_effect=launch):
            return run_shard(self.binary, 2, self.report, ["--gtest_filter=Example.*"])

    def test_isolation_filter_forwarding_and_stale_report_removal(self):
        self.report.write_text("stale report", encoding="utf-8")
        with patch.dict(os.environ, {"GTEST_TOTAL_SHARDS": "99", "GTEST_SHARD_INDEX": "98"}):
            self.assertEqual(self.run_with(self.launch), 0)
        self.assertEqual(list(self.root.glob("provider-shard-*")), [])

    def test_nonzero_exit_is_propagated(self):
        self.assertEqual(self.run_with(lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 7)), 7)

    def test_missing_shard_acknowledgment_is_rejected(self):
        def launch(command, **kwargs):
            result = self.launch(command, **kwargs)
            Path(kwargs["env"]["GTEST_SHARD_STATUS_FILE"]).unlink()
            return result

        with self.assertRaisesRegex(RuntimeError, "did not acknowledge"):
            self.run_with(launch)

    def test_zero_selected_tests_are_rejected(self):
        def launch(command, **kwargs):
            result = self.launch(command, **kwargs)
            self.report.write_text(
                '<testsuites><testsuite><testcase status="notrun"/></testsuite></testsuites>',
                encoding="utf-8",
            )
            return result

        with self.assertRaisesRegex(RuntimeError, "executed zero tests"):
            self.run_with(launch)

    def test_missing_report_is_rejected(self):
        def launch(command, **kwargs):
            result = self.launch(command, **kwargs)
            self.report.unlink()
            return result

        with self.assertRaises(FileNotFoundError):
            self.run_with(launch)


class TestProviderShardRegistration(unittest.TestCase):
    def test_four_entries_share_binary_and_preserve_environment(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="ort shard cmake ") as directory:
            source = Path(directory)
            build = source / "build"
            (source / "CMakeLists.txt").write_text(
                f"""
cmake_minimum_required(VERSION 3.28)
project(ProviderShards LANGUAGES NONE)
enable_testing()
set(REPO_ROOT "{repo_root.as_posix()}")
set(Python_EXECUTABLE "{Path(sys.executable).as_posix()}")
include("${{REPO_ROOT}}/cmake/onnxruntime_provider_test_shards.cmake")
add_executable(provider IMPORTED)
set_target_properties(provider PROPERTIES IMPORTED_LOCATION "{Path(sys.executable).as_posix()}")
onnxruntime_add_provider_test_shards(provider "--gtest_filter=Example.*")
foreach(test_name IN LISTS onnxruntime_provider_test_names)
  set_tests_properties(${{test_name}} PROPERTIES ENVIRONMENT "PLUGIN_CONFIG=retained" LABELS "provider")
endforeach()
""",
                encoding="utf-8",
            )
            subprocess.run(
                ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Debug"],
                check=True,
                timeout=60,
            )
            result = subprocess.run(
                ["ctest", "--test-dir", str(build), "-C", "Debug", "--show-only=json-v1"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            tests = json.loads(result.stdout)["tests"]
            self.assertEqual(len(tests), 4)
            reports = set()
            for index, test in enumerate(tests):
                self.assertEqual(test["name"], f"onnxruntime_provider_test_shard_{index}")
                command = test["command"]
                self.assertEqual(command[command.index("--shard-index") + 1], str(index))
                self.assertEqual(command[command.index("--binary") + 1], str(Path(sys.executable)).replace("\\", "/"))
                self.assertEqual(command[-2:], ["--", "--gtest_filter=Example.*"])
                reports.add(command[command.index("--report") + 1])
                properties = {prop["name"]: prop["value"] for prop in test["properties"]}
                self.assertEqual(properties["TIMEOUT"], 10800)
                self.assertEqual(properties["PROCESSORS"], 4)
                self.assertEqual(properties["ENVIRONMENT"], ["PLUGIN_CONFIG=retained"])
                self.assertEqual(properties["LABELS"], ["provider"])
                self.assertEqual(
                    properties["ENVIRONMENT_MODIFICATION"],
                    ["ORT_INTRA_OP_NUM_THREADS=set:3", "ORT_INTER_OP_NUM_THREADS=set:2", "OMP_NUM_THREADS=set:3"],
                )
            self.assertEqual(len(reports), 4)
            self.assertTrue(all(report.endswith(".results.xml") and ".Debug.shard_" in report for report in reports))


if __name__ == "__main__":
    unittest.main()
