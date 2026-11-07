# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Portable runner contract checks using a fake xcrun, not Apple execution."""

import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


class AppleTestRunnerTest(unittest.TestCase):
    def run_runner(self, *, count=1, status=0, test_filter="", archived=False, test_env=""):
        bash = os.environ.get("BAZEL_SH") or shutil.which("bash")
        if os.name == "nt":
            bash = os.environ.get("BAZEL_SH", r"C:\Program Files\Git\bin\bash.exe")
        self.assertTrue(bash and Path(bash).is_file(), "Runner tests require Bash (Git for Windows on Windows).")
        template = Path(__file__).with_name("apple_test_runner.template.sh").read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "inference.xctest"
            bundle.mkdir()
            (bundle / "fixture").write_text("fixture")
            source = bundle.name
            if archived:
                source = "inference.zip"
                with zipfile.ZipFile(root / source, "w") as archive:
                    archive.write(bundle / "fixture", "inference.xctest/fixture")
            script = template.replace("%(test_bundle_path)s", source).replace("%(test_env)s", test_env)
            (root / "runner.sh").write_text(script, newline="\n")
            mock = root / "xcrun"
            mock.write_text(
                "#!/bin/bash\n"
                '[[ -f "${@: -1}/fixture" ]] || exit 9\n'
                'printf "%s\\n" "$@" > arguments\n'
                'echo "${ORT_RUNNER_TEST_VALUE:-missing}"\n'
                f"echo 'Executed {count} tests, with 0 failures (0 unexpected) in 0.1 seconds'\n"
                f"exit {status}\n",
                newline="\n",
            )
            mock.chmod(0o755)
            env = dict(os.environ, TEST_TMPDIR=".", TESTBRIDGE_TEST_ONLY=test_filter)
            env["PATH"] = str(root) + os.pathsep + env["PATH"]
            result = subprocess.run([bash, "runner.sh"], cwd=root, env=env, text=True, capture_output=True, check=False)
            args = (root / "arguments").read_text().splitlines() if (root / "arguments").exists() else []
            self.assertFalse(list(root.glob("ort-xctest.*")), "Runner must clean its temporary bundle.")
            return result, args

    def test_selected_case_is_forwarded(self):
        result, args = self.run_runner(test_filter="ORTEnvTest/testInitOk")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[:3], ["xctest", "-XCTest", "ORTEnvTest/testInitOk"])

    def test_selected_class_is_forwarded(self):
        result, args = self.run_runner(test_filter="ORTEnvTest")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[:3], ["xctest", "-XCTest", "ORTEnvTest"])

    def test_swift_case_is_forwarded(self):
        test_filter = "ORTSwiftInferenceTest/testAddInference"
        result, args = self.run_runner(test_filter=test_filter)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[:3], ["xctest", "-XCTest", test_filter])

    def test_unfiltered_archive(self):
        result, args = self.run_runner(archived=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[:3], ["xctest", "-XCTest", "All"])

    def test_zero_matches_fail(self):
        result, _ = self.run_runner(count=0, test_filter="Missing/testMissing")
        self.assertEqual(result.returncode, 1)
        self.assertIn("executed no tests", result.stderr)

    def test_xctest_failure_is_preserved(self):
        result, _ = self.run_runner(status=7)
        self.assertEqual(result.returncode, 7)

    def test_test_environment_is_forwarded(self):
        result, _ = self.run_runner(test_env="ORT_RUNNER_TEST_VALUE=value with spaces")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("value with spaces", result.stdout)

    def test_invalid_filter_fails_before_execution(self):
        result, args = self.run_runner(test_filter="ORTEnvTest/*")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(args, [])
        self.assertIn("exact Class or Class/testMethod", result.stderr)


if __name__ == "__main__":
    unittest.main()
