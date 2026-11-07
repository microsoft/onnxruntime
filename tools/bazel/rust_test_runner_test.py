# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.bazel import rust_test_runner


class RustTestRunnerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.paths = {}
        for name in ("binary", "runtime", "rust/crate/Cargo.toml"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            self.paths[name] = str(path)
        self.listing = subprocess.CompletedProcess([], 0, stdout="sample::case: test\n", stderr="")

    def run_case(self, output="test result: ok. 1 passed; 0 failed; 0 ignored;", returncode=0, selection=None):
        resolver = Mock()
        resolver.Rlocation.side_effect = self.paths.get
        environment = {} if selection is None else {"TESTBRIDGE_TEST_ONLY": selection}
        with (
            patch.object(sys, "argv", ["runner", "binary", "runtime", "rust/crate/Cargo.toml", "sample::case"]),
            patch.dict(os.environ, environment, clear=True),
            patch.object(rust_test_runner.runfiles, "Create", return_value=resolver),
            patch.object(rust_test_runner.os, "chdir"),
            patch.object(
                rust_test_runner.subprocess,
                "run",
                side_effect=[self.listing, subprocess.CompletedProcess([], returncode, stdout=output)],
            ) as run,
        ):
            rust_test_runner.main()
            return run

    def test_runs_exact_case_with_explicit_runtime(self):
        run = self.run_case()
        self.assertIn("--exact", run.call_args.args[0])
        self.assertEqual(
            run.call_args.kwargs["env"]["RUST_ONNXRUNTIME_LIBRARY_PATH"],
            str(Path(self.paths["runtime"]).absolute()),
        )

    def test_unknown_selection_fails_before_execution(self):
        with self.assertRaisesRegex(ValueError, "Unknown or empty"):
            self.run_case(selection="not_a_test")

    def test_option_as_selection_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown or empty"):
            self.run_case(selection="--list")

    def test_empty_discovery_fails(self):
        self.listing.stdout = ""
        with self.assertRaisesRegex(ValueError, "no runnable libtest cases"):
            self.run_case()

    def test_discovery_error_preserves_diagnostics(self):
        self.listing.returncode = 1
        self.listing.stderr = "discovery diagnostic"
        with self.assertRaisesRegex(RuntimeError, "discovery diagnostic"):
            self.run_case()

    def test_zero_executed_fails(self):
        with self.assertRaisesRegex(RuntimeError, "did not execute and pass"):
            self.run_case(output="test result: ok. 0 passed; 0 failed; 0 ignored;")

    def test_ignored_only_fails(self):
        with self.assertRaisesRegex(RuntimeError, "did not execute and pass"):
            self.run_case(output="test result: ok. 0 passed; 0 failed; 1 ignored;")

    def test_failed_process_fails(self):
        with self.assertRaisesRegex(RuntimeError, "exit 1"):
            self.run_case(returncode=1)


if __name__ == "__main__":
    unittest.main()
