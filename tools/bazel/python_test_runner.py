# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run registered existing unittest cases with exact module/class/method selection."""

import os
import sys
import unittest
from pathlib import Path

from python.runfiles import runfiles


def select_tests(names, selection):
    """Reject typos rather than accepting unittest's zero-test success."""
    selected = [name for name in names if not selection or name == selection or name.startswith(selection + ".")]
    if not selected:
        raise ValueError(
            f"No registered Python tests match {selection!r}. "
            "Use an exact module, module.Class, or module.Class.test_method from bazel/python_tests.bzl."
        )
    return selected


def main():
    resolver = runfiles.Create()
    helper = resolver.Rlocation("_main/onnxruntime/test/python/helper.py")
    if not helper or not Path(helper).is_file():
        raise FileNotFoundError("Python test helper is missing from Bazel runfiles; run the native py_test target.")
    sys.path.insert(0, str(Path(helper).parent))
    # Existing tests write profiles/serialized models to cwd; keep runfiles read-only.
    os.chdir(os.environ["TEST_TMPDIR"])
    names = select_tests(sys.argv[1:], os.environ.get("TESTBRIDGE_TEST_ONLY", ""))
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(names)
    if loader.errors:
        raise RuntimeError("Unable to load registered Python tests:\n" + "\n".join(loader.errors))
    if suite.countTestCases() == 0:
        raise RuntimeError("No Python tests were loaded; check the registered test names in bazel/python_tests.bzl.")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
