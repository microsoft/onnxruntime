# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import unittest
from pathlib import Path
from unittest import mock

CI_BUILD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI_BUILD_DIR))
sys.path.insert(0, str(CI_BUILD_DIR.parent / "python"))

from build_args import parse_arguments  # noqa: E402


class BuildArgumentsTest(unittest.TestCase):
    def test_cmake_extra_args(self):
        argv = [
            "build.py",
            "--build_dir",
            "build",
            "--update",
            "--cmake_extra_args=-Wno-dev",
            "--cmake_extra_args=--warn-uninitialized",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = parse_arguments()

        self.assertEqual(args.cmake_extra_args, ["-Wno-dev", "--warn-uninitialized"])


if __name__ == "__main__":
    unittest.main()
