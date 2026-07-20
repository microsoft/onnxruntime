# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression test for issue #23108: PEP 561 `py.typed` marker presence."""

from __future__ import annotations

import os
import unittest

import onnxruntime


class TestPep561Marker(unittest.TestCase):
    def test_py_typed_marker_exists_in_installed_package(self):
        package_root = os.path.dirname(onnxruntime.__file__)
        marker_path = os.path.join(package_root, "py.typed")
        self.assertTrue(
            os.path.isfile(marker_path),
            f"PEP 561 marker not found at {marker_path}; type checkers will fall back to 'import-untyped'.",
        )


if __name__ == "__main__":
    unittest.main()
