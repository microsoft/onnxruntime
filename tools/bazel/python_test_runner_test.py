# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression coverage for exact, nonempty Python test selection."""

import unittest
from typing import ClassVar

from tools.bazel.python_test_runner import select_tests


class TestSelection(unittest.TestCase):
    names: ClassVar[list[str]] = ["module.Class.test_one", "module.Class.test_two", "other.Class.test_one"]

    def test_default(self):
        self.assertEqual(select_tests(self.names, ""), self.names)

    def test_module(self):
        self.assertEqual(select_tests(self.names, "module"), self.names[:2])

    def test_class(self):
        self.assertEqual(select_tests(self.names, "module.Class"), self.names[:2])

    def test_method(self):
        self.assertEqual(select_tests(self.names, self.names[0]), self.names[:1])

    def test_invalid_selection(self):
        for selection in ("missing", "module.Class.test_missing", "mod", "module.Class.test_*"):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "No registered Python tests"):
                select_tests(self.names, selection)

    def test_empty_registry(self):
        with self.assertRaisesRegex(ValueError, "No registered Python tests"):
            select_tests([], "")


if __name__ == "__main__":
    unittest.main()
