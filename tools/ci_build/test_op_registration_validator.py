#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from op_registration_validator import RegistrationValidator


class RegistrationValidatorTest(unittest.TestCase):
    def _validate(self, operator, versions):
        validator = RegistrationValidator()
        for start_version, end_version in versions:
            validator.process_registration([], "kOnnxDomain", operator, start_version, end_version)
        return validator.ok()

    def test_space_to_depth_native_kernels_end_before_function_version(self):
        self.assertTrue(self._validate("SpaceToDepth", [(1, 12), (13, 27)]))

    def test_space_to_depth_function_exception_requires_exact_boundary(self):
        for end_version in (26, 28):
            with self.subTest(end_version=end_version):
                self.assertFalse(self._validate("SpaceToDepth", [(1, 12), (13, end_version)]))

    def test_space_to_depth_function_exception_does_not_allow_registration_gaps(self):
        self.assertFalse(self._validate("SpaceToDepth", [(1, 11), (13, 27)]))

    def test_space_to_depth_accepts_future_native_kernel(self):
        self.assertTrue(self._validate("SpaceToDepth", [(1, 12), (13, 27), (28, None)]))

    def test_other_operators_still_require_unversioned_registration(self):
        self.assertFalse(self._validate("DepthToSpace", [(1, 12), (13, 27)]))
        self.assertTrue(self._validate("DepthToSpace", [(1, 12), (13, None)]))


if __name__ == "__main__":
    unittest.main()
