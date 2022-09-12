#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy

from onnxruntime.quantization.quant_utils import compute_scale_zp


class TestQuantUtil(unittest.TestCase):
    def test_compute_scale_zp(self):
        self.assertEqual(compute_scale_zp(0.0, 0.0, -127, 127, symmetric=True), [0, 1.0])
        self.assertEqual(compute_scale_zp(1.0, -1.0, -127, 127, symmetric=True), [0, 1.0])
        self.assertEqual(compute_scale_zp(0.0, 0.0, 0, 255, symmetric=True), [0, 1.0])
        self.assertEqual(compute_scale_zp(1.0, -1.0, 0, 255, symmetric=True), [0, 1.0])

        self.assertEqual(compute_scale_zp(-1.0, 2.0, -127, 127, symmetric=True), [0, 2.0 / 127])
        self.assertEqual(compute_scale_zp(-1.0, 2.0, -127, 127, symmetric=False), [-42, 3.0 / 254])

        self.assertEqual(compute_scale_zp(-1.0, 2.0, 0, 255, symmetric=True), [128, 4.0 / 255])
        self.assertEqual(compute_scale_zp(-1.0, 2.0, 0, 255, symmetric=False), [85, 3.0 / 255])

        tiny_float = numpy.float32(numpy.finfo(numpy.float32).tiny * 0.1)
        self.assertEqual(compute_scale_zp(-tiny_float, tiny_float, 0, 255, symmetric=True), [0, 1.0])
        self.assertEqual(compute_scale_zp(-tiny_float, 0.0, 0, 255, symmetric=False), [0, 1.0])


if __name__ == "__main__":
    unittest.main()
