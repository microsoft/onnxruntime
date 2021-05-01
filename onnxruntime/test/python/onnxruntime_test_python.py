# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import numpy as np

import onnxruntime as onnxrt
import threading
import sys
from helper import get_name
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

class TestInferenceSession(unittest.TestCase):

    def run_model(self, session_object, run_options):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = session_object.get_inputs()[0].name
        res = session_object.run([], {input_name: x}, run_options=run_options)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

if __name__ == '__main__':
    unittest.main()
