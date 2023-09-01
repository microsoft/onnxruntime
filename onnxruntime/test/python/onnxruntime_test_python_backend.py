# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest

import numpy as np
from helper import get_name
from numpy.testing import assert_allclose

import onnxruntime as onnxrt
import onnxruntime.backend as backend


class TestBackend(unittest.TestCase):
    def test_run_model(self):
        name = get_name("mul_1.onnx")
        rep = backend.prepare(name)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        res = rep.run(x)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def test_allocation_plan_works_with_only_execute_path_to_fetches_option(self):
        """
               (inp0)  (inp1)
                  |  \\/  |
                  |  /\\  |
                 Add    Sub
                  |      |
              (tsor0)  (tsor1)
                  |      |
                 Neg    Neg
                  |      |
              (outp0)  (outp1)

        In this model, tsor0 and tsor1 has the same size. Allocation plan sets tsor1 to re-uses tsor0's memory.
        With run_options.only_execute_path_to_fetches == True and only to fetch outp1, the Add op is not executed.
        As a result tsor0 is not allocated through computation. It would fail to allocate tsor1 via re-use tsor0.
        This case is handled specifically in ExecutionFrame::AllocateAsPerAllocationPlan().
        This test is to ensure that the case is covered.
        """
        name = get_name("alloc_tensor_reuse.onnx")
        sess = onnxrt.InferenceSession(name, providers=onnxrt.get_available_providers())

        run_options = onnxrt.RunOptions()
        run_options.only_execute_path_to_fetches = True
        inp0, inp1 = np.ones((10,), dtype=np.float32), np.ones((10,), dtype=np.float32)

        session_run_results = sess.run(["outp0"], {"inp0": inp0, "inp1": inp1}, run_options)
        assert_allclose(session_run_results[0], -(inp0 + inp1))

        session_run_results = sess.run(["outp1"], {"inp0": inp0, "inp1": inp1}, run_options)
        assert_allclose(session_run_results[0], -(inp0 - inp1))


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
