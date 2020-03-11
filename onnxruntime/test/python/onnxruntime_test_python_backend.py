# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import sys
import numpy as np
from numpy.testing import assert_allclose
import onnxruntime as onnxrt
from onnxruntime import datasets
import onnxruntime.backend as backend
from onnxruntime.backend.backend import OnnxRuntimeBackend as ort_backend
from onnx import load
from helper import get_name


class TestBackend(unittest.TestCase):

    def testRunModel(self):
        name = get_name("mul_1.onnx")
        rep = backend.prepare(name)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        res = rep.run(x)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModelNonTensor(self):
        name = get_name("pipeline_vectorize.onnx")
        rep = backend.prepare(name)
        x = {0: 25.0, 1: 5.13, 2: 0.0, 3: 0.453, 4: 5.966}
        res = rep.run(x)
        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModelProto(self):
        name = datasets.get_example("logreg_iris.onnx")
        model = load(name)

        rep = backend.prepare(model)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        res = rep.run(x)
        output_expected = np.array([0, 0, 0], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)
        output_expected = [{0: 0.950599730014801, 1: 0.027834169566631317, 2: 0.02156602405011654},
                           {0: 0.9974970817565918, 1: 5.6299926654901356e-05, 2: 0.0024466661270707846},
                           {0: 0.9997311234474182, 1: 1.1918064757310276e-07, 2: 0.00026869276189245284}]
        self.assertEqual(output_expected, res[1])

    def testRunModelProtoApi(self):
        name = datasets.get_example("logreg_iris.onnx")
        model = load(name)

        inputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        outputs = ort_backend.run_model(model, inputs)

        output_expected = np.array([0, 0, 0], dtype=np.float32)
        np.testing.assert_allclose(output_expected, outputs[0], rtol=1e-05, atol=1e-08)
        output_expected = [{0: 0.950599730014801, 1: 0.027834169566631317, 2: 0.02156602405011654},
                           {0: 0.9974970817565918, 1: 5.6299926654901356e-05, 2: 0.0024466661270707846},
                           {0: 0.9997311234474182, 1: 1.1918064757310276e-07, 2: 0.00026869276189245284}]
        self.assertEqual(output_expected, outputs[1])

    def testAllocationPlanWorksWithOnlyExecutePathToFetchesOption(self):
        """
               (inp0)  (inp1)
                  |  \/  |
                  |  /\  |
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
        sess = onnxrt.InferenceSession(name)

        run_options = onnxrt.RunOptions()
        run_options.only_execute_path_to_fetches = True
        inp0, inp1 = np.ones((10,), dtype=np.float32), np.ones((10,), dtype=np.float32)

        session_run_results = sess.run(['outp0'], {'inp0': inp0, 'inp1': inp1}, run_options)
        assert_allclose(session_run_results[0], -(inp0 + inp1))

        session_run_results = sess.run(['outp1'], {'inp0': inp0, 'inp1': inp1}, run_options)
        assert_allclose(session_run_results[0], -(inp0 - inp1))


if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)
