# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import sys
import numpy as np
import onnxruntime as onnxrt
from onnxruntime import datasets
import onnxruntime.backend as backend
from onnxruntime.backend.backend import OnnxRuntimeBackend as ort_backend
from onnx import load


class TestBackend(unittest.TestCase):

    def get_name(self, name):
        if os.path.exists(name):
            return name
        rel = os.path.join("testdata", name)
        if os.path.exists(rel):
            return rel
        this = os.path.dirname(__file__)
        data = os.path.join(this, "..", "testdata")
        res = os.path.join(data, name)
        if os.path.exists(res):
            return res
        raise FileNotFoundError("Unable to find '{0}' or '{1}' or '{2}'".format(name, rel, res))

    def testRunModel(self):
        name = self.get_name("mul_1.pb")
        rep = backend.prepare(name)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        res = rep.run(x)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModelNonTensor(self):
        name = self.get_name("pipeline_vectorize.onnx")
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


if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)
