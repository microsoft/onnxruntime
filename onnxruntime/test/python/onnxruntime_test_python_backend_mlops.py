# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest

import numpy as np
from helper import get_name
from onnx import load

import onnxruntime.backend as backend
from onnxruntime import datasets
from onnxruntime.backend.backend import OnnxRuntimeBackend as ort_backend  # noqa: N813


def check_list_of_map_to_float(testcase, expected_rows, actual_rows):
    """Validate two list<map<key, float>> instances match closely enough."""

    num_rows = len(expected_rows)
    sorted_keys = sorted(expected_rows[0].keys())
    testcase.assertEqual(num_rows, len(actual_rows))
    testcase.assertEqual(sorted_keys, sorted(actual_rows[0].keys()))

    for i in range(num_rows):
        # use np.testing.assert_allclose so we can specify the tolerance
        np.testing.assert_allclose(
            [expected_rows[i][key] for key in sorted_keys],
            [actual_rows[i][key] for key in sorted_keys],
            rtol=1e-05,
            atol=1e-07,
        )


class TestBackend(unittest.TestCase):
    def test_run_model_non_tensor(self):
        name = get_name("pipeline_vectorize.onnx")
        rep = backend.prepare(name)
        x = {0: 25.0, 1: 5.13, 2: 0.0, 3: 0.453, 4: 5.966}
        res = rep.run(x)
        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def test_run_model_proto(self):
        name = datasets.get_example("logreg_iris.onnx")
        model = load(name)

        rep = backend.prepare(model)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        res = rep.run(x)
        output_expected = np.array([0, 0, 0], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)
        output_expected = [
            {0: 0.950599730014801, 1: 0.027834169566631317, 2: 0.02156602405011654},
            {
                0: 0.9974970817565918,
                1: 5.6299926654901356e-05,
                2: 0.0024466661270707846,
            },
            {
                0: 0.9997311234474182,
                1: 1.1918064757310276e-07,
                2: 0.00026869276189245284,
            },
        ]

        check_list_of_map_to_float(self, output_expected, res[1])

    def test_run_model_proto_api(self):
        name = datasets.get_example("logreg_iris.onnx")
        model = load(name)

        inputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        outputs = ort_backend.run_model(model, inputs)

        output_expected = np.array([0, 0, 0], dtype=np.float32)
        np.testing.assert_allclose(output_expected, outputs[0], rtol=1e-05, atol=1e-08)
        output_expected = [
            {0: 0.950599730014801, 1: 0.027834169566631317, 2: 0.02156602405011654},
            {
                0: 0.9974970817565918,
                1: 5.6299926654901356e-05,
                2: 0.0024466661270707846,
            },
            {
                0: 0.9997311234474182,
                1: 1.1918064757310276e-07,
                2: 0.00026869276189245284,
            },
        ]

        check_list_of_map_to_float(self, output_expected, outputs[1])


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
