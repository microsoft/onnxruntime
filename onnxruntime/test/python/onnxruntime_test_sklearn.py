# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0116,W0212,R1720,C0103,C0114

import unittest

import numpy as np
from helper import get_name

import onnxruntime


class TestInferenceSessionSklearn(unittest.TestCase):
    def test_empty_sizes(self):
        # The model is valid assuming operators allow empty inputs.
        model = get_name("SklearnKNNImputer21.model.onnx")
        sess = onnxruntime.InferenceSession(model, providers=["CPUExecutionProvider"])
        feeds = dict(input=np.array([[1.3, 2.4, np.nan, 1], [-1.3, np.nan, 3.1, np.nan]], dtype=np.float32))
        got = sess.run(None, feeds)
        self.assertFalse(np.any(np.isnan(got[0])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
