# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import sys
import numpy as np
import onnxruntime as onnxrt
from onnxruntime.capi._pybind_state import onnxruntime_ostream_redirect
from onnxruntime.sklapi import OnnxTransformer


class TestInferenceSessionSklearn(unittest.TestCase):
    
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

    def test_transform(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        name = self.get_name("mul_1.pb")
        with open(name, "rb") as f:
            content = f.read()
            
        tr = OnnxTransformer(content)
        tr.fit()
        res = tr.transform(x)
        exp = np.array([[ 1.,  4.], [ 9., 16.], [25., 36.]], dtype=np.float32)
        assert list(res.ravel()) == list(exp.ravel())
        

        
if __name__ == '__main__':
    unittest.main()
