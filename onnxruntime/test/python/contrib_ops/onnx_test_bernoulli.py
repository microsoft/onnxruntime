# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Test reference implementation and model for ONNX Runtime conrtib op bernoulli

import onnx
import unittest
import numpy as np
from onnx_contrib_ops_helper import expect


def bernoulli_reference_implementation(x):
    # binomial n = 1 equal bernoulli
    return np.random.binomial(1, p=x)

class ONNXReferenceImplementationTest(unittest.TestCase):
    def test_bernoulli_float(self):
        node = onnx.helper.make_node(
            'Bernoulli',
            inputs=['x'],
            outputs=['y'],
            domain="com.microsoft",
        )

        x = np.random.uniform(0.0, 1.0, 10).astype(np.float)
        y = bernoulli_reference_implementation(x)
        expect(node, inputs=[x], outputs=[y], name='test_bernoulli_float')

    def test_bernoulli_double(self):
        node = onnx.helper.make_node(
            'Bernoulli',
            inputs=['x'],
            outputs=['y'],
            domain="com.microsoft",
        )

        x = np.random.uniform(0.0, 1.0, 10).astype(np.double)
        y = bernoulli_reference_implementation(x)
        expect(node, inputs=[x], outputs=[y], name='test_bernoulli_double')

if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)
