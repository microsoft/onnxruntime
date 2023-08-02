import unittest

import numpy as np
from helper import get_name

import onnxruntime as ort


class TestAmlEndpoint(unittest.TestCase):
    # test model loading with AzureExecutionProvider
    def test_addf(self):
        sess = ort.InferenceSession(get_name("matmul_1.onnx"), providers=["CPUExecutionProvider", "AzureExecutionProvider"])

if __name__ == "__main__":
    unittest.main()
