import unittest

import numpy as np
from helper import get_name

import onnxruntime as ort

class TestAzureEP(unittest.TestCase):

    def test_availability(self):
        self.assertTrue("AzureExecutionProvider" in ort.get_available_providers())


if __name__ == "__main__":
    unittest.main()
