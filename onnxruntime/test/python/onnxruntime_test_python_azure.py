# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import onnxruntime as ort


class TestAzureEP(unittest.TestCase):
    def test_availability(self):
        self.assertIn("AzureExecutionProvider", ort.get_available_providers())


if __name__ == "__main__":
    unittest.main()
