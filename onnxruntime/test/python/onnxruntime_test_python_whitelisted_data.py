# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import unittest

import numpy as np
from helper import get_name

import onnxruntime as ort


class TestWhitelistedData(unittest.TestCase):
    def test_whitelisted_data(self):
        # We use the existing test data:
        # Model: testdata/whitelist/model/test_whitelist_external_data.onnx
        # Data: testdata/whitelist/data/test_whitelist_data.bin
        # The model references "../data/test_whitelist_data.bin"

        try:
            model_path = get_name("whitelist/model/test_whitelist_external_data.onnx")
        except FileNotFoundError:
            # Fallback if running from build directory or similar where layouts differ
            # Try to construct path manually if helper fails or adjust expectation
            # For now assume helper works as per analysis
            raise

        # We need to whitelist the directory containing the data file
        model_dir = os.path.dirname(os.path.abspath(model_path))
        data_dir = os.path.normpath(os.path.join(model_dir, "..", "data"))

        # Verify data file exists
        data_file = os.path.join(data_dir, "test_whitelist_data.bin")
        self.assertTrue(os.path.exists(data_file), f"Data file not found at {data_file}")

        so = ort.SessionOptions()
        so.whitelisted_data_folders = data_dir

        # Verify the property was set correctly
        self.assertEqual(so.whitelisted_data_folders, data_dir)

        # Create session
        sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

        # The model adds a constant (from external data) to input.
        # Constant is sequence of 100 floats: 0.0, 1.0, ..., 99.0
        # Input shape is [100]

        input_data = np.zeros(100, dtype=np.float32)
        res = sess.run(["output"], {"input": input_data})

        # Expected output is just the constant values since input is 0
        expected = np.array([float(i) for i in range(100)], dtype=np.float32)
        np.testing.assert_allclose(res[0], expected)

    def test_whitelisted_data_failure(self):
        # Test that loading fails if not whitelisted
        model_path = get_name("whitelist/model/test_whitelist_external_data.onnx")

        so = ort.SessionOptions()
        # Don't set whitelist
        with self.assertRaises(Exception) as cm:
            ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

        # We expect an error about external data not being in whitelisted directories
        self.assertIn("External data path validation failed", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
