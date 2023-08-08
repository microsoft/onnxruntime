import unittest

import numpy as np
from helper import get_name

import onnxruntime as ort


class TestAmlEndpoint(unittest.TestCase):
    # test an endpoint of adding floats
    def test_addf(self):
        sess_opt = ort.SessionOptions()
        sess_opt.add_session_config_entry("azure.endpoint_type", "triton")
        sess_opt.add_session_config_entry("azure.uri", "https://endpoint-2930.westus2.inference.ml.azure.com")
        sess_opt.add_session_config_entry("azure.model_name", "addf")
        sess_opt.add_session_config_entry("azure.model_version", "1")
        sess_opt.add_session_config_entry("azure.verbose", "true")

        sess = ort.InferenceSession(get_name("azure_models/addf.onnx"), sess_opt, providers=["CPUExecutionProvider"])

        run_opt = ort.RunOptions()
        run_opt.log_severity_level = 1
        run_opt.add_run_config_entry("use_azure", "1")
        run_opt.add_run_config_entry("azure.auth_key", "lVUg86Ba7abAZU3GeKAMEtYy1wy9LacX")

        x = np.array([1, 2, 3, 4]).astype(np.float32)
        y = np.array([4, 3, 2, 1]).astype(np.float32)

        z = sess.run(None, {"X": x, "Y": y}, run_opt)[0]

        expected_z = np.array([5, 5, 5, 5]).astype(np.float32)
        np.testing.assert_allclose(z, expected_z, rtol=1e-05, atol=1e-08)

    # test an endpoint of adding doubles
    def test_addf8(self):
        sess_opt = ort.SessionOptions()
        sess_opt.add_session_config_entry("azure.endpoint_type", "triton")
        sess_opt.add_session_config_entry("azure.uri", "https://endpoint-1364.westus2.inference.ml.azure.com")
        sess_opt.add_session_config_entry("azure.model_name", "addf8")
        sess_opt.add_session_config_entry("azure.model_version", "1")
        sess_opt.add_session_config_entry("azure.verbose", "true")

        sess = ort.InferenceSession(get_name("azure_models/addf8.onnx"), sess_opt, providers=["CPUExecutionProvider"])

        run_opt = ort.RunOptions()
        run_opt.log_severity_level = 1
        run_opt.add_run_config_entry("use_azure", "1")
        run_opt.add_run_config_entry("azure.auth_key", "bUxzw3kZxJMjUbntLcVU3Duqq1nc87m5")

        x = np.array([1, 2, 3, 4]).astype(np.double)
        y = np.array([4, 3, 2, 1]).astype(np.double)

        z = sess.run(None, {"X": x, "Y": y}, run_opt)[0]

        expected_z = np.array([5, 5, 5, 5]).astype(np.double)
        np.testing.assert_allclose(z, expected_z, rtol=1e-05, atol=1e-08)

    # test an endpoint of adding int
    def test_addi4(self):
        sess_opt = ort.SessionOptions()
        sess_opt.add_session_config_entry("azure.endpoint_type", "triton")
        sess_opt.add_session_config_entry("azure.uri", "https://endpoint-9879.westus2.inference.ml.azure.com")
        sess_opt.add_session_config_entry("azure.model_name", "addi4")
        sess_opt.add_session_config_entry("azure.model_version", "1")
        sess_opt.add_session_config_entry("azure.verbose", "true")

        sess = ort.InferenceSession(get_name("azure_models/addi4.onnx"), sess_opt, providers=["CPUExecutionProvider"])

        run_opt = ort.RunOptions()
        run_opt.log_severity_level = 1
        run_opt.add_run_config_entry("use_azure", "1")
        run_opt.add_run_config_entry("azure.auth_key", "hRflo7KIj1DoOdfLw5R8PphBiMBOY4C8")

        x = np.array([1, 2, 3, 4]).astype(np.int32)
        y = np.array([4, 3, 2, 1]).astype(np.int32)

        z = sess.run(None, {"X": x, "Y": y}, run_opt)[0]

        expected_z = np.array([5, 5, 5, 5]).astype(np.int32)
        np.testing.assert_allclose(z, expected_z, rtol=1e-05, atol=1e-08)

    # test an endpoint of "And"
    def test_and(self):
        sess_opt = ort.SessionOptions()
        sess_opt.add_session_config_entry("azure.endpoint_type", "triton")
        sess_opt.add_session_config_entry("azure.uri", "https://endpoint-6811.westus2.inference.ml.azure.com")
        sess_opt.add_session_config_entry("azure.model_name", "and")
        sess_opt.add_session_config_entry("azure.model_version", "1")
        sess_opt.add_session_config_entry("azure.verbose", "true")

        sess = ort.InferenceSession(get_name("azure_models/and.onnx"), sess_opt, providers=["CPUExecutionProvider"])

        run_opt = ort.RunOptions()
        run_opt.log_severity_level = 1
        run_opt.add_run_config_entry("use_azure", "1")
        run_opt.add_run_config_entry("azure.auth_key", "fdCZuuoHEimRb4ukWZhtLhbcwzyKYgUu")

        x = np.array([True, False]).astype(bool)
        y = np.array([True, True]).astype(bool)

        z = sess.run(None, {"X": x, "Y": y}, run_opt)[0]

        expected_z = np.array([True, False]).astype(bool)
        np.testing.assert_allclose(z, expected_z, rtol=1e-05, atol=1e-08)


if __name__ == "__main__":
    unittest.main()
