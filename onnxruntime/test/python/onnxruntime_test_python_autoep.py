# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import os
import platform
import sys
import tempfile
import unittest

from helper import get_name

import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())

available_providers = list(onnxrt.get_available_providers())


class TestAutoEP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.autoep_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def test_cuda_ep_devices(self):
        """
        Test registration of CUDA EP and retrieval of its OrtEpDevice.
        """
        ep_lib_path = "onnxruntime_providers_cuda.dll"
        ep_registration_name = "CUDAExecutionProvider"

        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        if not os.path.exists(ep_lib_path):
            self.skipTest(f"Skipping test because EP library '{ep_lib_path}' cannot be found")

        onnxrt.register_execution_provider_library(ep_registration_name, os.path.realpath(ep_lib_path))

        ep_devices = onnxrt.get_ep_devices()
        has_cpu_ep = False
        has_test_ep = False
        for ep_device in ep_devices:
            ep_name = ep_device.ep_name()
            if ep_name == "CPUExecutionProvider":
                has_cpu_ep = True
            if ep_name == ep_registration_name:
                has_test_ep = True

        self.assertTrue(has_cpu_ep)
        self.assertTrue(has_test_ep)
        onnxrt.unregister_execution_provider_library(ep_registration_name)

    def test_example_plugin_ep_devices(self):
        """
        Test registration of an example EP plugin and retrieval of its OrtEpDevice.
        """
        ep_lib_path = "example_plugin_ep.dll"
        ep_registration_name = "example_ep"

        if sys.platform != "win32":
            self.skipTest("Skipping test because it device discovery is only supported on Windows")

        if not os.path.exists(ep_lib_path):
            self.skipTest(f"Skipping test because EP library '{ep_lib_path}' cannot be found")

        onnxrt.register_execution_provider_library(ep_registration_name, os.path.realpath(ep_lib_path))

        ep_devices = onnxrt.get_ep_devices()
        has_cpu_ep = False
        test_ep_device = None
        for ep_device in ep_devices:
            ep_name = ep_device.ep_name()

            if ep_name == "CPUExecutionProvider":
                has_cpu_ep = True
            if ep_name == ep_registration_name:
                test_ep_device = ep_device

        self.assertTrue(has_cpu_ep)
        self.assertIsNotNone(test_ep_device)

        # Test the OrtEpDevice getters. Expected values are from /onnxruntime/test/autoep/library/example_plugin_ep.cc
        self.assertEqual(test_ep_device.ep_vendor(), "Contoso")

        ep_metadata = test_ep_device.ep_metadata()
        self.assertEqual(ep_metadata["version"], "0.1")

        ep_options = test_ep_device.ep_options()
        self.assertEqual(ep_options["run_really_fast"], "true")

        # The CPU hw device info will vary by machine so check for the common values.
        hw_device = test_ep_device.device()
        self.assertEqual(hw_device.type(), onnxrt.OrtHardwareDeviceType.CPU)
        self.assertGreaterEqual(hw_device.vendor_id(), 0)
        self.assertGreaterEqual(hw_device.device_id(), 0)
        self.assertGreater(len(hw_device.vendor()), 0)

        hw_metadata = hw_device.metadata()
        self.assertGreater(len(hw_metadata), 0)  # Should have at least SPDRP_HARDWAREID on Windows

        # Test creating an InferenceSession with this plugin EP.
        sess_options = onnxrt.SessionOptions()
        ep_devices_config = onnxrt.EpDevicesConfig(ep_devices=[test_ep_device], ep_options={"opt1": "val1"})
        input_model_path = get_name("mul_1.onnx")
        with self.assertRaises(InvalidArgument) as context:
            # Will raise InvalidArgument because ORT currently only supports provider bridge APIs.
            # Actual plugin EPs will be supported in the future.
            onnxrt.InferenceSession(
                input_model_path,
                sess_options=sess_options,
                ep_devices_config=ep_devices_config,
            )
        self.assertIn("EP is not currently supported", str(context.exception))

        onnxrt.unregister_execution_provider_library(ep_registration_name)


if __name__ == "__main__":
    unittest.main(verbosity=1)
