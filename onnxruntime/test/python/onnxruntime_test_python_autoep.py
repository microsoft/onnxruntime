# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import os
import platform
import sys
import unittest
from collections.abc import Sequence

import numpy as np
import onnx
from autoep_helper import AutoEpTestCase
from helper import get_name

import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())

available_providers = list(onnxrt.get_available_providers())


class TestAutoEP(AutoEpTestCase):
    EXAMPLE_EP_NAME = "example_ep"

    def test_cuda_ep_register_and_inference(self):
        """
        Test registration of CUDA EP, adding its OrtDevice to the SessionOptions, and running inference.
        """
        ep_lib_path = "onnxruntime_providers_cuda.dll"
        ep_name = "CUDAExecutionProvider"

        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        if ep_name not in available_providers:
            self.skipTest("Skipping test because it needs to run on CUDA EP")

        self.register_execution_provider_library(ep_name, ep_lib_path)

        ep_devices = onnxrt.get_ep_devices()
        has_cpu_ep = False
        cuda_ep_device = None
        for ep_device in ep_devices:
            if ep_device.ep_name == "CPUExecutionProvider":
                has_cpu_ep = True
            if ep_device.ep_name == ep_name:
                cuda_ep_device = ep_device

        self.assertTrue(has_cpu_ep)
        self.assertIsNotNone(cuda_ep_device)
        self.assertEqual(cuda_ep_device.ep_vendor, "Microsoft")

        hw_device = cuda_ep_device.device
        self.assertEqual(hw_device.type, onnxrt.OrtHardwareDeviceType.GPU)

        # Add CUDA's OrtEpDevice to session options
        sess_options = onnxrt.SessionOptions()
        sess_options.add_provider_for_devices([cuda_ep_device], {"prefer_nhwc": "1"})
        self.assertTrue(sess_options.has_providers())

        # Run sample model and check output
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

        del sess  # Delete session before unregistering library
        self.unregister_execution_provider_library(ep_name)

    def test_cuda_prefer_gpu_and_inference(self):
        """
        Test selecting CUDA EP via the PREFER_GPU policy and running inference.
        """
        ep_lib_path = "onnxruntime_providers_cuda.dll"
        ep_name = "CUDAExecutionProvider"

        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        if ep_name not in available_providers:
            self.skipTest("Skipping test because it needs to run on CUDA EP")

        self.register_execution_provider_library(ep_name, ep_lib_path)

        # Set a policy to prefer GPU. Cuda should be selected.
        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_GPU)
        self.assertTrue(sess_options.has_providers())

        # Run sample model and check output
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

        del sess  # Delete session before unregistering library
        self.unregister_execution_provider_library(ep_name)

    def test_cuda_ep_selection_delegate_and_inference(self):
        """
        Test selecting CUDA EP via the custom EP selection delegate function and then run inference.
        """
        ep_lib_path = "onnxruntime_providers_cuda.dll"
        ep_name = "CUDAExecutionProvider"

        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        if ep_name not in available_providers:
            self.skipTest("Skipping test because it needs to run on CUDA EP")

        self.register_execution_provider_library(ep_name, ep_lib_path)

        # User's custom EP selection function.
        def my_delegate(
            ep_devices: Sequence[onnxrt.OrtEpDevice],
            model_metadata: dict[str, str],
            runtime_metadata: dict[str, str],
            max_selections: int,
        ) -> Sequence[onnxrt.OrtEpDevice]:
            self.assertGreater(len(model_metadata), 0)
            self.assertGreaterEqual(len(ep_devices), 2)
            self.assertGreaterEqual(max_selections, 2)

            cuda_ep_device = next((d for d in ep_devices if d.ep_name == ep_name), None)
            self.assertIsNotNone(cuda_ep_device)

            # Select the CUDA device and the ORT CPU EP device (should always be last)
            return [cuda_ep_device, ep_devices[-1]]

        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy_delegate(my_delegate)
        self.assertTrue(sess_options.has_providers())

        # Run sample model and check output
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

        del sess  # Delete session before unregistering library
        self.unregister_execution_provider_library(ep_name)

    def test_custom_ep_selection_delegate_that_raises_error(self):
        """
        Test a custom EP selection delegate function that raises a Python exception. ORT should re-raise as FAIL.
        """
        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        # User's custom EP selection function.
        custom_error_message = "MY ERROR"

        def my_delegate_that_fails(
            ep_devices: Sequence[onnxrt.OrtEpDevice],
            model_metadata: dict[str, str],
            runtime_metadata: dict[str, str],
            max_selections: int,
        ) -> Sequence[onnxrt.OrtEpDevice]:
            self.assertGreaterEqual(len(ep_devices), 1)
            raise ValueError(custom_error_message)

        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy_delegate(my_delegate_that_fails)

        # Create session and expect ORT to raise a Fail exception that contains our message.
        with self.assertRaises(Fail) as context:
            onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)
        self.assertIn(custom_error_message, str(context.exception))

    def test_example_plugin_ep_devices(self):
        """
        Test registration of an example EP plugin and retrieval of its OrtEpDevice.
        """
        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        ep_lib_path = "example_plugin_ep.dll"
        try:
            ep_lib_path = get_name("example_plugin_ep.dll")
        except FileNotFoundError:
            self.skipTest(f"Skipping test because EP library '{ep_lib_path}' cannot be found")

        ep_name = "example_ep"
        self.register_execution_provider_library(ep_name, os.path.realpath(ep_lib_path))

        ep_devices = onnxrt.get_ep_devices()
        has_cpu_ep = False
        test_ep_device = None
        for ep_device in ep_devices:
            if ep_device.ep_name == "CPUExecutionProvider":
                has_cpu_ep = True
            if ep_device.ep_name == ep_name:
                test_ep_device = ep_device

        self.assertTrue(has_cpu_ep)
        self.assertIsNotNone(test_ep_device)

        # Test the OrtEpDevice getters. Expected values are from /onnxruntime/test/autoep/library/example_plugin_ep.cc
        self.assertEqual(test_ep_device.ep_vendor, "Contoso")

        ep_metadata = test_ep_device.ep_metadata
        self.assertEqual(ep_metadata["version"], "0.1.0")
        self.assertEqual(ep_metadata["supported_devices"], "CrackGriffin 7+")

        ep_options = test_ep_device.ep_options
        self.assertEqual(ep_options["run_really_fast"], "true")

        # The CPU hw device info will vary by machine so check for the common values.
        hw_device = test_ep_device.device
        self.assertEqual(hw_device.type, onnxrt.OrtHardwareDeviceType.CPU)
        self.assertGreaterEqual(hw_device.vendor_id, 0)
        self.assertGreaterEqual(hw_device.device_id, 0)
        self.assertGreater(len(hw_device.vendor), 0)

        hw_metadata = hw_device.metadata
        self.assertGreater(len(hw_metadata), 0)  # Should have at least SPDRP_HARDWAREID on Windows

        test_mem_info = test_ep_device.memory_info(onnxrt.OrtDeviceMemoryType.DEFAULT)
        self.assertIsNotNone(test_mem_info)
        del test_mem_info

        test_sync_stream = test_ep_device.create_sync_stream()
        self.assertIsNotNone(test_sync_stream)
        stream_handle = test_sync_stream.get_handle()
        self.assertIsNotNone(stream_handle)
        del test_sync_stream

        # Add EP plugin's OrtEpDevice to the SessionOptions.
        sess_options = onnxrt.SessionOptions()
        sess_options.add_provider_for_devices([test_ep_device], {"opt1": "val1"})
        sess_options.log_severity_level = 1  # INFO
        self.assertTrue(sess_options.has_providers())

        # Run sample model and check output
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

        del sess  # Delete session before unregistering library
        self.unregister_execution_provider_library(ep_name)

    def test_example_plugin_ep_data_transfer(self):
        """
        Test usage of shared data transfer and allocator from plugin EP.
        """
        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        if "DmlExecutionProvider" in onnxrt.get_available_providers():
            self.skipTest("Skipping because DML EP data transfer is broken if we haven't created an inference session")

        ep_lib_path = "example_plugin_ep.dll"
        try:
            ep_lib_path = get_name("example_plugin_ep.dll")
        except FileNotFoundError:
            self.skipTest(f"Skipping test because EP library '{ep_lib_path}' cannot be found")

        ep_name = "example_ep"
        self.register_execution_provider_library(ep_name, os.path.realpath(ep_lib_path))

        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        data2 = data + 1

        # the example EP pretends to use GPU memory so we can test data transfer.
        # by matching its OrtDevice info we will hit its allocator and data transfer implementations.
        # copy data from CPU to the fake GPU memory
        gpu_value = onnxrt.OrtValue.ortvalue_from_numpy(data, "gpu", 0, 0xBE57)
        # copy back to CPU
        cpu_data = gpu_value.numpy()
        np.testing.assert_equal(cpu_data, data)

        gpu_value.update_inplace(data2)  # update the fake GPU data
        cpu_data_2 = gpu_value.numpy()  # copy back to CPU
        np.testing.assert_equal(cpu_data_2, data2)

        gpu_value = None  # Delete OrtValue before unregistering library as the allocator will be destroyed.

        self.unregister_execution_provider_library(ep_name)

    def test_copy_tensors(self):
        """
        Test global api copy_tensors between OrtValue objects
        using EP plug-in data transfer
        """
        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        ep_lib_path = "example_plugin_ep.dll"
        try:
            ep_lib_path = get_name("example_plugin_ep.dll")
        except FileNotFoundError:
            self.skipTest(f"Skipping test because EP library '{ep_lib_path}' cannot be found")

        ep_name = "example_ep"
        self.register_execution_provider_library(ep_name, os.path.realpath(ep_lib_path))

        # Generate 2 numpy arrays
        a = np.random.rand(3, 2).astype(np.float32)
        b = np.random.rand(3, 2).astype(np.float32)

        # Create OrtValue from numpy arrays on EP device
        # the example EP pretends to use GPU memory, so we place it there
        a_device = onnxrt.OrtValue.ortvalue_from_numpy(a, "gpu", 0, 0xBE57)
        b_device = onnxrt.OrtValue.ortvalue_from_numpy(b, "gpu", 0, 0xBE57)

        # Create destination ort values with the same shape on CPU
        a_cpu_copy = onnxrt.OrtValue.ortvalue_from_shape_and_type(a.shape, a.dtype)
        b_cpu_copy = onnxrt.OrtValue.ortvalue_from_shape_and_type(b.shape, b.dtype)

        # source list
        src_list = [a_device, b_device]
        dst_list = [a_cpu_copy, b_cpu_copy]
        # Passing None for stream as we copy between CPU
        # Test None because it is allowed
        onnxrt.copy_tensors(src_list, dst_list, None)

        # Release the OrtValue on the EP device
        # before the EP library is unregistered
        del src_list
        del a_device
        del b_device

        # Verify the contents
        np.testing.assert_array_equal(a_cpu_copy.numpy(), a)
        np.testing.assert_array_equal(b_cpu_copy.numpy(), b)

        self.unregister_execution_provider_library(ep_name)

    def _register_example_plugin_ep_or_skip(self):
        """Register the example plugin EP and return its OrtEpDevice, or skip the test."""
        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        try:
            ep_lib_path = get_name("example_plugin_ep.dll")
        except FileNotFoundError:
            self.skipTest("Skipping test because example_plugin_ep.dll cannot be found")

        self.register_execution_provider_library(self.EXAMPLE_EP_NAME, os.path.realpath(ep_lib_path))

        ep_device = next(
            (d for d in onnxrt.get_ep_devices() if d.ep_name == self.EXAMPLE_EP_NAME),
            None,
        )
        self.assertIsNotNone(ep_device, f"Could not find OrtEpDevice for registered EP '{self.EXAMPLE_EP_NAME}'")
        return ep_device

    def test_ortvalue_from_shape_and_type_host_accessible_numpy_dtype(self):
        ep_device = self._register_example_plugin_ep_or_skip()
        mem_info = ep_device.memory_info(onnxrt.OrtDeviceMemoryType.HOST_ACCESSIBLE)
        self.assertIsNotNone(mem_info)

        ort_value = onnxrt.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, memory_info=mem_info)

        self.assertEqual(ort_value.shape(), [3, 2])
        self.assertEqual(ort_value.data_type(), "tensor(float)")
        # The example EP advertises HOST_ACCESSIBLE on a fake GPU device, so the allocator
        # came from memory_info rather than the default CPU path.
        self.assertNotEqual(ort_value.device_name().lower(), "cpu")

        result = ort_value.numpy()
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.dtype, np.float32)

        del ort_value
        self.unregister_execution_provider_library(self.EXAMPLE_EP_NAME)

    def test_ortvalue_from_shape_and_type_host_accessible_onnx_int_type(self):
        ep_device = self._register_example_plugin_ep_or_skip()
        mem_info = ep_device.memory_info(onnxrt.OrtDeviceMemoryType.HOST_ACCESSIBLE)

        ort_value = onnxrt.OrtValue.ortvalue_from_shape_and_type(
            [4], onnx.TensorProto.FLOAT, memory_info=mem_info
        )

        self.assertEqual(ort_value.shape(), [4])
        self.assertEqual(ort_value.data_type(), "tensor(float)")
        self.assertEqual(ort_value.numpy().dtype, np.float32)

        del ort_value
        self.unregister_execution_provider_library(self.EXAMPLE_EP_NAME)

    def test_ortvalue_host_accessible_zero_copy_numpy_view(self):
        # Writing through view1 must be visible through view2 - if numpy() ever copies,
        # this test fails and the UsesCpuMemory() zero-copy guarantee has regressed.
        ep_device = self._register_example_plugin_ep_or_skip()
        mem_info = ep_device.memory_info(onnxrt.OrtDeviceMemoryType.HOST_ACCESSIBLE)

        ort_value = onnxrt.OrtValue.ortvalue_from_shape_and_type([2, 3], np.float32, memory_info=mem_info)
        ort_value.numpy().fill(7.5)
        np.testing.assert_array_equal(ort_value.numpy(), np.full((2, 3), 7.5, dtype=np.float32))

        cpu_value = onnxrt.OrtValue.ortvalue_from_shape_and_type([2, 3], np.float32)
        cpu_value.numpy().fill(-1.25)
        np.testing.assert_array_equal(cpu_value.numpy(), np.full((2, 3), -1.25, dtype=np.float32))

        del ort_value
        del cpu_value
        self.unregister_execution_provider_library(self.EXAMPLE_EP_NAME)

    def test_ortvalue_from_shape_and_type_memory_info_no_allocator(self):
        bogus_mem_info = onnxrt.OrtMemoryInfo.create_v2(
            "Bogus",
            onnxrt.OrtMemoryInfoDeviceType.GPU,
            0xDEAD,
            0,
            onnxrt.OrtDeviceMemoryType.HOST_ACCESSIBLE,
            0,
            onnxrt.OrtAllocatorType.ORT_DEVICE_ALLOCATOR,
        )

        with self.assertRaisesRegex(RuntimeError, "No shared allocator found"):
            onnxrt.OrtValue.ortvalue_from_shape_and_type([2], np.float32, memory_info=bogus_mem_info)

    def test_ortvalue_from_shape_and_onnx_type_memory_info_string_rejected(self):
        ep_device = self._register_example_plugin_ep_or_skip()
        mem_info = ep_device.memory_info(onnxrt.OrtDeviceMemoryType.HOST_ACCESSIBLE)

        with self.assertRaisesRegex(RuntimeError, "non-string numpy arrays"):
            onnxrt.OrtValue.ortvalue_from_shape_and_type(
                [2], onnx.TensorProto.STRING, memory_info=mem_info
            )

        self.unregister_execution_provider_library(self.EXAMPLE_EP_NAME)

    def test_ortvalue_from_shape_and_type_memory_info_overrides_device_args(self):
        ep_device = self._register_example_plugin_ep_or_skip()
        mem_info = ep_device.memory_info(onnxrt.OrtDeviceMemoryType.HOST_ACCESSIBLE)

        # Bogus device args alongside a valid memory_info: if the wrapper ever stops ignoring
        # them, this would fail (unknown device) or silently allocate elsewhere.
        ort_value = onnxrt.OrtValue.ortvalue_from_shape_and_type(
            [3],
            np.float32,
            device_type="cuda",
            device_id=99,
            vendor_id=0xFFFF,
            memory_info=mem_info,
        )

        ort_value_baseline = onnxrt.OrtValue.ortvalue_from_shape_and_type([3], np.float32, memory_info=mem_info)
        self.assertEqual(ort_value.device_name(), ort_value_baseline.device_name())

        del ort_value
        del ort_value_baseline
        self.unregister_execution_provider_library(self.EXAMPLE_EP_NAME)

    def test_ortvalue_from_shape_and_type_default_memory_info(self):
        # Pins the false-branch of UsesCpuMemory(): DEFAULT memory on a non-CPU device must
        # round-trip through data_transfer rather than the zero-copy view path.
        ep_device = self._register_example_plugin_ep_or_skip()
        mem_info = ep_device.memory_info(onnxrt.OrtDeviceMemoryType.DEFAULT)
        self.assertIsNotNone(mem_info)

        ort_value = onnxrt.OrtValue.ortvalue_from_shape_and_type([2, 3], np.float32, memory_info=mem_info)

        self.assertEqual(ort_value.shape(), [2, 3])
        self.assertEqual(ort_value.data_type(), "tensor(float)")
        self.assertNotEqual(ort_value.device_name().lower(), "cpu")

        arr = ort_value.numpy()
        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr.dtype, np.float32)

        del ort_value
        self.unregister_execution_provider_library(self.EXAMPLE_EP_NAME)


if __name__ == "__main__":
    unittest.main(verbosity=1)
