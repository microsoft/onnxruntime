# Copyright (c) NVIDIA Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import sys
import unittest
from collections.abc import Sequence

import numpy as np
import torch
from autoep_helper import AutoEpTestCase
from helper import get_name
from numpy.testing import assert_almost_equal
from onnx import TensorProto, helper
from onnx.defs import onnx_opset_version

import onnxruntime as onnxrt
from onnxruntime.capi._pybind_state import OrtDevice as C_OrtDevice
from onnxruntime.capi._pybind_state import OrtValue as C_OrtValue
from onnxruntime.capi._pybind_state import OrtValueVector, SessionIOBinding


class TestNvTensorRTRTXAutoEP(AutoEpTestCase):
    """
    Test suite for the NvTensorRTRTX Execution Provider.

    This class contains tests for registering the NvTensorRTRTX EP,
    selecting it using different policies, and running inference with various
    I/O binding configurations.
    """

    ep_lib_path = "onnxruntime_providers_nv_tensorrt_rtx.dll"
    ep_name = "NvTensorRTRTXExecutionProvider"

    def setUp(self):
        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")
        self.register_execution_provider_library(self.ep_name, self.ep_lib_path)

    def tearDown(self):
        self.unregister_execution_provider_library(self.ep_name)

    def _create_ortvalue_input_on_gpu(self, device):
        return onnxrt.OrtValue.ortvalue_from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32), device, 0
        )

    def _create_ortvalue_alternate_input_on_gpu(self, device):
        return onnxrt.OrtValue.ortvalue_from_numpy(
            np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], dtype=np.float32),
            device,
            0,
        )

    def _create_uninitialized_ortvalue_input_on_gpu(self, device):
        return onnxrt.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, device, 0)

    def _create_numpy_input(self):
        return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    def _create_expected_output(self):
        return np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)

    def _create_expected_output_alternate(self):
        return np.array([[2.0, 8.0], [18.0, 32.0], [50.0, 72.0]], dtype=np.float32)

    def torch_to_onnx_type(self, torch_dtype):
        if torch_dtype == torch.float32:
            return TensorProto.FLOAT
        elif torch_dtype == torch.float16:
            return TensorProto.FLOAT16
        elif torch_dtype == torch.bfloat16:
            return TensorProto.BFLOAT16
        elif torch_dtype == torch.int8:
            return TensorProto.int8
        elif torch_dtype == torch.int32:
            return TensorProto.INT32
        elif torch_dtype == torch.int64:
            return TensorProto.INT64
        else:
            raise TypeError(f"Unsupported dtype: {torch_dtype}")

    def test_nv_tensorrt_rtx_ep_register_and_inference(self):
        """
        Test registration of NvTensorRTRTX EP, adding its OrtDevice to the SessionOptions, and running inference.
        """
        ep_devices = onnxrt.get_ep_devices()
        nv_tensorrt_rtx_ep_device = next((d for d in ep_devices if d.ep_name == self.ep_name), None)
        self.assertIsNotNone(nv_tensorrt_rtx_ep_device)
        self.assertEqual(nv_tensorrt_rtx_ep_device.ep_vendor, "NVIDIA")

        hw_device = nv_tensorrt_rtx_ep_device.device
        self.assertEqual(hw_device.type, onnxrt.OrtHardwareDeviceType.GPU)

        # Run sample model and check output
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        del sess  # Delete session before unregistering library

    def test_nv_tensorrt_rtx_ep_prefer_gpu_and_inference(self):
        """
        Test selecting NvTensorRTRTX EP via the PREFER_GPU policy and running inference.
        """
        # Set a policy to prefer GPU. NvTensorRTRTX should be selected.
        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_GPU)
        self.assertTrue(sess_options.has_providers())

        # Run sample model and check output
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        del sess  # Delete session before unregistering library

    def test_nv_tensorrt_rtx_ep_selection_delegate_and_inference(self):
        """
        Test selecting NvTensorRTRTX EP via the custom EP selection delegate function and then run inference.
        """

        # User's custom EP selection function.
        def my_delegate(
            ep_devices: Sequence[onnxrt.OrtEpDevice],
            model_metadata: dict[str, str],
            runtime_metadata: dict[str, str],
            max_selections: int,
        ) -> Sequence[onnxrt.OrtEpDevice]:
            self.assertGreater(len(model_metadata), 0)
            self.assertGreaterEqual(len(ep_devices), 1)
            self.assertGreaterEqual(max_selections, 2)

            nv_tensorrt_rtx_ep_device = next((d for d in ep_devices if d.ep_name == self.ep_name), None)
            self.assertIsNotNone(nv_tensorrt_rtx_ep_device)

            # Select the NvTensorRTRTX device
            return [nv_tensorrt_rtx_ep_device]

        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy_delegate(my_delegate)
        self.assertTrue(sess_options.has_providers())

        # Run sample model and check output
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        del sess  # Delete session before unregistering library

    def test_bind_input_only(self):
        """
        Test I/O binding with input data only.
        """
        # Set a policy to prefer GPU. NvTensorRTRTX should be selected.
        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_GPU)
        self.assertTrue(sess_options.has_providers())

        input = self._create_ortvalue_input_on_gpu("cuda")

        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)
        io_binding = session.io_binding()

        # Bind input to the GPU
        io_binding.bind_input("X", "cuda", 0, np.float32, [3, 2], input.data_ptr())

        # Sync if different streams
        io_binding.synchronize_inputs()

        # Bind output to CPU
        io_binding.bind_output("Y")

        # Invoke Run
        session.run_with_iobinding(io_binding)

        # Sync if different streams
        io_binding.synchronize_outputs()

        # Get outputs over to CPU (the outputs which were bound to the GPU will get copied over to the host
        # here)
        ort_output = io_binding.copy_outputs_to_cpu()[0]

        # Validate results
        self.assertTrue(np.array_equal(self._create_expected_output(), ort_output))

        del session

    def test_bind_input_and_bind_output_with_ortvalues(self):
        """
        Test I/O binding with OrtValues for both input and output.
        """
        # Set a policy to prefer GPU. NvTensorRTRTX EP should be selected.
        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_GPU)
        self.assertTrue(sess_options.has_providers())

        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)
        io_binding = session.io_binding()

        # Bind ortvalue as input
        input_ortvalue = self._create_ortvalue_input_on_gpu("cuda")
        io_binding.bind_ortvalue_input("X", input_ortvalue)

        # Bind ortvalue as output
        output_ortvalue = self._create_uninitialized_ortvalue_input_on_gpu("cuda")
        io_binding.bind_ortvalue_output("Y", output_ortvalue)

        # Sync if different streams
        io_binding.synchronize_inputs()

        # Invoke Run
        session.run_with_iobinding(io_binding)

        # Sync if different streams
        io_binding.synchronize_outputs()

        # Inspect contents of output_ortvalue and make sure that it has the right contents
        self.assertTrue(np.array_equal(self._create_expected_output(), output_ortvalue.numpy()))

        # Bind another ortvalue as input
        input_ortvalue_2 = self._create_ortvalue_alternate_input_on_gpu("cuda")
        io_binding.bind_ortvalue_input("X", input_ortvalue_2)

        # Sync if different streams
        io_binding.synchronize_inputs()

        # Invoke Run
        session.run_with_iobinding(io_binding)

        # Sync if different streams
        io_binding.synchronize_outputs()

        # Inspect contents of output_ortvalue and make sure that it has the right contents
        self.assertTrue(np.array_equal(self._create_expected_output_alternate(), output_ortvalue.numpy()))

    def test_bind_input_and_non_preallocated_output(self):
        """
        Test I/O binding with non-preallocated output.
        """
        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_GPU)
        self.assertTrue(sess_options.has_providers())

        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)
        io_binding = session.io_binding()

        input = self._create_ortvalue_input_on_gpu("cuda")

        # Bind input to the GPU
        io_binding.bind_input("X", "cuda", 0, np.float32, [3, 2], input.data_ptr())

        # Bind output to the GPU
        io_binding.bind_output("Y", "cuda")

        # Sync if different streams
        io_binding.synchronize_inputs()

        # Invoke Run
        session.run_with_iobinding(io_binding)

        # Sync if different streams
        io_binding.synchronize_outputs()

        # This call returns an OrtValue which has data allocated by ORT on the GPU
        ort_outputs = io_binding.get_outputs()
        self.assertEqual(len(ort_outputs), 1)
        self.assertEqual(ort_outputs[0].device_name(), "cuda")
        # Validate results (by copying results to CPU by creating a Numpy object)
        self.assertTrue(np.array_equal(self._create_expected_output(), ort_outputs[0].numpy()))

        # We should be able to repeat the above process as many times as we want - try once more
        ort_outputs = io_binding.get_outputs()
        self.assertEqual(len(ort_outputs), 1)
        self.assertEqual(ort_outputs[0].device_name(), "cuda")
        # Validate results (by copying results to CPU by creating a Numpy object)
        self.assertTrue(np.array_equal(self._create_expected_output(), ort_outputs[0].numpy()))

        input = self._create_ortvalue_alternate_input_on_gpu("cuda")

        # Change the bound input and validate the results in the same bound OrtValue
        # Bind alternate input to the GPU
        io_binding.bind_input(
            "X",
            "cuda",
            0,
            np.float32,
            [3, 2],
            input.data_ptr(),
        )

        # Sync if different streams
        io_binding.synchronize_inputs()

        # Invoke Run
        session.run_with_iobinding(io_binding)

        # Sync if different streams
        io_binding.synchronize_outputs()

        # This call returns an OrtValue which has data allocated by ORT on the GPU
        ort_outputs = io_binding.get_outputs()
        self.assertEqual(len(ort_outputs), 1)
        self.assertEqual(ort_outputs[0].device_name(), "cuda")
        # Validate results (by copying results to CPU by creating a Numpy object)
        self.assertTrue(np.array_equal(self._create_expected_output_alternate(), ort_outputs[0].numpy()))

    def test_bind_input_and_preallocated_output(self):
        """
        Test I/O binding with preallocated output.
        """
        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_GPU)
        self.assertTrue(sess_options.has_providers())

        input = self._create_ortvalue_input_on_gpu("cuda")

        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)
        io_binding = session.io_binding()

        # Bind input to the GPU
        io_binding.bind_input("X", "cuda", 0, np.float32, [3, 2], input.data_ptr())

        # Bind output to the GPU
        output = self._create_uninitialized_ortvalue_input_on_gpu("cuda")
        io_binding.bind_output("Y", "cuda", 0, np.float32, [3, 2], output.data_ptr())

        # Sync if different streams
        io_binding.synchronize_inputs()

        # Invoke Run
        session.run_with_iobinding(io_binding)

        # Sync if different streams
        io_binding.synchronize_outputs()

        # Get outputs over to CPU (the outputs which were bound to the GPU will get copied over to the host
        # here)
        ort_output_vals = io_binding.copy_outputs_to_cpu()[0]
        # Validate results
        self.assertTrue(np.array_equal(self._create_expected_output(), ort_output_vals))

        # Validate if ORT actually wrote to pre-allocated buffer by copying the allocated buffer
        # to the host and validating its contents
        ort_output_vals_in_cpu = output.numpy()
        # Validate results
        self.assertTrue(np.array_equal(self._create_expected_output(), ort_output_vals_in_cpu))

    def test_bind_input_types(self):
        """
        Test I/O binding with various input data types.
        """
        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_GPU)
        self.assertTrue(sess_options.has_providers())
        opset = onnx_opset_version()
        device = C_OrtDevice(C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)

        for dtype in [
            np.float32,
            # np.float64,
            np.int32,
            # np.uint32,
            np.int64,
            # np.uint64,
            # np.int16,
            # np.uint16,
            # np.int8,
            np.uint8,
            np.float16,
            np.bool_,
        ]:
            with self.subTest(dtype=dtype, inner_device=str(device)):
                x = np.arange(8).reshape((-1, 2)).astype(dtype)
                proto_dtype = helper.np_dtype_to_tensor_dtype(x.dtype)

                X = helper.make_tensor_value_info("X", proto_dtype, [None, x.shape[1]])  # noqa: N806
                Y = helper.make_tensor_value_info("Y", proto_dtype, [None, x.shape[1]])  # noqa: N806

                # inference
                node_add = helper.make_node("Identity", ["X"], ["Y"])

                # graph
                graph_def = helper.make_graph([node_add], "lr", [X], [Y], [])
                model_def = helper.make_model(
                    graph_def,
                    producer_name="dummy",
                    ir_version=7,
                    producer_version="0",
                    opset_imports=[helper.make_operatorsetid("", opset)],
                )

                sess = onnxrt.InferenceSession(model_def.SerializeToString(), sess_options=sess_options)

                bind = SessionIOBinding(sess._sess)
                ort_value = C_OrtValue.ortvalue_from_numpy(x, device)
                bind.bind_ortvalue_input("X", ort_value)
                bind.bind_output("Y", device)
                sess._sess.run_with_iobinding(bind, None)
                ortvaluevector = bind.get_outputs()
                self.assertIsInstance(ortvaluevector, OrtValueVector)
                ortvalue = bind.get_outputs()[0]
                y = ortvalue.numpy()
                assert_almost_equal(x, y)

                bind = SessionIOBinding(sess._sess)
                bind.bind_input("X", device, dtype, x.shape, ort_value.data_ptr())
                bind.bind_output("Y", device)
                sess._sess.run_with_iobinding(bind, None)
                ortvalue = bind.get_outputs()[0]
                y = ortvalue.numpy()
                assert_almost_equal(x, y)

    def test_bind_onnx_types_from_torch(self):
        """
        Test I/O binding with various input data types.
        """
        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_GPU)
        self.assertTrue(sess_options.has_providers())
        opset = onnx_opset_version()

        for dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
        ]:
            with self.subTest(dtype=dtype):
                proto_dtype = self.torch_to_onnx_type(dtype)

                x_ = helper.make_tensor_value_info("X", proto_dtype, [None])
                y_ = helper.make_tensor_value_info("Y", proto_dtype, [None])
                node_add = helper.make_node("Identity", ["X"], ["Y"])
                graph_def = helper.make_graph([node_add], "lr", [x_], [y_], [])
                model_def = helper.make_model(
                    graph_def,
                    producer_name="dummy",
                    ir_version=10,
                    producer_version="0",
                    opset_imports=[helper.make_operatorsetid("", opset)],
                )
                sess = onnxrt.InferenceSession(model_def.SerializeToString(), sess_options=sess_options)

                dev = "cuda" if torch.cuda.is_available() else "cpu"
                device = (
                    C_OrtDevice(C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)
                    if dev == "cuda"
                    else C_OrtDevice(C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
                )

                x = torch.arange(8, dtype=dtype, device=dev)
                y = torch.empty(8, dtype=dtype, device=dev)

                bind = SessionIOBinding(sess._sess)
                bind.bind_input("X", device, proto_dtype, x.shape, x.data_ptr())
                bind.bind_output("Y", device, proto_dtype, y.shape, y.data_ptr())
                sess._sess.run_with_iobinding(bind, None)
                self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main(verbosity=1)
