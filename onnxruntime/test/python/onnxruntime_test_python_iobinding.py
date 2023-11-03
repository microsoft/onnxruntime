# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0115,W0212,C0103,C0114

import unittest

import numpy as np
from helper import get_name
from numpy.testing import assert_almost_equal
from onnx import helper
from onnx.defs import onnx_opset_version
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

import onnxruntime as onnxrt
from onnxruntime.capi._pybind_state import OrtDevice as C_OrtDevice  # pylint: disable=E0611
from onnxruntime.capi._pybind_state import OrtValue as C_OrtValue
from onnxruntime.capi._pybind_state import OrtValueVector, SessionIOBinding

test_params = [
    ("cuda", "CUDAExecutionProvider", C_OrtDevice.cuda),
    ("dml", "DmlExecutionProvider", C_OrtDevice.dml),
]


class TestIOBinding(unittest.TestCase):
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

    def test_bind_input_to_cpu_arr(self):
        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=onnxrt.get_available_providers())
        io_binding = session.io_binding()

        # Bind Numpy object (input) that's on CPU to wherever the model needs it
        io_binding.bind_cpu_input("X", self._create_numpy_input())

        # Bind output to CPU
        io_binding.bind_output("Y")

        # Invoke Run
        session.run_with_iobinding(io_binding)

        # Sync if different streams
        io_binding.synchronize_outputs()

        # Get outputs over to CPU (the outputs which were bound to the GPU will get copied over to the host here)
        ort_output = io_binding.copy_outputs_to_cpu()[0]

        # Validate results
        self.assertTrue(np.array_equal(self._create_expected_output(), ort_output))

    def test_bind_input_types(self):
        for device, execution_provider, generate_device in test_params:
            with self.subTest(execution_provider):
                if execution_provider not in onnxrt.get_available_providers():
                    self.skipTest(f"Skipping on {device.upper()}.")

                opset = onnx_opset_version()
                devices = [
                    (
                        C_OrtDevice(C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0),
                        ["CPUExecutionProvider"],
                    ),
                    (
                        C_OrtDevice(generate_device(), C_OrtDevice.default_memory(), 0),
                        [execution_provider],
                    ),
                ]

                for inner_device, provider in devices:
                    for dtype in [
                        np.float32,
                        np.float64,
                        np.int32,
                        np.uint32,
                        np.int64,
                        np.uint64,
                        np.int16,
                        np.uint16,
                        np.int8,
                        np.uint8,
                        np.float16,
                        np.bool_,
                    ]:
                        with self.subTest(dtype=dtype, inner_device=str(inner_device)):
                            x = np.arange(8).reshape((-1, 2)).astype(dtype)
                            proto_dtype = NP_TYPE_TO_TENSOR_TYPE[x.dtype]

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

                            sess = onnxrt.InferenceSession(model_def.SerializeToString(), providers=provider)

                            bind = SessionIOBinding(sess._sess)
                            ort_value = C_OrtValue.ortvalue_from_numpy(x, inner_device)
                            bind.bind_ortvalue_input("X", ort_value)
                            bind.bind_output("Y", inner_device)
                            sess._sess.run_with_iobinding(bind, None)
                            ortvaluevector = bind.get_outputs()
                            self.assertIsInstance(ortvaluevector, OrtValueVector)
                            ortvalue = bind.get_outputs()[0]
                            y = ortvalue.numpy()
                            assert_almost_equal(x, y)

                            bind = SessionIOBinding(sess._sess)
                            bind.bind_input("X", inner_device, dtype, x.shape, ort_value.data_ptr())
                            bind.bind_output("Y", inner_device)
                            sess._sess.run_with_iobinding(bind, None)
                            ortvalue = bind.get_outputs()[0]
                            y = ortvalue.numpy()
                            assert_almost_equal(x, y)

    def test_bind_input_only(self):
        for device, execution_provider, _ in test_params:
            with self.subTest(execution_provider):
                if execution_provider not in onnxrt.get_available_providers():
                    self.skipTest(f"Skipping on {device.upper()}.")
                input = self._create_ortvalue_input_on_gpu(device)

                session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=onnxrt.get_available_providers())
                io_binding = session.io_binding()

                # Bind input to the GPU
                io_binding.bind_input("X", device, 0, np.float32, [3, 2], input.data_ptr())

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

    def test_bind_input_and_preallocated_output(self):
        for device, execution_provider, _ in test_params:
            with self.subTest(execution_provider):
                if execution_provider not in onnxrt.get_available_providers():
                    self.skipTest(f"Skipping on {device.upper()}.")

                input = self._create_ortvalue_input_on_gpu(device)

                session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=onnxrt.get_available_providers())
                io_binding = session.io_binding()

                # Bind input to the GPU
                io_binding.bind_input("X", device, 0, np.float32, [3, 2], input.data_ptr())

                # Bind output to the GPU
                output = self._create_uninitialized_ortvalue_input_on_gpu(device)
                io_binding.bind_output("Y", device, 0, np.float32, [3, 2], output.data_ptr())

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

    def test_bind_input_and_non_preallocated_output(self):
        for device, execution_provider, _ in test_params:
            with self.subTest(execution_provider):
                if execution_provider not in onnxrt.get_available_providers():
                    self.skipTest(f"Skipping on {device.upper()}.")

                session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=onnxrt.get_available_providers())
                io_binding = session.io_binding()

                input = self._create_ortvalue_input_on_gpu(device)

                # Bind input to the GPU
                io_binding.bind_input(
                    "X",
                    device,
                    0,
                    np.float32,
                    [3, 2],
                    input.data_ptr(),
                )

                # Bind output to the GPU
                io_binding.bind_output("Y", device)

                # Sync if different streams
                io_binding.synchronize_inputs()

                # Invoke Run
                session.run_with_iobinding(io_binding)

                # Sync if different streams
                io_binding.synchronize_outputs()

                # This call returns an OrtValue which has data allocated by ORT on the GPU
                ort_outputs = io_binding.get_outputs()
                self.assertEqual(len(ort_outputs), 1)
                self.assertEqual(ort_outputs[0].device_name(), device)
                # Validate results (by copying results to CPU by creating a Numpy object)
                self.assertTrue(np.array_equal(self._create_expected_output(), ort_outputs[0].numpy()))

                # We should be able to repeat the above process as many times as we want - try once more
                ort_outputs = io_binding.get_outputs()
                self.assertEqual(len(ort_outputs), 1)
                self.assertEqual(ort_outputs[0].device_name(), device)
                # Validate results (by copying results to CPU by creating a Numpy object)
                self.assertTrue(np.array_equal(self._create_expected_output(), ort_outputs[0].numpy()))

                input = self._create_ortvalue_alternate_input_on_gpu(device)

                # Change the bound input and validate the results in the same bound OrtValue
                # Bind alternate input to the GPU
                io_binding.bind_input(
                    "X",
                    device,
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
                self.assertEqual(ort_outputs[0].device_name(), device)
                # Validate results (by copying results to CPU by creating a Numpy object)
                self.assertTrue(np.array_equal(self._create_expected_output_alternate(), ort_outputs[0].numpy()))

    def test_bind_input_and_bind_output_with_ortvalues(self):
        for device, execution_provider, _ in test_params:
            with self.subTest(execution_provider):
                if execution_provider not in onnxrt.get_available_providers():
                    self.skipTest(f"Skipping on {device.upper()}.")

                session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=onnxrt.get_available_providers())
                io_binding = session.io_binding()

                # Bind ortvalue as input
                input_ortvalue = self._create_ortvalue_input_on_gpu(device)
                io_binding.bind_ortvalue_input("X", input_ortvalue)

                # Bind ortvalue as output
                output_ortvalue = self._create_uninitialized_ortvalue_input_on_gpu(device)
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
                input_ortvalue_2 = self._create_ortvalue_alternate_input_on_gpu(device)
                io_binding.bind_ortvalue_input("X", input_ortvalue_2)

                # Sync if different streams
                io_binding.synchronize_inputs()

                # Invoke Run
                session.run_with_iobinding(io_binding)

                # Sync if different streams
                io_binding.synchronize_outputs()

                # Inspect contents of output_ortvalue and make sure that it has the right contents
                self.assertTrue(np.array_equal(self._create_expected_output_alternate(), output_ortvalue.numpy()))


if __name__ == "__main__":
    unittest.main()
