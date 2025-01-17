# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
from typing import Dict, List

import numpy as np
from helper import get_name

import onnxruntime as onnxrt


class CudaGraphHelper:
    def __init__(
        self,
        ort_session: onnxrt.InferenceSession,
        input_and_output_shape: Dict[str, List[int]],
        device_id: int = 0,
    ):
        self.input_names = [input.name for input in ort_session.get_inputs()]
        self.output_names = [output.name for output in ort_session.get_outputs()]

        self.input_and_output_shape = input_and_output_shape
        self.io_numpy_type = self.get_io_numpy_type_map(ort_session)
        self.io_binding = ort_session.io_binding()
        self.io_ort_value = {}

        for name in self.input_names + self.output_names:
            ort_value = onnxrt.OrtValue.ortvalue_from_shape_and_type(
                input_and_output_shape[name], self.io_numpy_type[name], "cuda", device_id
            )
            self.io_ort_value[name] = ort_value
            if name in self.input_names:
                self.io_binding.bind_ortvalue_input(name, ort_value)
            else:
                self.io_binding.bind_ortvalue_output(name, ort_value)

    def get_io_numpy_type_map(self, ort_session: onnxrt.InferenceSession):
        ort_type_to_numpy_type = {
            "tensor(int64)": np.longlong,
            "tensor(int32)": np.intc,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
        }

        name_to_numpy_type = {}
        for _input in ort_session.get_inputs():
            name_to_numpy_type[_input.name] = ort_type_to_numpy_type[_input.type]

        for output in ort_session.get_outputs():
            name_to_numpy_type[output.name] = ort_type_to_numpy_type[output.type]

        return name_to_numpy_type

    def update_inputs(self, inputs: Dict[str, np.ndarray]):
        for input_name in self.input_names:
            self.io_ort_value[input_name].update_inplace(inputs[input_name])

    def get_output(self, output_name: str):
        return self.io_ort_value[output_name].numpy()


class TestInferenceSessionWithCudaGraph(unittest.TestCase):
    def test_ort_value_update_in_place(self):
        x0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        ortvalue_cpu = onnxrt.OrtValue.ortvalue_from_numpy(x0)
        np.testing.assert_allclose(x0, ortvalue_cpu.numpy())

        x1 = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        ortvalue_cpu.update_inplace(x1)
        np.testing.assert_allclose(x1, ortvalue_cpu.numpy())

        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            ortvalue_gpu = onnxrt.OrtValue.ortvalue_from_numpy(x0, "cuda", 0)
            np.testing.assert_allclose(x0, ortvalue_gpu.numpy())

            ortvalue_gpu.update_inplace(x1)
            np.testing.assert_allclose(x1, ortvalue_gpu.numpy())

    def test_select_ep_to_run_cuda_graph(self):
        if "TensorrtExecutionProvider" in onnxrt.get_available_providers():
            providers = [("TensorrtExecutionProvider", {"trt_cuda_graph_enable": True})]
            self.run_model_with_cuda_graph(providers)
        elif "CUDAExecutionProvider" in onnxrt.get_available_providers():
            providers = [("CUDAExecutionProvider", {"enable_cuda_graph": True})]
            self.run_model_with_cuda_graph(providers)
            self.run_model_with_cuda_graph_annotation(providers)

    def run_model_with_cuda_graph(self, providers):
        INPUT_SIZE = 1280  # noqa: N806
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] * INPUT_SIZE, dtype=np.float32)
        y = np.array([[0.0], [0.0], [0.0]] * INPUT_SIZE, dtype=np.float32)
        x_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
        y_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(y, "cuda", 0)

        onnxrt.set_default_logger_severity(0)
        session = onnxrt.InferenceSession(get_name("matmul_2.onnx"), providers=providers)
        io_binding = session.io_binding()

        # Bind the input and output
        io_binding.bind_ortvalue_input("X", x_ortvalue)
        io_binding.bind_ortvalue_output("Y", y_ortvalue)

        ro = onnxrt.RunOptions()

        # One regular run for the necessary memory allocation and cuda graph capturing
        session.run_with_iobinding(io_binding, ro)
        expected_y = np.array([[5.0], [11.0], [17.0]] * INPUT_SIZE, dtype=np.float32)
        np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

        # After capturing, CUDA graph replay happens from this Run onwards
        session.run_with_iobinding(io_binding, ro)
        np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

        # Update input and then replay CUDA graph
        x_ortvalue.update_inplace(
            np.array(
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]] * INPUT_SIZE,
                dtype=np.float32,
            )
        )
        session.run_with_iobinding(io_binding, ro)
        np.testing.assert_allclose(
            np.array([[50.0], [110.0], [170.0]] * INPUT_SIZE, dtype=np.float32),
            y_ortvalue.numpy(),
            rtol=1e-05,
            atol=1e-05,
        )

    def run_model_with_cuda_graph_annotation(self, providers):
        INPUT_SIZE = 1280  # noqa: N806

        x_base = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        y_base = [[0.0], [0.0], [0.0], [0.0]]
        expected_y_base = [[5.0], [11.0], [17.0], [23.0]]

        x_base_mul_10 = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]
        expected_y_base_mul_10 = [[50.0], [110.0], [170.0], [230.0]]

        test_num = 4

        x_ortvalues = []
        y_ortvalues = []
        for i in range(test_num):
            x = np.array(x_base[: i + 1][:] * INPUT_SIZE, dtype=np.float32)
            y = np.array(y_base[: i + 1][:] * INPUT_SIZE, dtype=np.float32)
            x_ortvalues.append(onnxrt.OrtValue.ortvalue_from_numpy(x, "cuda", 0))
            y_ortvalues.append(onnxrt.OrtValue.ortvalue_from_numpy(y, "cuda", 0))

        onnxrt.set_default_logger_severity(0)
        session = onnxrt.InferenceSession(get_name("matmul_2.onnx"), providers=providers)
        io_bindings = [session.io_binding()] * test_num
        ro = onnxrt.RunOptions()

        # Regular run to capture CUDA graph
        for i in range(test_num):
            io_bindings[i].bind_ortvalue_input("X", x_ortvalues[i])
            io_bindings[i].bind_ortvalue_output("Y", y_ortvalues[i])
            # TODO: Temporarily remove the default cuda graph capture test for the first regular run
            # because it fails on a training CI. Need to investigate the root cause.
            ro.add_run_config_entry("gpu_graph_id", str(i + 1))
            io_bindings[i].synchronize_inputs()
            session.run_with_iobinding(io_bindings[i], ro)
            io_bindings[i].synchronize_outputs()
            expected_y = np.array(expected_y_base[: i + 1][:] * INPUT_SIZE, dtype=np.float32)
            np.testing.assert_allclose(expected_y, y_ortvalues[i].numpy(), rtol=1e-05, atol=1e-05)

        del ro
        ro = onnxrt.RunOptions()

        # After capturing, CUDA graph replay happens from this Run onwards
        for i in range(test_num):
            # Update input and then replay CUDA graph
            x_ortvalues[i].update_inplace(np.array(x_base_mul_10[: i + 1][:] * INPUT_SIZE, dtype=np.float32))
            ro.add_run_config_entry("gpu_graph_id", str(i + 1))
            io_bindings[i].synchronize_inputs()
            session.run_with_iobinding(io_bindings[i], ro)
            io_bindings[i].synchronize_outputs()
            expected_y = np.array(expected_y_base_mul_10[: i + 1][:] * INPUT_SIZE, dtype=np.float32)
            np.testing.assert_allclose(expected_y, y_ortvalues[i].numpy(), rtol=1e-05, atol=1e-05)

    def test_arena_with_cuda_graph(self):
        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            # To test cuda graph catpure, we set Arena extend strategy to be SameAsRequested so as to detect any
            # potential memory allocation after the first run.
            providers = [
                ("CUDAExecutionProvider", {"enable_cuda_graph": True, "arena_extend_strategy": "kSameAsRequested"})
            ]
            test_model_path = get_name("squeezenet/model.onnx")

            input_and_output_shape = {
                "data_0": [16, 3, 224, 224],
                "softmaxout_1": [16, 1000, 1, 1],
            }

            session_options = onnxrt.SessionOptions()
            # It is optional to disable memory pattern since min_num_runs_before_cuda_graph_capture_ = 2.
            session_options.enable_mem_pattern = False
            session = onnxrt.InferenceSession(test_model_path, session_options, providers=providers)

            cuda_graph_helper = CudaGraphHelper(session, input_and_output_shape)
            io_binding = cuda_graph_helper.io_binding

            # Create a random input for testing.
            np.random.seed(0)
            inputs = {"data_0": np.random.randint(0, 256, size=input_and_output_shape["data_0"]).astype(np.float32)}

            # One regular run for the necessary memory allocation and cuda graph capturing
            cuda_graph_helper.update_inputs(inputs)
            session.run_with_iobinding(io_binding)
            expected_output = cuda_graph_helper.get_output("softmaxout_1")

            # After capturing, CUDA graph replay happens from this Run onwards
            cuda_graph_helper.update_inputs(inputs)
            session.run_with_iobinding(io_binding)
            output = cuda_graph_helper.get_output("softmaxout_1")

            np.testing.assert_allclose(expected_output, output, rtol=1e-02, atol=1e-02)


if __name__ == "__main__":
    unittest.main()
