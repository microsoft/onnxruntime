# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Focused WebGPU GEMV tests. Set ORT_WEBGPU_PLUGIN_PATH for a plugin build."""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx

import onnxruntime as ort


def webgpu_devices():
    plugin = os.environ.get("ORT_WEBGPU_PLUGIN_PATH")
    if plugin:
        if "WebGpuExecutionProvider" in ort.get_available_providers():
            raise RuntimeError("Use a core ORT build without built-in WebGPU when selecting a WebGPU plugin.")
        ort.register_execution_provider_library("webgpu_matmul_test", plugin)
    return [device for device in ort.get_ep_devices() if device.ep_name == "WebGpuExecutionProvider"]


def matmul_model(a_shape, weights, node_count=1, dynamic_b=False):
    dtype = onnx.TensorProto.FLOAT16 if weights[0].dtype == np.float16 else onnx.TensorProto.FLOAT
    inputs = [onnx.helper.make_tensor_value_info("A", dtype, a_shape)]
    initializers = []
    if dynamic_b:
        if len(weights) != 1 or node_count != 1:
            raise ValueError("Dynamic B is only supported for the single-node correctness cases.")
        inputs.append(onnx.helper.make_tensor_value_info("B0", dtype, weights[0].shape))
    else:
        initializers = [onnx.numpy_helper.from_array(weight, f"B{i}") for i, weight in enumerate(weights)]
    nodes = [
        onnx.helper.make_node("MatMul", ["A", f"B{i % len(weights)}"], [f"Y{i}"], name=f"projection_{i}")
        for i in range(node_count)
    ]
    # Shape inference handles vector promotion and broadcasted leading dimensions.
    outputs = [onnx.helper.make_tensor_value_info(f"Y{i}", dtype, None) for i in range(node_count)]
    model = onnx.helper.make_model(
        onnx.helper.make_graph(nodes, "webgpu_matmul", inputs, outputs, initializers),
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        ir_version=9,
    )
    return onnx.shape_inference.infer_shapes(model).SerializeToString()


def pointwise_conv_model(b, bias=None, relu=False):
    k, n = b.shape
    inputs = ["A", "W"]
    initializers = [onnx.numpy_helper.from_array(b.T.reshape(n, k, 1, 1), "W")]
    if bias is not None:
        inputs.append("bias")
        initializers.append(onnx.numpy_helper.from_array(bias, "bias"))
    # This is the existing layout transform's representation of a fused NHWC Conv.
    node = onnx.helper.make_node(
        "Conv",
        inputs,
        ["Y0"],
        domain="com.ms.internal.nhwc",
        kernel_shape=[1, 1],
        **({"activation": "Relu"} if relu else {}),
    )
    return onnx.helper.make_model(
        onnx.helper.make_graph(
            [node],
            "pointwise_conv_matmul",
            [onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT16, [1, 1, 1, k])],
            [onnx.helper.make_tensor_value_info("Y0", onnx.TensorProto.FLOAT16, [1, 1, 1, n])],
            initializers,
        ),
        opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("com.ms.internal.nhwc", 11)],
        ir_version=9,
    ).SerializeToString()


def matmul_session(model, device, *, capture=False, robustness=True):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.use_deterministic_compute = True
    options.log_severity_level = 1
    provider_options = {"validationMode": "full", "enableRobustness": str(int(robustness))}
    if capture:
        provider_options["enableGraphCapture"] = "1"
    options.add_provider_for_devices([device], provider_options)
    return ort.InferenceSession(model, options)


@contextmanager
def capture_native_stderr():
    # ORT's native logger writes to fd 2, not Python's redirected sys.stderr.
    with tempfile.TemporaryFile(mode="w+b") as log:
        sys.stderr.flush()
        original = os.dup(2)
        try:
            os.dup2(log.fileno(), 2)
            yield log
        finally:
            sys.stderr.flush()
            os.dup2(original, 2)
            os.close(original)


def bind_matmul(session, a, weights=None):
    binding = session.io_binding()
    values = {}
    feeds = {"A": a}
    if weights is not None:
        feeds["B0"] = weights
    for name, array in feeds.items():
        value = session.create_ortvalue_from_shape_and_type(array.shape, array.dtype, "webgpu")
        ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(array)], [value])
        values[name] = value
        binding.bind_ortvalue_input(name, value)
    for output in session.get_outputs():
        output_dtype = {"tensor(float16)": np.float16, "tensor(float)": np.float32}[output.type]
        value = session.create_ortvalue_from_shape_and_type(output.shape, output_dtype, "webgpu")
        values[output.name] = value
        binding.bind_ortvalue_output(output.name, value)
    return binding, values


class TestWebGpuMatMul(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        devices = webgpu_devices()
        if not devices:
            raise unittest.SkipTest("No WebGPU execution provider device available.")
        cls.device = devices[0]

    def assert_route(self, programs, gemv, log):
        self.assertTrue(programs, f"No WebGPU program diagnostics captured:\n{log}")
        if gemv:
            self.assertIn("MatMulGemv", programs, log)
        else:
            self.assertNotIn("MatMulGemv", programs, log)

    def run_case(
        self,
        a,
        b,
        *,
        gemv=True,
        dynamic_b=False,
        capture=False,
        updates=False,
        robustness=True,
        model=None,
        bias=None,
        relu=False,
    ):
        with capture_native_stderr() as log:
            session = matmul_session(
                model if model is not None else matmul_model(a.shape, [b], dynamic_b=dynamic_b),
                self.device,
                capture=capture,
                robustness=robustness,
            )
            log.seek(0)
            initialization_log = log.read().decode("utf-8", errors="replace")
        # A still-live shared context can otherwise silently ignore a changed robustness option.
        self.assertNotIn("initialized with enableRobustness=", initialization_log)
        self.assertNotIn("enableRobustness cannot affect", initialization_log)
        binding, values = bind_matmul(session, a, b if dynamic_b else None)
        try:
            previous = None
            for iteration in range(4):
                if updates and iteration:
                    a = np.ascontiguousarray(-a)
                    ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(a)], [values["A"]])
                    if dynamic_b:
                        b = np.ascontiguousarray(b * np.float16(0.5))
                        ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(b)], [values["B0"]])
                with capture_native_stderr() as log:
                    session.run_with_iobinding(binding)
                    log.seek(0)
                    run_log = log.read().decode("utf-8", errors="replace")
                programs = re.findall(r'Starting program "(\w+)', run_log)
                if capture and iteration:
                    # Replay uses the recorded commands without selecting/encoding new programs.
                    self.assertEqual(programs, [], run_log)
                else:
                    self.assert_route(programs, gemv, run_log)
                    if model is not None:
                        self.assertTrue(any("MatMul" in program for program in programs), run_log)
                actual = binding.copy_outputs_to_cpu()[0]
                expected = a.astype(np.float64) @ b.astype(np.float64)
                if bias is not None:
                    expected += bias.astype(np.float64)
                if relu:
                    expected = np.maximum(expected, 0)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, a.dtype)
                self.assertTrue(np.isfinite(actual).all())
                # FP16 storage rounds once; a small absolute term covers cancellation near zero.
                np.testing.assert_allclose(actual, expected, atol=2e-5 if gemv else 0.02, rtol=6e-4 if gemv else 0.03)
                if previous is not None and not updates:
                    np.testing.assert_array_equal(actual, previous)
                previous = actual
        finally:
            if capture:
                session.release_captured_graph()

    def test_exact_qwen_shapes(self):
        rng = np.random.default_rng(40)
        b = rng.normal(0, 0.1, (5120, 48)).astype(np.float16)
        for m in (1, 8, 66):
            with self.subTest(m=m, output_shape=(m, 48)):
                self.run_case(rng.normal(0, 0.1, (m, 5120)).astype(np.float16), b, gemv=m == 1)

    def test_target_and_nearby_shapes(self):
        rng = np.random.default_rng(42)
        for k, n in [(5120, 48), (2048, 16), (4096, 32), (5119, 44), (5121, 52), (8192, 64)]:
            with self.subTest(k=k, n=n):
                a = rng.normal(0, 0.1, (1, k)).astype(np.float16)
                b = rng.normal(0, 0.1, (k, n)).astype(np.float16)
                self.run_case(a, b)

    def test_vector_and_singleton_dimensions(self):
        rng = np.random.default_rng(43)
        for a_shape, b_shape in [((5120,), (5120, 48)), ((1, 1, 5120), (5120, 48)), ((1, 1, 1, 5120), (1, 5120, 48))]:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                self.run_case(
                    rng.normal(0, 0.1, a_shape).astype(np.float16),
                    rng.normal(0, 0.1, b_shape).astype(np.float16),
                )

    def test_selector_boundaries(self):
        rng = np.random.default_rng(47)
        cases = [
            *(
                (1, k, 48, np.float16, accepted)
                for k, accepted in [
                    (2047, False),
                    (2048, True),
                    (2049, True),
                    (8191, True),
                    (8192, True),
                    (8193, False),
                ]
            ),
            *(
                (1, 5120, n, np.float16, accepted)
                for n, accepted in [(15, False), (16, True), (17, False), (63, False), (64, True), (65, False)]
            ),
            (1, 5120, 48, np.float16, True),
            (2, 5120, 48, np.float16, False),
            (1, 5120, 48, np.float32, False),
        ]
        for m, k, n, dtype, gemv in cases:
            with self.subTest(m=m, k=k, n=n, dtype=dtype):
                self.run_case(
                    rng.normal(0, 0.1, (m, k)).astype(dtype),
                    rng.normal(0, 0.1, (k, n)).astype(dtype),
                    gemv=gemv,
                )

    def test_batch_broadcast_stays_outside_gate(self):
        rng = np.random.default_rng(48)
        for a_shape, b_shape in [((2, 1, 5120), (5120, 48)), ((1, 5120), (2, 5120, 48))]:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                self.run_case(
                    rng.normal(0, 0.1, a_shape).astype(np.float16),
                    rng.normal(0, 0.1, b_shape).astype(np.float16),
                    gemv=False,
                )

    def test_pointwise_conv_bias_and_activation_exclude_gemv(self):
        rng = np.random.default_rng(49)
        a = rng.normal(0, 0.1, (1, 1, 1, 5120)).astype(np.float16)
        b = rng.normal(0, 0.1, (5120, 48)).astype(np.float16)
        for has_bias, relu in [(False, False), (True, False), (False, True), (True, True)]:
            with self.subTest(has_bias=has_bias, relu=relu):
                bias = np.linspace(-1, 1, 48, dtype=np.float16) if has_bias else None
                self.run_case(
                    a, b, gemv=not (has_bias or relu), model=pointwise_conv_model(b, bias, relu), bias=bias, relu=relu
                )

    def test_robustness_off_decode_and_nonzero_tail(self):
        # Device robustness is immutable and a context can outlive an individual session.
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "TestWebGpuMatMul.run_robustness_off_cases"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("Ran 1 test", output)
        self.assertNotIn("skipped", output)

    def run_robustness_off_cases(self):
        rng = np.random.default_rng(50)
        self.run_case(
            rng.normal(0, 0.1, (1, 5120)).astype(np.float16),
            rng.normal(0, 0.1, (5120, 48)).astype(np.float16),
            robustness=False,
        )
        a = np.zeros((1, 5121), dtype=np.float16)
        b = np.zeros((5121, 52), dtype=np.float16)
        a[0, -1] = 2
        b[-1] = np.arange(1, 53, dtype=np.float16) / 16
        self.run_case(a, b, robustness=False)

    def test_fp32_accumulation_and_cancellation(self):
        a = np.ones((1, 5120), dtype=np.float16)
        b = np.empty((5120, 48), dtype=np.float16)
        b[:2560] = 32
        b[2560:] = -32
        b[-1] += np.arange(48, dtype=np.float16) / 16
        self.run_case(a, b)
        self.run_case(np.zeros_like(a), b)
        # Each product exceeds f16 range, but the exact sum is zero.
        a.fill(256)
        b[:2560] = 256
        b[2560:] = -256
        self.run_case(a, b)

    def test_mixed_magnitude_accumulation(self):
        a = np.ones((1, 5120), dtype=np.float16)
        b = np.zeros((5120, 48), dtype=np.float16)
        # Each K lane adds a small term between large opposite terms.
        b[:128] = 256
        b[128:256] = np.arange(1, 49, dtype=np.float16) / 1024
        b[256:384] = -256
        b[384:512] = -np.arange(1, 49, dtype=np.float16) / 2048
        self.run_case(a, b)

    def test_dynamic_weights_and_capture_replay(self):
        rng = np.random.default_rng(44)
        for capture in (False, True):
            with self.subTest(capture=capture):
                self.run_case(
                    rng.normal(0, 0.1, (1, 5120)).astype(np.float16),
                    rng.normal(0, 0.1, (5120, 48)).astype(np.float16),
                    dynamic_b=True,
                    capture=capture,
                    updates=True,
                )

    def test_outside_gate(self):
        rng = np.random.default_rng(45)
        cases = [
            ((2, 5120), (5120, 48), np.float16),
            ((2, 1, 2048), (2, 2048, 48), np.float16),
            ((1, 1024), (1024, 48), np.float16),
            ((1, 8193), (8193, 48), np.float16),
            ((1, 2048), (2048, 12), np.float16),
            ((1, 2048), (2048, 47), np.float16),
            ((1, 2048), (2048, 68), np.float16),
            ((1, 5120), (5120, 48), np.float32),
        ]
        for a_shape, b_shape, dtype in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape, dtype=dtype):
                self.run_case(
                    rng.normal(0, 0.1, a_shape).astype(dtype),
                    rng.normal(0, 0.1, b_shape).astype(dtype),
                    gemv=False,
                )


if __name__ == "__main__":
    unittest.main()
