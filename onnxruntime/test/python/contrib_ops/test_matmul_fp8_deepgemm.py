# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Opt-in integration tests; run with ORT_TEST_FP8_DEEPGEMM=1 against an SM90 CUDA build.

For device-kernel evidence, profile this test with nsys and check for
sm90_fp8_gemm_1d1d_impl. The operator also logs its selected path at VERBOSE level.
"""

import contextlib
import os
import subprocess
import sys
import unittest

from onnx import TensorProto, helper

import onnxruntime as ort

if os.environ.get("ORT_TEST_FP8_DEEPGEMM") == "1":
    import torch


@contextlib.contextmanager
def environment(**values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@unittest.skipUnless(os.environ.get("ORT_TEST_FP8_DEEPGEMM") == "1", "Requires an SM90 DeepGEMM build")
class TestMatMulFp8DeepGemm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            raise unittest.SkipTest("Requires SM90")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise unittest.SkipTest("Requires CUDAExecutionProvider")

    def run_case(
        self, dtype, m=67, n=4096, k=2048, block_size=128, bias=True, graph=False, w8a8=True, scale_value=0.137
    ):
        generator = torch.Generator(device="cuda").manual_seed(123)
        onnx_dtype = TensorProto.BFLOAT16 if dtype == torch.bfloat16 else TensorProto.FLOAT16
        a = torch.randn(m, k, device="cuda", generator=generator).to(dtype)
        b = torch.randn(n, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)
        blocks = (k + block_size - 1) // block_size
        scales = 0.01 + torch.rand(n, blocks, device="cuda", generator=generator) * 0.2
        activation_scale = torch.tensor(scale_value, device="cuda")
        bias_tensor = torch.randn(n, device="cuda", generator=generator).to(dtype) if bias else None
        output = torch.empty(m, n, device="cuda", dtype=dtype)
        initializers = [
            helper.make_tensor(
                "B", TensorProto.FLOAT8E4M3FN, [n, k], b.view(torch.uint8).cpu().numpy().tobytes(), raw=True
            )
        ]
        inputs = [
            helper.make_tensor_value_info("A", onnx_dtype, [m, k]),
            helper.make_tensor_value_info("b_scale", TensorProto.FLOAT, [n, blocks]),
        ]
        node_inputs = ["A", "B", "b_scale"]
        if w8a8:
            inputs.append(helper.make_tensor_value_info("a_scale", TensorProto.FLOAT, []))
            node_inputs.append("a_scale")
        else:
            node_inputs.append("")
        if bias:
            initializers.append(
                helper.make_tensor(
                    "bias", onnx_dtype, [n], bias_tensor.view(torch.uint8).cpu().numpy().tobytes(), raw=True
                )
            )
            node_inputs.append("bias")
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "MatMulBlockQuantizedFp8Weight",
                        node_inputs,
                        ["Y"],
                        domain="com.microsoft",
                        block_size=block_size,
                    )
                ],
                "deepgemm_test",
                inputs,
                [helper.make_tensor_value_info("Y", onnx_dtype, [m, n])],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 10
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        stream = torch.cuda.Stream()
        session = ort.InferenceSession(
            model.SerializeToString(),
            options,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "enable_cuda_graph": str(int(graph)),
                        "user_compute_stream": str(stream.cuda_stream),
                    },
                )
            ],
        )
        binding = session.io_binding()
        for name, tensor, element_type in [("A", a, onnx_dtype), ("b_scale", scales, TensorProto.FLOAT)]:
            binding.bind_input(name, "cuda", 0, element_type, list(tensor.shape), tensor.data_ptr())
        if w8a8:
            binding.bind_input("a_scale", "cuda", 0, TensorProto.FLOAT, [], activation_scale.data_ptr())
        binding.bind_output("Y", "cuda", 0, onnx_dtype, [m, n], output.data_ptr())

        # Change both scale tensors in place between calls, including CUDA graph replays.
        # Leaving FP32 scratch uncleared would accumulate earlier results into later outputs.
        for iteration in range(3):
            if iteration == 1:
                a.mul_(0.5)
                scales.mul_(1.7)
                activation_scale.fill_(0.03125)
            elif iteration == 2:
                activation_scale.zero_()
            torch.cuda.synchronize()
            session.run_with_iobinding(binding)
            torch.cuda.synchronize()
            effective_a = a.float()
            if w8a8:
                inv_scale = 1.0 / activation_scale if activation_scale.item() else 0.0
                effective_a = (a.float() * inv_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
                effective_a = effective_a * activation_scale
            effective_b = b.float() * scales.repeat_interleave(block_size, dim=1)[:, :k]
            product = effective_a.double() @ effective_b.double().T
            expected = product.to(dtype)
            if bias:
                expected = (expected.float() + bias_tensor.float()).to(dtype)
            # Allow output rounding in units of the output dtype, plus an RMS-scaled
            # absolute bound for cancellation and the fallback's operand rounding.
            error = (output.float() - expected.float()).abs()
            rms = expected.float().square().mean().sqrt().item() if expected.numel() else 0.0
            magnitude = product.abs()
            if bias:
                magnitude = magnitude + bias_tensor.float().abs()
            eps = torch.finfo(dtype).eps
            limit = eps * (2 * magnitude + 4 * max(rms, 1.0))
            self.assertTrue((error <= limit).all().item(), f"Maximum absolute error: {error.max().item()}")
            self.assertLessEqual(error.square().mean().sqrt().item(), 2 * eps * max(rms, 1.0))
            self.assertTrue(torch.isfinite(output).all().item())

    def test_dense_scales_bias_and_graph_replay(self):
        with environment(ORT_FP8_MATMUL_DEEPGEMM="1", ORT_FP8_GEMV_MAX_M="32"):
            for dtype in (torch.float16, torch.bfloat16):
                for bias in (False, True):
                    with self.subTest(dtype=dtype, bias=bias):
                        self.run_case(dtype, bias=bias, graph=True)

    def test_output_scratch_tiling(self):
        # 1 MiB permits 2048 output columns at M=127, so N=4160 needs three tiles.
        with environment(ORT_FP8_MATMUL_DEEPGEMM="1", ORT_FP8_DEQUANT_SCRATCH_MIB="1"):
            self.run_case(torch.float16, m=127, n=4160)

    def test_activation_clipping(self):
        with environment(ORT_FP8_MATMUL_DEEPGEMM="1"):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    self.run_case(dtype, scale_value=0.001)

    def test_tile_and_pipeline_boundaries(self):
        with environment(ORT_FP8_MATMUL_DEEPGEMM="1", ORT_FP8_GEMV_MAX_M="32"):
            for m, n, k in ((33, 65536, 128), (65, 4096, 2048), (127, 65536, 128)):
                with self.subTest(m=m, n=n, k=k):
                    self.run_case(torch.float16, m=m, n=n, k=k, bias=False)

    def test_native_dispatch(self):
        # Value-only tests can pass through the fallback on a build without DeepGEMM.
        result = subprocess.run(
            [sys.executable, __file__, "TestMatMulFp8DeepGemm.test_dense_scales_bias_and_graph_replay"],
            env={**os.environ, "ORT_DEEPGEMM_DISPATCH_PROBE": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("MatMulBlockQuantizedFp8Weight: using SM90 DeepGEMM", result.stderr)

    def test_fallbacks(self):
        with environment(ORT_FP8_MATMUL_DEEPGEMM="1"):
            for kwargs in (
                {"m": 4},
                {"m": 129},
                {"n": 65},
                {"n": 512},
                {"k": 272},
                {"block_size": 64},
                {"w8a8": False},
                {"k": 0},
            ):
                with self.subTest(**kwargs):
                    self.run_case(torch.float16, **kwargs)
        with environment(ORT_FP8_MATMUL_DEEPGEMM="0"):
            self.run_case(torch.float16)


if __name__ == "__main__":
    if os.environ.get("ORT_DEEPGEMM_DISPATCH_PROBE") == "1":
        ort.set_default_logger_severity(0)
    unittest.main()
