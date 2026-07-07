#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

import onnxruntime as ort
from onnxruntime.capi import _pybind_state as _pybind


@contextmanager
def set_env(name: str, value: str):
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


@unittest.skipIf("CUDAExecutionProvider" not in ort.get_available_providers(), "CUDA is not available")
@unittest.skipUnless(hasattr(_pybind, "pack_weights_for_cuda_mixed_gemm"), "fpA_intB weight packer is unavailable")
class TestMatMulNBitsPrepackedCuda(unittest.TestCase):
    def _quantize_weight(self, weight: np.ndarray, bits: int, block_size: int):
        k, n = weight.shape
        k_blocks = (k + block_size - 1) // block_size
        blob_size = block_size * bits // 8
        q_weight = np.zeros((n, k_blocks, blob_size), dtype=np.uint8)
        scales = np.zeros((n, k_blocks), dtype=np.float16)
        if bits == 4:
            zero_points = np.zeros((n, (k_blocks + 1) // 2), dtype=np.uint8)
            _pybind.quantize_matmul_4bits(q_weight, weight, scales, zero_points, block_size, n, k, True)
        elif bits == 8:
            zero_points = np.zeros((n, k_blocks), dtype=np.uint8)
            _pybind.quantize_matmul_8bits(q_weight, weight, scales, zero_points, block_size, n, k, True)
        else:
            raise ValueError(f"unsupported bits: {bits}")

        return q_weight, np.abs(scales)

    def _make_model(
        self,
        a_shape: tuple[int, int],
        b: np.ndarray,
        scales: np.ndarray,
        bits: int,
        block_size: int,
        weight_prepacked: int,
        bias: np.ndarray | None = None,
    ) -> ModelProto:
        m, k = a_shape
        n = b.shape[0]
        inputs = ["A", "B", "scales"]
        initializer = [
            numpy_helper.from_array(b, name="B"),
            numpy_helper.from_array(scales, name="scales"),
        ]
        if bias is not None:
            # bias is input index 5; indices 3 (zero_points) and 4 (g_idx) are left empty.
            inputs.extend(["", "", "bias"])
            initializer.append(numpy_helper.from_array(bias, name="bias"))
        node = helper.make_node(
            "MatMulNBits",
            inputs,
            ["Y"],
            domain="com.microsoft",
            K=k,
            N=n,
            bits=bits,
            block_size=block_size,
            weight_prepacked=weight_prepacked,
        )
        graph = helper.make_graph(
            [node],
            "matmulnbits_prepacked_cuda_test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [m, k])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [m, n])],
            initializer=initializer,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 10
        return model

    def _run_model(self, model: ModelProto, a: np.ndarray) -> np.ndarray:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
        return sess.run(None, {"A": a})[0]

    def _check_prepacked_parity(
        self,
        bits: int,
        block_size: int,
        m: int,
        has_bias: bool = False,
        force_arch: int = 80,
        weight_prepacked: int = 1,
    ):
        rng = np.random.default_rng(1234 + bits * 10 + block_size + m)
        k = 256
        n = 256 if bits == 8 else 512
        a = rng.normal(0.0, 0.25, size=(m, k)).astype(np.float16)
        weight = rng.normal(0.0, 0.25, size=(k, n)).astype(np.float16)
        bias = rng.normal(0.0, 1.0, size=(n,)).astype(np.float16) if has_bias else None

        q_weight, scales = self._quantize_weight(weight, bits, block_size)
        prepacked_flat = _pybind.pack_weights_for_cuda_mixed_gemm(q_weight.reshape(n, -1), n, k, bits, force_arch)
        prepacked_weight = np.asarray(prepacked_flat, dtype=np.int8).view(np.uint8).reshape(q_weight.shape)

        raw_model = self._make_model((m, k), q_weight, scales, bits, block_size, weight_prepacked=0, bias=bias)
        prepacked_model = self._make_model(
            (m, k), prepacked_weight, scales, bits, block_size, weight_prepacked=weight_prepacked, bias=bias
        )

        with set_env("ORT_FPA_INTB_GEMM", "1"):
            raw_output = self._run_model(raw_model, a)
            prepacked_output = self._run_model(prepacked_model, a)

        np.testing.assert_allclose(prepacked_output, raw_output, rtol=1e-3, atol=1e-3)

    def test_int4_sm80_prepacked_weight_matches_runtime_prepack(self):
        self._check_prepacked_parity(bits=4, block_size=64, m=1)
        self._check_prepacked_parity(bits=4, block_size=128, m=32)

    def test_int4_bs32_sm80_prepacked_weight_matches_runtime_prepack(self):
        # Production rc2/rc3 models use block_size=32 (SM80/Ampere layout, weight_prepacked=1).
        self._check_prepacked_parity(bits=4, block_size=32, m=1)
        self._check_prepacked_parity(bits=4, block_size=32, m=32)

    def test_int8_sm80_prepacked_weight_matches_runtime_prepack(self):
        self._check_prepacked_parity(bits=8, block_size=64, m=1)
        self._check_prepacked_parity(bits=8, block_size=128, m=32)

    def test_int4_sm80_prepacked_weight_with_bias_matches_runtime_prepack(self):
        self._check_prepacked_parity(bits=4, block_size=64, m=1, has_bias=True)
        self._check_prepacked_parity(bits=4, block_size=128, m=32, has_bias=True)

    def _check_sm90_parity(self, **kwargs):
        # The native SM90 (Hopper) layout (force_arch=90, weight_prepacked=2) only runs on an SM90
        # device; the MatMulNBits kernel rejects it up front elsewhere. Self-gate by skipping when
        # the compute-capability guard fires so the test is a no-op on non-Hopper CI.
        try:
            self._check_prepacked_parity(force_arch=90, weight_prepacked=2, **kwargs)
        except Exception as exc:
            if "compute capability 9.0" in str(exc):
                self.skipTest("native SM90 fpA_intB requires a Hopper (SM90) device")
            raise

    def test_int4_sm90_prepacked_weight_matches_runtime_prepack(self):
        self._check_sm90_parity(bits=4, block_size=64, m=1)
        self._check_sm90_parity(bits=4, block_size=128, m=32)

    def test_int4_sm90_prepacked_weight_with_bias_matches_runtime_prepack(self):
        self._check_sm90_parity(bits=4, block_size=128, m=32, has_bias=True)

    def test_int8_sm90_prepacked_weight_matches_runtime_prepack(self):
        self._check_sm90_parity(bits=8, block_size=128, m=32)


if __name__ == "__main__":
    unittest.main()
