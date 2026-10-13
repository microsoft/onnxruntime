#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

import onnxruntime as ort
from onnxruntime.capi import _pybind_state as _pybind
from onnxruntime.quantization.cuda_quantizer import _pack_weights_for_cuda_mixed_gemm

try:
    from onnxruntime.capi import onnxruntime_cuda_quant_preprocess as _cuda_quant
except ImportError:
    _cuda_quant = None


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
@unittest.skipUnless(_cuda_quant is not None, "fpA_intB weight packer is unavailable")
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
        prepacked_flat = _cuda_quant.pack_weights_for_cuda_mixed_gemm(q_weight.reshape(n, -1), n, k, bits, force_arch)
        prepacked_weight = np.asarray(prepacked_flat, dtype=np.int8).view(np.uint8).reshape(q_weight.shape)

        raw_model = self._make_model((m, k), q_weight, scales, bits, block_size, weight_prepacked=0, bias=bias)
        prepacked_model = self._make_model(
            (m, k), prepacked_weight, scales, bits, block_size, weight_prepacked=weight_prepacked, bias=bias
        )

        with set_env("ORT_FPA_INTB_GEMM", "1"):
            raw_output = self._run_model(raw_model, a)
            try:
                prepacked_output = self._run_model(prepacked_model, a)
            except Exception as exc:
                outside_compact_contract = block_size != 32 or has_bias or weight_prepacked != 1
                if outside_compact_contract and "compact fpA_intB build supports prepacked weights for" in str(exc):
                    self.skipTest("case is outside the compact fpA_intB build contract")
                raise

        np.testing.assert_allclose(prepacked_output, raw_output, rtol=1e-3, atol=1e-3)

    def test_int4_sm80_prepacked_weight_matches_runtime_prepack(self):
        self._check_prepacked_parity(bits=4, block_size=64, m=1)
        self._check_prepacked_parity(bits=4, block_size=128, m=32)

    def test_int4_bs32_sm80_prepacked_weight_matches_runtime_prepack(self):
        # Production rc2/rc3 models use block_size=32 (SM80/Ampere layout, weight_prepacked=1).
        for m in (1, 15, 16, 32, 128, 512):
            with self.subTest(m=m):
                self._check_prepacked_parity(bits=4, block_size=32, m=m)

    def test_int8_sm80_prepacked_weight_matches_runtime_prepack(self):
        self._check_prepacked_parity(bits=8, block_size=64, m=1)
        self._check_prepacked_parity(bits=8, block_size=128, m=32)

    def test_int8_bs32_sm80_prepacked_weight_matches_runtime_prepack(self):
        for m in (1, 15, 16, 32, 128, 512):
            with self.subTest(m=m):
                self._check_prepacked_parity(bits=8, block_size=32, m=m)

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


class TestMatMulNBitsCompactContract(unittest.TestCase):
    def test_skips_only_unsupported_compact_prepacked_cases(self):
        message = "This compact fpA_intB build supports prepacked weights for FP16/BF16 activations"
        cases = (
            (64, message, unittest.SkipTest),
            (32, message, RuntimeError),
            (64, "Unexpected execution failure", RuntimeError),
        )
        for block_size, error, expected_exception in cases:
            with self.subTest(block_size=block_size, error=error):
                case = TestMatMulNBitsPrepackedCuda()
                weights = np.zeros((512, 256 // block_size, block_size // 2), dtype=np.uint8)
                scales = np.ones(weights.shape[:2], dtype=np.float16)
                packer = MagicMock()
                packer.pack_weights_for_cuda_mixed_gemm.return_value = weights.reshape(-1).view(np.int8)
                with (
                    patch.object(sys.modules[__name__], "_cuda_quant", packer),
                    patch.object(case, "_quantize_weight", return_value=(weights, scales)),
                    patch.object(case, "_run_model", side_effect=[np.zeros((1, 512)), RuntimeError(error)]),
                    self.assertRaises(expected_exception),
                ):
                    case._check_prepacked_parity(bits=4, block_size=block_size, m=1)


@unittest.skipIf("CUDAExecutionProvider" not in ort.get_available_providers(), "CUDA is not available")
@unittest.skipUnless(_cuda_quant is not None, "standalone CUDA weight packer (parity oracle) is unavailable")
class TestCudaQuantizerTorchPackerParity(unittest.TestCase):
    """Validate the PyTorch mixed-GEMM packer in cuda_quantizer.py against the CUDA oracle.

    ``cuda_quantizer._pack_weights_for_cuda_mixed_gemm`` (PyTorch, used in production, and the
    only option on Windows where the standalone module is not built) must be byte-identical to
    the standalone ``onnxruntime_cuda_quant_preprocess.pack_weights_for_cuda_mixed_gemm`` (the
    CUDA code the runtime prepack uses). This test is the guard against silent drift; it only
    runs where the oracle is built (non-Windows CUDA).
    """

    def _check(self, bits: int, force_arch: int, n: int, k: int):
        pack = 8 // bits
        rng = np.random.default_rng(20260708 + bits * 100 + force_arch + n + k)
        q = rng.integers(0, 256, size=(n, k // pack), dtype=np.uint8)
        oracle = np.asarray(_cuda_quant.pack_weights_for_cuda_mixed_gemm(q, n, k, bits, force_arch), dtype=np.int8)
        torch_out = _pack_weights_for_cuda_mixed_gemm(q, n, k, bits, force_arch).astype(np.int8)
        self.assertEqual(oracle.shape, torch_out.shape, f"shape mismatch bits={bits} arch={force_arch} N={n} K={k}")
        np.testing.assert_array_equal(
            torch_out, oracle, err_msg=f"byte mismatch bits={bits} arch={force_arch} N={n} K={k}"
        )

    def test_torch_packer_matches_cuda_oracle(self):
        # Cover both weight bit-widths, both mixed-GEMM layouts (SM80/SM90), and a GPT-OSS-20B
        # MoE shape (fused gate+up FC1 [5760, 2880] and down FC2 [2880, 2880]).
        shapes = [(256, 256), (512, 256), (256, 512), (5760, 2880), (2880, 2880), (128, 128)]
        for bits in (4, 8):
            for force_arch in (80, 90):
                for n, k in shapes:
                    with self.subTest(bits=bits, force_arch=force_arch, n=n, k=k):
                        self._check(bits, force_arch, n, k)

    def test_explicit_sm90_layout_does_not_fall_back_to_sm80(self):
        # Layout selection must not depend on whether SM90 compute kernels were compiled.
        n = k = 128
        for bits in (4, 8):
            with self.subTest(bits=bits):
                q = np.random.default_rng(42).integers(0, 256, size=(n, k // (8 // bits)), dtype=np.uint8)
                sm80 = np.asarray(_cuda_quant.pack_weights_for_cuda_mixed_gemm(q, n, k, bits, 80))
                sm90 = np.asarray(_cuda_quant.pack_weights_for_cuda_mixed_gemm(q, n, k, bits, 90))
                self.assertFalse(np.array_equal(sm80, sm90), "force_arch=90 must preserve the SM90 weight layout")


@unittest.skipIf("CUDAExecutionProvider" not in ort.get_available_providers(), "CUDA is not available")
@unittest.skipUnless(hasattr(_pybind, "quantize_matmul_4bits"), "MatMulNBits 4-bit quantizer is unavailable")
class TestFpAIntBConfigKeys(unittest.TestCase):
    """Session-config keys ep.cuda.fpa_intb_gemm / ep.cuda.fpa_intb_profile_m.

    These do not need the offline weight packer (pack_weights_for_cuda_mixed_gemm), so they run in
    more build configurations than TestMatMulNBitsPrepackedCuda. They cover: the config key enabling
    the fpA_intB path (on/off only), session config overriding the ORT_FPA_INTB_GEMM env var, the
    profile-M key being accepted, and env-var backward compatibility.
    """

    def setUp(self):
        # Make sure no env override leaks in from the process / other tests.
        for name in (
            "ORT_FPA_INTB_GEMM",
            "ORT_FPA_INTB_PROFILE_M",
            "ORT_MATMULNBITS_M_CHUNK_SIZE",
            "ORT_MATMULNBITS_FORCE_CHUNKED",
            "ORT_FPA_INTB_GEMV_PAIRED_K",
        ):
            os.environ.pop(name, None)

    def _quantize_weight(self, weight: np.ndarray, bits: int, block_size: int):
        k, n = weight.shape
        k_blocks = (k + block_size - 1) // block_size
        blob_size = block_size * bits // 8
        q_weight = np.zeros((n, k_blocks, blob_size), dtype=np.uint8)
        scales = np.zeros((n, k_blocks), dtype=np.float16)
        zero_points = np.zeros((n, (k_blocks + 1) // 2), dtype=np.uint8)
        _pybind.quantize_matmul_4bits(q_weight, weight, scales, zero_points, block_size, n, k, True)
        return q_weight, np.abs(scales)

    def _make_model(self, m, k, n, q_weight, scales, bits, block_size, weight_prepacked=0) -> ModelProto:
        node = helper.make_node(
            "MatMulNBits",
            ["A", "B", "scales"],
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
            "fpa_intb_config_keys_test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [m, k])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [m, n])],
            initializer=[
                numpy_helper.from_array(q_weight, name="B"),
                numpy_helper.from_array(scales, name="scales"),
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 10
        return model

    def _run(self, model: ModelProto, a: np.ndarray, config: dict[str, str] | None = None) -> np.ndarray:
        so = ort.SessionOptions()
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        for key, value in (config or {}).items():
            so.add_session_config_entry(key, value)
        sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CUDAExecutionProvider"])
        return sess.run(None, {"A": a})[0]

    def _make_int4_case(self, m=32, k=256, n=512, block_size=32, weight_prepacked=0):
        rng = np.random.default_rng(2024)
        a = rng.normal(0.0, 0.25, size=(m, k)).astype(np.float16)
        weight = rng.normal(0.0, 0.25, size=(k, n)).astype(np.float16)
        q_weight, scales = self._quantize_weight(weight, 4, block_size)
        model = self._make_model(m, k, n, q_weight, scales, 4, block_size, weight_prepacked=weight_prepacked)
        return model, a, q_weight, scales

    def _require_fpa_intb(self):
        # Constant zero coefficients are invariant under the prepacked weight permutation.
        model = self._make_model(
            1, 64, 64, np.full((64, 2, 16), 0x88, dtype=np.uint8), np.ones((64, 2), dtype=np.float16), 4, 32, 1
        )
        try:
            self._run(model, np.ones((1, 64), dtype=np.float16), {"ep.cuda.fpa_intb_profile_m": "1"})
        except Exception as exc:
            if any(
                message in str(exc)
                for message in (
                    "weight_prepacked requires an ONNX Runtime build with onnxruntime_USE_FPA_INTB_GEMM=ON",
                    "This compact fpA_intB build supports prepacked weights for",
                    "weight_prepacked requires the fpA_intB path, but it is unsupported for this node",
                )
            ):
                self.skipTest(f"fpA_intB GEMM is unavailable on this build/device: {exc}")
            raise

    def _make_paired_k_case(self, m, k, n=512):
        row = np.arange(k)[:, None]
        col = np.arange(n)[None, :]
        coefficients = ((row + row // 32 + 3 * col) % 15 - 7).astype(np.int8)
        unsigned = (coefficients + 8).astype(np.uint8).T
        q_weight = (unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)).reshape(n, k // 32, 16)
        scales = ((1 + (np.arange(n)[:, None] + np.arange(k // 32)[None, :]) % 4) / 256).astype(np.float16)
        a = ((1 + (np.arange(m)[:, None] + np.arange(k)[None, :]) % 3) / 4).astype(np.float16)
        weight = coefficients.astype(np.float32) * np.repeat(scales.T.astype(np.float32), 32, axis=0)
        expected = (a.astype(np.float32) @ weight).astype(np.float16)
        return self._make_model(m, k, n, q_weight, scales, 4, 32), a, expected

    def test_config_key_enables_fpa_intb(self):
        # On fpA_intB-capable hardware (compute capability >= 7.5) the baseline (no config) runs the
        # standard dequant path -- for a non-prepacked node the enable flag defaults to disabled --
        # while the config key selects the fpA_intB path; the two paths must stay numerically
        # equivalent. On sm < 75 both fall back to the dequant path, so this asserts equivalence
        # rather than the switch itself (the prepacked tests force and exercise the fpA_intB kernel).
        # Only on/off is accepted.
        model, a, _, _ = self._make_int4_case()
        ref = self._run(model, a)
        for value in ("1", "on", "all", "true"):
            out = self._run(model, a, {"ep.cuda.fpa_intb_gemm": value})
            np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2, err_msg=f"value={value}")

    def test_profile_m_config_key_accepted(self):
        model, a, _, _ = self._make_int4_case()
        ref = self._run(model, a)
        out = self._run(model, a, {"ep.cuda.fpa_intb_gemm": "1", "ep.cuda.fpa_intb_profile_m": "1,8,32"})
        np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2)

    def test_gemv_paired_k_config_key(self):
        self._require_fpa_intb()
        for m in (4, 5, 6, 7, 8, 9):
            model, a, _, _ = self._make_int4_case(m=m, k=1024, n=2048)
            ref = self._run(model, a, {"ep.cuda.fpa_intb_gemm": "1"})
            for value in ("0", "off", "1", "force"):
                out = self._run(
                    model,
                    a,
                    {
                        "ep.cuda.fpa_intb_gemm": "1",
                        "ep.cuda.fpa_intb_profile_m": str(m),
                        "ep.cuda.fpa_intb_gemv_paired_k": value,
                    },
                )
                np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2, err_msg=f"m={m} value={value}")

    def test_gemv_paired_k_multi_pass(self):
        self._require_fpa_intb()
        for m in (5, 6, 7, 8):
            for k in (1024, 2048, 4096):
                with self.subTest(m=m, k=k):
                    model, a, expected = self._make_paired_k_case(m, k)
                    out = self._run(
                        model,
                        a,
                        {
                            "ep.cuda.fpa_intb_gemm": "1",
                            "ep.cuda.fpa_intb_profile_m": str(m),
                            "ep.cuda.fpa_intb_gemv_paired_k": "force",
                        },
                    )
                    np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)

    def test_gemv_paired_k_known_overflow_limit(self):
        self._require_fpa_intb()
        m, k, n = 8, 1024, 512
        alternating_a = np.tile(np.array([40000, -40000], dtype=np.float16), (m, k // 2))
        for name, packed_byte, scale, a in (
            ("activation_cancellation", 0x99, 1, alternating_a),
            ("weight_cancellation", 0x1F, 2048, np.ones((m, k), dtype=np.float16)),
        ):
            with self.subTest(case=name):
                model = self._make_model(
                    m,
                    k,
                    n,
                    np.full((n, k // 32, 16), packed_byte, dtype=np.uint8),
                    np.full((n, k // 32), scale, dtype=np.float16),
                    4,
                    32,
                )
                config = {
                    "ep.cuda.fpa_intb_gemm": "1",
                    "ep.cuda.fpa_intb_profile_m": str(m),
                    "ep.cuda.fpa_intb_gemv_paired_k": "0",
                }
                baseline = self._run(model, a, config)
                np.testing.assert_array_equal(baseline, np.zeros((m, n), dtype=np.float16))
                out = self._run(
                    model,
                    a,
                    {**config, "ep.cuda.fpa_intb_gemv_paired_k": "force"},
                )
                # Document the experimental FP16-lane limit, not a safe-inference guarantee.
                self.assertTrue(np.isnan(out).all())

    def test_gemv_paired_k_tactic_selection(self):
        self._require_fpa_intb()
        # The native debug flag is cached on first use, so set it in a fresh process.
        script = """
import runpy
import sys

case = runpy.run_path(sys.argv[1])["TestFpAIntBConfigKeys"]()
for m in (4, 5, 6, 7, 8, 9):
    model, a, _, _ = case._make_int4_case(m=m, k=1024, n=2048)
    for value in ("0", "off", "1", "force", "0", "force"):
        print(f"paired_k_test M={m} value={value}", flush=True)
        case._run(model, a, {
            "ep.cuda.fpa_intb_gemm": "1",
            "ep.cuda.fpa_intb_profile_m": str(m),
            "ep.cuda.fpa_intb_gemv_paired_k": value,
            "ep.cuda.matmul_nbits_m_chunk_size": "0",
        })
"""
        result = subprocess.run(
            [sys.executable, "-c", script, os.path.abspath(__file__)],
            env={**os.environ, "ORT_FPA_INTB_DEBUG": "1"},
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[fpA_intB_debug]", result.stdout)
        dispatches = result.stdout.split("paired_k_test ")[1:]
        self.assertEqual(len(dispatches), 36, result.stdout)
        for dispatch in dispatches:
            header = dispatch.splitlines()[0]
            m = int(header.split()[0].split("=")[1])
            value = header.split()[1].split("=")[1]
            with self.subTest(m=m, value=value):
                if value == "force" and 5 <= m <= 8:
                    self.assertIn("kernel=GEMV(cuda)", dispatch)
                    self.assertIn("cuda kernel variant: 1", dispatch)
                    self.assertIn("GEMV launch: paired_k=1", dispatch)
                elif value in ("0", "off") or m not in (5, 6, 7, 8):
                    self.assertIn("cuda kernel variant: 0", dispatch)
                    self.assertNotIn("GEMV launch: paired_k=1", dispatch)

    def test_wave_aware_gemv_config_key_matches_default(self):
        for m in (7, 8, 9):
            for n in (512, 10240):
                with self.subTest(m=m, n=n):
                    model, a, _, _ = self._make_int4_case(m=m, n=n)
                    config = {"ep.cuda.fpa_intb_gemm": "1", "ep.cuda.fpa_intb_profile_m": "8,16"}
                    ref = self._run(model, a, config)
                    for value in ("0", "1", "0"):
                        out = self._run(model, a, {**config, "ep.cuda.fpa_intb_gemv_wave_aware": value})
                        np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2, err_msg=f"value={value}")

    def test_invalid_wave_aware_gemv_config_rejected(self):
        model, a, _, _ = self._make_int4_case(m=8, weight_prepacked=1)
        for value in ("", "-1", "2", "on"):
            with self.subTest(value=value):
                with self.assertRaises(Exception) as error:
                    self._run(model, a, {"ep.cuda.fpa_intb_gemm": "1", "ep.cuda.fpa_intb_gemv_wave_aware": value})
                if "weight_prepacked requires an ONNX Runtime build with onnxruntime_USE_FPA_INTB_GEMM=ON" in str(
                    error.exception
                ):
                    self.skipTest("fpA_intB GEMM is not compiled in this build")
                self.assertRegex(str(error.exception), "Invalid MatMulNBits wave-aware GEMV option")

    def test_session_config_overrides_env(self):
        # env var says off, session config says on -> the session config must win.
        model, a, _, _ = self._make_int4_case()
        ref = self._run(model, a)
        with set_env("ORT_FPA_INTB_GEMM", "0"):
            out = self._run(model, a, {"ep.cuda.fpa_intb_gemm": "1"})
        np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2)

    def test_env_var_backward_compatible(self):
        model, a, _, _ = self._make_int4_case()
        ref = self._run(model, a)
        # "1" plus a legacy non-zero numeric value (previously a bitmask) both mean "enabled" now.
        for value in ("1", "4"):
            with set_env("ORT_FPA_INTB_GEMM", value):
                out = self._run(model, a)
            np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2, err_msg=f"env={value}")

    def test_m_chunk_size_matches_unchunked(self):
        # M=100 with chunk 32 runs three CUTLASS chunks plus a 4-row trailing chunk on the GEMV;
        # chunk 8 runs GEMV-only chunks; chunk >= M is a single launch. The shape is below the
        # chunking size gate, so ORT_MATMULNBITS_FORCE_CHUNKED bypasses it.
        model, a, _, _ = self._make_int4_case(m=100)
        ref = self._run(model, a, {"ep.cuda.fpa_intb_gemm": "1"})
        with set_env("ORT_MATMULNBITS_FORCE_CHUNKED", "1"):
            for chunk in ("8", "32", "64", "100", "256", "0"):
                out = self._run(model, a, {"ep.cuda.fpa_intb_gemm": "1", "ep.cuda.matmul_nbits_m_chunk_size": chunk})
                np.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2, err_msg=f"chunk={chunk}")

            with set_env("ORT_MATMULNBITS_M_CHUNK_SIZE", "16"):
                out = self._run(model, a, {"ep.cuda.fpa_intb_gemm": "1"})
            np.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2, err_msg="env chunk=16")

    def test_empty_input_returns_empty_output(self):
        model, a, _, _ = self._make_int4_case(m=0)
        for config in (
            {},
            {"ep.cuda.fpa_intb_gemm": "1"},
            {"ep.cuda.fpa_intb_gemm": "1", "ep.cuda.matmul_nbits_m_chunk_size": "32"},
        ):
            out = self._run(model, a, config)
            self.assertEqual(out.shape, (0, 512), msg=f"config={config}")

    def test_invalid_m_chunk_size_rejected(self):
        # Prepacked weights require fpA_intB support. Invalid config is rejected before the weight
        # data is interpreted, so this test does not need the optional offline weight packer.
        model, a, _, _ = self._make_int4_case(weight_prepacked=1)
        for chunk in ("-1", "abc"):
            with self.assertRaises(Exception) as error:
                self._run(model, a, {"ep.cuda.fpa_intb_gemm": "1", "ep.cuda.matmul_nbits_m_chunk_size": chunk})
            if "weight_prepacked requires an ONNX Runtime build with onnxruntime_USE_FPA_INTB_GEMM=ON" in str(
                error.exception
            ):
                self.skipTest("fpA_intB GEMM is not compiled in this build")
            self.assertRegex(str(error.exception), "Invalid MatMulNBits M chunk size")


if __name__ == "__main__":
    unittest.main()
