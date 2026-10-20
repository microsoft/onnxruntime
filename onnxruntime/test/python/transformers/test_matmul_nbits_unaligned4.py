# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""CUDA tests for the opt-in ORT_MATMULNBITS_UNALIGNED4_FAST_PATH (FP16 INT4, N % 8 == 4, M <= 4).

The selector must be bit-identical to the existing aligned kernels evaluated on the same weights padded to the
next multiple of 8 output columns, and must not change any ineligible configuration.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from cuda_plugin_ep_helper import CUDA_PLUGIN_EP_NAME, ensure_cuda_plugin_ep_registered
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort

ENV_NAME = "ORT_MATMULNBITS_UNALIGNED4_FAST_PATH"
LOG_PREFIX = "[MatMulNBitsUnaligned4]"
BLOCK_SIZE = 32


def _cuda_session_options():
    """Returns a plugin EP device, the CUDA provider name, or None if CUDA is unavailable."""
    if ensure_cuda_plugin_ep_registered():
        devices = [d for d in ort.get_ep_devices() if d.ep_name == CUDA_PLUGIN_EP_NAME]
        if devices:
            return devices[0]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return "CUDAExecutionProvider"
    return None


def _create_session(target, n, k, weights, scales, enabled, bias=False, zero_points=False, g_idx=False):
    os.environ[ENV_NAME] = "1" if enabled else "0"
    inputs = ["A", "B", "scales"]
    initializers = [numpy_helper.from_array(weights, "B"), numpy_helper.from_array(scales, "scales")]
    if zero_points or g_idx or bias:
        inputs += ["zp" if zero_points else "", "g_idx" if g_idx else "", "bias" if bias else ""]
    if zero_points:
        initializers.append(numpy_helper.from_array(np.full((n, (k // BLOCK_SIZE + 1) // 2), 0x35, np.uint8), "zp"))
    if g_idx:
        num_blocks = k // BLOCK_SIZE
        reversed_g_idx = (num_blocks - 1) - (np.arange(k, dtype=np.int32) // BLOCK_SIZE)
        initializers.append(numpy_helper.from_array(reversed_g_idx, "g_idx"))
    if bias:
        initializers.append(numpy_helper.from_array(np.linspace(0.125, 1.5, n, dtype=np.float16), "bias"))
    node = helper.make_node(
        "MatMulNBits",
        inputs,
        ["Y"],
        name=f"tail_N{n}_K{k}",
        domain="com.microsoft",
        N=n,
        K=k,
        bits=4,
        block_size=BLOCK_SIZE,
    )
    graph = helper.make_graph(
        [node],
        "unaligned4",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT16, ["M", k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["M", n])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if isinstance(target, str):
        return ort.InferenceSession(
            model.SerializeToString(), options, providers=[(target, {"enable_cuda_graph": "0"})]
        )
    options.add_provider_for_devices([target], {"enable_cuda_graph": "0"})
    return ort.InferenceSession(model.SerializeToString(), options)


def _pattern(name, rng, n, k):
    if name == "constant":
        return (
            np.full((n, k // BLOCK_SIZE, 16), 0x99, np.uint8),
            np.full((n, k // BLOCK_SIZE), 0.03125, np.float16),
            np.full((8, k), 0.125, np.float16),
        )
    if name == "signed_zero":
        return (
            rng.integers(0, 256, (n, k // BLOCK_SIZE, 16), dtype=np.uint8),
            np.resize(np.array([0.0, -0.0, 0.0625, -0.0625], np.float16), (n, k // BLOCK_SIZE)),
            np.resize(np.array([0.0, -0.0], np.float16), (8, k)),
        )
    if name == "adversarial":
        return (
            np.resize(np.array([0x00, 0x0F, 0xF0, 0xFF, 0x88, 0x17, 0xE9], np.uint8), (n, k // BLOCK_SIZE, 16)),
            np.resize(np.array([0.0625, -0.0625, 2**-14, 2**-24, 0.0, -0.0], np.float16), (n, k // BLOCK_SIZE)),
            np.resize(np.array([1, -1, 8, -8, 2**-14, -(2**-14), 2**-24, -0.0], np.float16), (8, k)),
        )
    return (
        rng.integers(0, 256, (n, k // BLOCK_SIZE, 16), dtype=np.uint8),
        rng.uniform(0.001, 0.02, (n, k // BLOCK_SIZE)).astype(np.float16),
        rng.normal(0, 0.02, (8, k)).astype(np.float16),
    )


def _worker():
    target = _cuda_session_options()
    if target is None:
        print("SKIP")
        return
    for name in ("ORT_FPA_INTB_GEMM", "ORT_MATMULNBITS_DEQUANT_CACHE", "ORT_MATMULNBITS_FORCE_CHUNKED"):
        os.environ[name] = "0"

    rng = np.random.default_rng(34812)
    comparisons = 0
    for n in (4, 12, 324):
        for k in (32, 320, 10240):
            for pattern in ("constant", "signed_zero", "adversarial", "realish"):
                weights, scales, a = _pattern(pattern, rng, n, k)
                # Padded output rows are independent poison; the valid outputs must not depend on them.
                padded_weights = np.concatenate([weights, np.full((4, k // BLOCK_SIZE, 16), 0xFE, np.uint8)])
                padded_scales = np.concatenate([scales, np.full((4, k // BLOCK_SIZE), -0.375, np.float16)])
                tail = _create_session(target, n, k, weights, scales, True)
                reference = _create_session(target, n + 4, k, padded_weights, padded_scales, False)
                for m in (1, 2, 3, 4):
                    feed = {"A": a[:m].copy()}
                    actual = tail.run(["Y"], feed)[0]
                    expected = reference.run(["Y"], feed)[0][:, :n].copy()
                    if actual.tobytes() != expected.tobytes():
                        raise AssertionError(f"bit mismatch M={m} N={n} K={k} pattern={pattern}")
                    comparisons += 1
                del tail, reference

    n, k = 12, 320
    weights, scales, a_excl = _pattern("realish", rng, n, k)
    scale_only_session = _create_session(target, n, k, weights, scales, False)
    scale_only_m2 = scale_only_session.run(["Y"], {"A": a_excl[:2].copy()})[0].tobytes()
    del scale_only_session
    for label, kwargs, m in (
        ("M5", {}, 5),
        ("M8", {}, 8),
        ("M9", {}, 9),
        ("bias", {"bias": True}, 2),
        ("zero_points", {"zero_points": True}, 2),
        ("g_idx", {"g_idx": True}, 2),
    ):
        feed_a = np.resize(a_excl, (m, k)).copy()
        outputs = []
        for enabled in (False, True):
            session = _create_session(target, n, k, weights, scales, enabled, **kwargs)
            outputs.append(session.run(["Y"], {"A": feed_a})[0].tobytes())
            del session
        if outputs[0] != outputs[1]:
            raise AssertionError(f"{label}: selector changed an ineligible configuration")
        if kwargs and outputs[0] == scale_only_m2:
            raise AssertionError(f"{label}: exclusion input did not change output vs scale-only baseline")
    print(f"COMPARISONS={comparisons}")


class TestMatMulNBitsUnaligned4(unittest.TestCase):
    def test_tail_dispatch_matches_padded_aligned_kernels(self):
        # Run in a subprocess so the dispatch receipts written to stderr can be inspected.
        env = os.environ.copy()
        package_root = str(Path(ort.__file__).resolve().parent.parent)
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, pythonpath]))
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-6000:])
        if "SKIP" in result.stdout:
            self.skipTest("CUDA execution provider is not available")
        self.assertIn("COMPARISONS=144", result.stdout)
        lines = [line for line in result.stderr.splitlines() if line.startswith(LOG_PREFIX)]
        self.assertTrue(any("branch=M1_SHARED_TAIL" in line for line in lines))
        self.assertTrue(any("branch=BATCHED_REGISTER_TAIL" in line for line in lines))
        # Only the 36 tail-enabled sessions may report a receipt, at most once per branch each.
        self.assertLessEqual(len(lines), 72)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _worker()
    else:
        unittest.main()
