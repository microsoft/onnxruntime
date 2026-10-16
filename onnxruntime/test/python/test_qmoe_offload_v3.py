#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Unit tests for QMoEOffload v3 design (F1-F12) and NVFP4 CPU/GPU offload parity."""

from __future__ import annotations

import ctypes
import gc
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


ORT_ROOT = Path("/home/tianlei/git/ort-qmoe-offload-v3")
GENAI_ROOT = Path("/home/tianlei/git/genai-qmoe-offload-v3")
PROVIDER_LIB = os.environ.get(
    "QWEN_PROVIDER",
    "/home/tianlei/git/artifacts/qwen38_qmoe_offload_v3_20261014/env_v3/ort_home/lib/libonnxruntime_providers_cuda.so",
)


def ensure_plugin_registered() -> str:
    ep_name = "CUDAExecutionProvider"
    try:
        ort.register_execution_provider_library(ep_name, PROVIDER_LIB)
    except Exception as exc:
        if "already registered" not in str(exc).lower():
            raise
    return ep_name


def make_plugin_session(model_or_path, device_limit_bytes: int = 0) -> ort.InferenceSession:
    ep_name = ensure_plugin_registered()
    devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name and d.device.type == ort.OrtHardwareDeviceType.GPU]
    assert devices, f"No GPU devices found for {ep_name}"
    devices.sort(key=lambda d: int(d.ep_options.get("device_id", d.ep_metadata.get("cuda_device_id", "0"))))
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.add_session_config_entry("ep.cuda.enable_cuda_graph", "0")
    if device_limit_bytes > 0:
        opts.add_session_config_entry("ep.cuda.gpu_device_mem_limit", str(device_limit_bytes))
    opts.add_provider_for_devices([devices[0]], {})
    if isinstance(model_or_path, (str, Path)):
        return ort.InferenceSession(str(model_or_path), sess_options=opts)
    return ort.InferenceSession(model_or_path.SerializeToString(), sess_options=opts)


def float32_to_bf16_uint16(arr: np.ndarray) -> np.ndarray:
    u32 = arr.astype(np.float32).view(np.uint32)
    lsb = (u32 >> 16) & 1
    rounded = u32 + 0x7FFF + lsb
    return (rounded >> 16).astype(np.uint16)


def bf16_uint16_to_float32(arr: np.ndarray) -> np.ndarray:
    u32 = arr.astype(np.uint32) << 16
    return u32.view(np.float32)


@unittest.skipIf(sys.platform == "win32", "TestQMoEOffloadV3 requires POSIX g++ toolchain and Linux plugin library.")
class TestQMoEOffloadV3(unittest.TestCase):
    def test_f12_checked_arithmetic_and_alignment_lease(self):
        """Verifies F12 standalone checked arithmetic and lease alignment tracking in qmoe_shared_layout."""
        cpp_src = r"""
#include <cassert>
#include <cstdint>
#include <limits>
#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"
using namespace onnxruntime::contrib::qmoe_shared;
int main() {
    ExpertBlockLayout layout{};
    assert(MakeExpertBlockLayout(2560, 640, &layout));
    assert(layout.stride == 2772480);
    // Reject negative, zero, unaligned, and overflow dimensions without modifying layout.
    ExpertBlockLayout sentinel = layout;
    assert(!MakeExpertBlockLayout(0, 640, &sentinel));
    assert(!MakeExpertBlockLayout(-64, 640, &sentinel));
    assert(!MakeExpertBlockLayout(2560, (1LL << 31), &sentinel));
    assert(!MakeExpertBlockLayout((1LL << 31), 640, &sentinel));
    assert(sentinel.stride == 2772480);

    size_t total = 0;
    assert(ValidateExpertStorageRange(512, layout.stride, 0, 512ULL * layout.stride, &total));
    assert(total == 512ULL * 2772480ULL);
    assert(!ValidateExpertStorageRange(512, std::numeric_limits<size_t>::max() / 100, 0, 0, &total));
    assert(!ValidateExpertStorageRange(512, layout.stride, std::numeric_limits<uint64_t>::max() - 10, 512ULL * layout.stride, &total));

    alignas(256) uint8_t buf[512]{};
    ExpertBlockLease lease{};
    assert(MakeExpertBlockLease(buf, 256, 2, &lease));
    assert(lease.alignment >= 256);
    // 2-byte aligned but not 64-byte aligned is valid for 2-byte typed access, rejected for 64-byte requirement.
    assert(MakeExpertBlockLease(buf + 2, 256, 2, &lease));
    assert(lease.alignment == 2);
    assert(!MakeExpertBlockLease(buf + 2, 256, 64, &lease));
    // Odd address rejected for 2-byte typed access.
    assert(!MakeExpertBlockLease(buf + 1, 256, 2, &lease));
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "test_f12"
            subprocess.run(
                [
                    "g++",
                    "-O2",
                    "-std=c++20",
                    f"-I{ORT_ROOT / 'onnxruntime'}",
                    f"-I{ORT_ROOT / 'include/onnxruntime'}",
                    "-x",
                    "c++",
                    "-",
                    str(ORT_ROOT / "onnxruntime/contrib_ops/cpu/moe/qmoe_shared_layout.cc"),
                    "-o",
                    str(bin_path),
                ],
                input=cpp_src.encode(),
                check=True,
            )
            subprocess.run([str(bin_path)], check=True)

    def test_f4_budget_ledger_discharge_and_lock_free_freeing(self):
        """Verifies F4 DevicePhysicalLedger reconfiguration after discharge and Freeing state."""
        cpp_src = r"""
#include <cassert>
#include <memory>
#include "core/providers/cuda/cuda_device_memory_budget.h"
using namespace onnxruntime;
int main() {
    auto ledger = std::make_shared<CudaDeviceMemoryBudget>();
    {
        SessionBudgetAccount acc1(ledger, 1024);
        ledger->Configure(2048);
        auto res = acc1.Reserve(512);
        res.Commit();
        assert(ledger->GetStats().committed == 512);
        // Transition to Freeing and complete free successfully.
        acc1.BeginFree(512);
        assert(ledger->GetStats().freeing == 512);
        acc1.FinishFree(512, true);
        assert(ledger->GetStats().committed == 0);
        assert(ledger->GetStats().freeing == 0);
    }
    // After acc1 is destroyed and all allocations are discharged, ledger resets and accepts a new limit!
    assert(ledger->GetStats().limit == std::numeric_limits<size_t>::max());
    {
        SessionBudgetAccount acc2(ledger, 4096);
        ledger->Configure(4096);
        assert(ledger->GetStats().limit == 4096);
        // Simulate a failed driver free: charge is quarantined rather than leaked.
        auto res = acc2.Reserve(256);
        res.Commit();
        acc2.BeginFree(256);
        acc2.FinishFree(256, false);
        assert(ledger->GetStats().quarantined == 256);
        assert(ledger->GetStats().committed == 256);
    }
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "test_f4"
            subprocess.run(
                [
                    "g++",
                    "-O2",
                    "-std=c++20",
                    f"-I{ORT_ROOT / 'onnxruntime'}",
                    f"-I{ORT_ROOT / 'include/onnxruntime'}",
                    "-x",
                    "c++",
                    "-",
                    "-o",
                    str(bin_path),
                ],
                input=cpp_src.encode(),
                check=True,
            )
            subprocess.run([str(bin_path)], check=True)

    def test_f1_f2_same_request_error_delivery_on_final_async_call(self):
        """Verifies F1/F2/P0: an invalid phase on the final async call fails the owning Run() immediately."""
        sys.path.insert(0, str(ORT_ROOT / "onnxruntime/test/python/transformers"))
        import qmoe_shared_layout_ref as ref

        rng = np.random.default_rng(7)
        experts, hidden, inter, top_k = 8, 128, 64, 2
        blocks_list = []
        biases_list = []
        for _ in range(experts):
            w1 = rng.integers(0, 16, size=(2 * inter, hidden), dtype=np.uint8)
            s1 = (rng.uniform(0.01, 0.05, size=(2 * inter, hidden // 32))).astype(np.float16)
            b1 = np.zeros((2 * inter,), dtype=np.float16)
            w2 = rng.integers(0, 16, size=(hidden, inter), dtype=np.uint8)
            s2 = (rng.uniform(0.01, 0.05, size=(hidden, inter // 32))).astype(np.float16)
            b2 = (rng.uniform(-0.02, 0.02, size=(hidden,))).astype(np.float16)
            blocks_list.append(ref.assemble_block(hidden, inter, w1, s1, b1, w2, s2, b2))
            biases_list.append(b2)
        blocks = np.stack(blocks_list, axis=0)
        bias = np.stack(biases_list, axis=0)

        # Build a v3 compact 5-input shared model (inputs: X, R, expert_blocks, fc2_bias, phase)
        node = helper.make_node(
            "QMoEOffload",
            inputs=["X", "R", "expert_blocks", "fc2_bias", "phase"],
            outputs=["Y"],
            domain="com.microsoft",
            name="qmoe_v3_node",
            k=top_k,
            normalize_routing_weights=1,
            layout_version=1,
            layout_id="ort.qmoe.w4a16.block32.shared.v1",
            weights_prepacked=1,
            hidden_size=hidden,
            inter_size=inter,
            layer_index=0,
        )
        graph = helper.make_graph(
            [node],
            "v3_compact_graph",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT16, ["T", hidden]),
                helper.make_tensor_value_info("R", TensorProto.FLOAT16, ["T", experts]),
                helper.make_tensor_value_info("phase", TensorProto.INT8, ["T"]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["T", hidden])],
            [
                numpy_helper.from_array(blocks, name="expert_blocks"),
                numpy_helper.from_array(bias, name="fc2_bias"),
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        )

        old_env = {
            k: os.environ.get(k)
            for k in (
                "ORT_QMOE_SHARED_ASYNC",
                "ORT_QMOE_SHARED_PINNED_REUSE",
                "ORT_QMOE_SHARED_RESIDENT_SLOTS",
                "ORT_QMOE_SHARED_AUX_STREAM",
                "ORT_QMOE_SHARED_SPARE_SLOTS",
            )
        }
        try:
            os.environ["ORT_QMOE_SHARED_ASYNC"] = "1"
            os.environ["ORT_QMOE_SHARED_PINNED_REUSE"] = "1"
            os.environ["ORT_QMOE_SHARED_RESIDENT_SLOTS"] = "4"
            os.environ["ORT_QMOE_SHARED_AUX_STREAM"] = "1"
            os.environ["ORT_QMOE_SHARED_SPARE_SLOTS"] = "2"

            sess = make_plugin_session(model)
            x = rng.standard_normal((2, hidden)).astype(np.float16)
            r = rng.standard_normal((2, experts)).astype(np.float16)
            valid_phase = np.array([1, 1], dtype=np.int8)
            y = sess.run(None, {"X": x, "R": r, "phase": valid_phase})[0]
            self.assertEqual(y.shape, (2, hidden))

            # Now pass an invalid phase (7) on the FINAL call: OnRunEnd must join and fail THIS Run() call!
            bad_phase = np.array([1, 7], dtype=np.int8)
            with self.assertRaises(Exception) as ctx:
                sess.run(None, {"X": x, "R": r, "phase": bad_phase})
            self.assertIn("Invalid token_phase", str(ctx.exception))
            del sess
            gc.collect()
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_nvfp4_qmoe_offload_matches_gpu_qmoe_across_all_modes(self):
        """Verifies NVFP4 QMoEOffload (zero-cache CPU, resident hybrid, async aux, and prefill streaming) against GPU QMoE."""
        rng = np.random.default_rng(2026)
        experts, hidden, inter, top_k = 16, 2560, 640, 10
        fc1_w = rng.integers(0, 256, size=(experts, hidden, inter), dtype=np.uint8)
        fc1_s = rng.integers(0x34, 0x44, size=(experts, 2 * inter, hidden // 16), dtype=np.uint8)
        fc2_w = rng.integers(0, 256, size=(experts, inter, hidden // 2), dtype=np.uint8)
        fc2_s = rng.integers(0x34, 0x44, size=(experts, hidden, inter // 16), dtype=np.uint8)
        fc1_g = rng.uniform(0.01, 0.03, size=(experts,)).astype(np.float32)
        fc2_g = rng.uniform(0.01, 0.03, size=(experts,)).astype(np.float32)

        def build_nvfp4_graph(op_type: str):
            inputs = [
                "X",
                "R",
                "fc1_w",
                "fc1_s",
                "",
                "fc2_w",
                "fc2_s",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "fc1_g",
                "fc2_g",
            ]
            kwargs = dict(
                activation_type="swiglu",
                normalize_routing_weights=1,
                k=top_k,
                expert_weight_bits=4,
                block_size=16,
                swiglu_fusion=1,
                quant_type="nvfp4",
                weights_prepacked=1,
            )
            if op_type == "QMoEOffload":
                kwargs.update(
                    layer_index=0,
                    hidden_size=hidden,
                    inter_size=inter,
                    token_tile_size=8,
                    gpu_working_bytes=128 << 20,
                    pinned_staging_bytes=16 << 20,
                )
            node = helper.make_node(op_type, inputs=inputs, outputs=["Y"], domain="com.microsoft", **kwargs)
            inits = [
                numpy_helper.from_array(fc1_w, name="fc1_w"),
                helper.make_tensor("fc1_s", TensorProto.FLOAT8E4M3FN, fc1_s.shape, fc1_s.tobytes(), raw=True),
                numpy_helper.from_array(fc2_w, name="fc2_w"),
                helper.make_tensor("fc2_s", TensorProto.FLOAT8E4M3FN, fc2_s.shape, fc2_s.tobytes(), raw=True),
                numpy_helper.from_array(fc1_g, name="fc1_g"),
                numpy_helper.from_array(fc2_g, name="fc2_g"),
            ]
            graph = helper.make_graph(
                [node],
                f"{op_type}_nvfp4",
                [
                    helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [1, "S", hidden]),
                    helper.make_tensor_value_info("R", TensorProto.BFLOAT16, ["S", experts]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [1, "S", hidden])],
                inits,
            )
            return helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
            )

        ref_model = build_nvfp4_graph("QMoE")
        off_model = build_nvfp4_graph("QMoEOffload")
        ref_sess = make_plugin_session(ref_model)

        for S in (2, 12):
            x_f32 = (rng.standard_normal((1, S, hidden)) * 0.5).astype(np.float32)
            r_f32 = rng.standard_normal((S, experts)).astype(np.float32)
            x_bf16 = float32_to_bf16_uint16(x_f32)
            r_bf16 = float32_to_bf16_uint16(r_f32)

            ort_x = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(x_bf16, TensorProto.BFLOAT16)
            ort_r = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r_bf16, TensorProto.BFLOAT16)
            ref_out_ort = ref_sess.run_with_ort_values(["Y"], {"X": ort_x, "R": ort_r})[0]
            ref_out = bf16_uint16_to_float32(
                np.frombuffer(
                    ctypes.string_at(ref_out_ort.data_ptr(), x_bf16.size * 2), dtype=np.uint16
                ).reshape(x_bf16.shape)
            )

            configs = [
                (
                    "all_gpu_resident",
                    {
                        "ORT_QMOE_SHARED_RESIDENT_SLOTS": "16",
                        "ORT_QMOE_SHARED_SPARE_SLOTS": "0",
                        "ORT_QMOE_SHARED_ASYNC": "0",
                        "ORT_QMOE_SHARED_STREAM_ROWS": "0",
                    },
                ),
                (
                    "zero_cache_cpu",
                    {
                        "ORT_QMOE_SHARED_RESIDENT_SLOTS": "0",
                        "ORT_QMOE_SHARED_SPARE_SLOTS": "0",
                        "ORT_QMOE_SHARED_ASYNC": "0",
                        "ORT_QMOE_SHARED_STREAM_ROWS": "0",
                    },
                ),
                (
                    "resident_sync",
                    {
                        "ORT_QMOE_SHARED_RESIDENT_SLOTS": "6",
                        "ORT_QMOE_SHARED_SPARE_SLOTS": "2",
                        "ORT_QMOE_SHARED_ASYNC": "0",
                        "ORT_QMOE_SHARED_STREAM_ROWS": "0",
                    },
                ),
                (
                    "resident_async_aux",
                    {
                        "ORT_QMOE_SHARED_RESIDENT_SLOTS": "6",
                        "ORT_QMOE_SHARED_SPARE_SLOTS": "2",
                        "ORT_QMOE_SHARED_ASYNC": "1",
                        "ORT_QMOE_SHARED_PINNED_REUSE": "1",
                        "ORT_QMOE_SHARED_AUX_STREAM": "1",
                        "ORT_QMOE_SHARED_STREAM_ROWS": "0",
                    },
                ),
                (
                    "stream_prefill",
                    {
                        "ORT_QMOE_SHARED_RESIDENT_SLOTS": "4",
                        "ORT_QMOE_SHARED_PIN_EXPERTS": "1",
                        "ORT_QMOE_SHARED_STREAM_ROWS": "16",
                        "ORT_QMOE_SHARED_STREAM_MIN_ROWS": "2",
                        "ORT_QMOE_SHARED_STREAM_WAVE": "4",
                        "ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE": "1",
                        "ORT_QMOE_SHARED_ASYNC": "0",
                    },
                ),
            ]
            for name, env_vars in configs:
                saved = {k: os.environ.get(k) for k in env_vars}
                sess = None
                try:
                    os.environ.update(env_vars)
                    sess = make_plugin_session(off_model)
                    out_ort = sess.run_with_ort_values(["Y"], {"X": ort_x, "R": ort_r})[0]
                    got_out = bf16_uint16_to_float32(
                        np.frombuffer(
                            ctypes.string_at(out_ort.data_ptr(), x_bf16.size * 2), dtype=np.uint16
                        ).reshape(x_bf16.shape)
                    )
                    rel_l2 = np.linalg.norm(got_out - ref_out) / np.linalg.norm(ref_out)
                    if name == "all_gpu_resident" and S == 2:
                        self.assertEqual(rel_l2, 0.0, msg=f"mode={name} S={S} expected exact bit parity")
                    elif S <= 6:
                        self.assertLess(rel_l2, 5e-4, msg=f"mode={name} S={S} rel_l2={rel_l2}")
                    else:
                        # At S=12 (expanded=120 > 64), stock QMoE switches to dequant + CUTLASS grouped GEMM
                        self.assertLess(rel_l2, 1e-2, msg=f"mode={name} S={S} rel_l2={rel_l2}")
                finally:
                    if sess is not None:
                        del sess
                        gc.collect()
                    for k, v in saved.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
        del ref_sess
        gc.collect()

    def test_mxfp4_qmoe_offload_matches_cpu_and_gpu_across_modes(self):
        """Verifies MXFP4 (block_size=32, Float8E8M0 scales) QMoEOffload across CPU, resident hybrid, async, and streaming modes."""
        rng = np.random.default_rng(2027)
        experts, hidden, inter, top_k = 16, 256, 128, 4
        fc1_w = rng.integers(0, 256, size=(experts, hidden, inter), dtype=np.uint8)
        fc1_s = rng.integers(123, 127, size=(experts, 2 * inter, hidden // 32), dtype=np.uint8)
        fc2_w = rng.integers(0, 256, size=(experts, inter, hidden // 2), dtype=np.uint8)
        fc2_s = rng.integers(123, 127, size=(experts, hidden, inter // 32), dtype=np.uint8)
        fc1_g = rng.uniform(0.5, 1.5, size=(experts,)).astype(np.float32)
        fc2_g = rng.uniform(0.5, 1.5, size=(experts,)).astype(np.float32)

        inputs = [
            "X",
            "R",
            "fc1_w",
            "fc1_s",
            "",
            "fc2_w",
            "fc2_s",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "fc1_g",
            "fc2_g",
        ]
        node = helper.make_node(
            "QMoEOffload",
            inputs=inputs,
            outputs=["Y"],
            domain="com.microsoft",
            activation_type="swiglu",
            normalize_routing_weights=1,
            k=top_k,
            expert_weight_bits=4,
            block_size=32,
            swiglu_fusion=1,
            quant_type="mxfp4",
            weights_prepacked=1,
            layer_index=0,
            hidden_size=hidden,
            inter_size=inter,
            token_tile_size=8,
            gpu_working_bytes=64 << 20,
            pinned_staging_bytes=8 << 20,
        )
        inits = [
            numpy_helper.from_array(fc1_w, name="fc1_w"),
            helper.make_tensor("fc1_s", TensorProto.FLOAT8E8M0, fc1_s.shape, fc1_s.tobytes(), raw=True),
            numpy_helper.from_array(fc2_w, name="fc2_w"),
            helper.make_tensor("fc2_s", TensorProto.FLOAT8E8M0, fc2_s.shape, fc2_s.tobytes(), raw=True),
            numpy_helper.from_array(fc1_g, name="fc1_g"),
            numpy_helper.from_array(fc2_g, name="fc2_g"),
        ]
        graph = helper.make_graph(
            [node],
            "QMoEOffload_mxfp4",
            [
                helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [1, "S", hidden]),
                helper.make_tensor_value_info("R", TensorProto.BFLOAT16, ["S", experts]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [1, "S", hidden])],
            inits,
        )
        off_model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        )

        S = 3
        x_f32 = (rng.standard_normal((1, S, hidden)) * 0.5).astype(np.float32)
        r_f32 = rng.standard_normal((S, experts)).astype(np.float32)
        x_bf16 = float32_to_bf16_uint16(x_f32)
        r_bf16 = float32_to_bf16_uint16(r_f32)
        ort_x = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(x_bf16, TensorProto.BFLOAT16)
        ort_r = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r_bf16, TensorProto.BFLOAT16)

        configs = [
            (
                "all_gpu_resident",
                {
                    "ORT_QMOE_SHARED_RESIDENT_SLOTS": "16",
                    "ORT_QMOE_SHARED_SPARE_SLOTS": "0",
                    "ORT_QMOE_SHARED_ASYNC": "0",
                    "ORT_QMOE_SHARED_STREAM_ROWS": "0",
                },
            ),
            (
                "zero_cache_cpu",
                {
                    "ORT_QMOE_SHARED_RESIDENT_SLOTS": "0",
                    "ORT_QMOE_SHARED_SPARE_SLOTS": "0",
                    "ORT_QMOE_SHARED_ASYNC": "0",
                    "ORT_QMOE_SHARED_STREAM_ROWS": "0",
                },
            ),
            (
                "resident_async_aux",
                {
                    "ORT_QMOE_SHARED_RESIDENT_SLOTS": "6",
                    "ORT_QMOE_SHARED_SPARE_SLOTS": "2",
                    "ORT_QMOE_SHARED_ASYNC": "1",
                    "ORT_QMOE_SHARED_PINNED_REUSE": "1",
                    "ORT_QMOE_SHARED_AUX_STREAM": "1",
                    "ORT_QMOE_SHARED_STREAM_ROWS": "0",
                },
            ),
            (
                "stream_prefill",
                {
                    "ORT_QMOE_SHARED_RESIDENT_SLOTS": "4",
                    "ORT_QMOE_SHARED_PIN_EXPERTS": "1",
                    "ORT_QMOE_SHARED_STREAM_ROWS": "16",
                    "ORT_QMOE_SHARED_STREAM_MIN_ROWS": "1",
                    "ORT_QMOE_SHARED_STREAM_WAVE": "4",
                    "ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE": "1",
                    "ORT_QMOE_SHARED_ASYNC": "0",
                },
            ),
        ]
        outputs = {}
        for name, env_vars in configs:
            saved = {k: os.environ.get(k) for k in env_vars}
            sess = None
            try:
                os.environ.update(env_vars)
                sess = make_plugin_session(off_model)
                out_ort = sess.run_with_ort_values(["Y"], {"X": ort_x, "R": ort_r})[0]
                outputs[name] = bf16_uint16_to_float32(
                    np.frombuffer(
                        ctypes.string_at(out_ort.data_ptr(), x_bf16.size * 2), dtype=np.uint16
                    ).reshape(x_bf16.shape)
                )
            finally:
                if sess is not None:
                    del sess
                    gc.collect()
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        ref_out = outputs["all_gpu_resident"]
        for name in ("zero_cache_cpu", "resident_async_aux", "stream_prefill"):
            rel_l2 = np.linalg.norm(outputs[name] - ref_out) / np.linalg.norm(ref_out)
            self.assertLess(rel_l2, 5e-4, msg=f"MXFP4 mode={name} rel_l2={rel_l2}")

    def test_phase1_run_outcome_and_disable_sync_rejection(self):
        """Verifies Phase 1: disable_synchronize_execution_providers=1 is rejected when offload is active."""
        sys.path.insert(0, str(ORT_ROOT / "onnxruntime/test/python/transformers"))
        import qmoe_shared_layout_ref as ref

        rng = np.random.default_rng(101)
        experts, hidden, inter, top_k = 8, 128, 64, 2
        blocks_list, biases_list = [], []
        for _ in range(experts):
            w1 = rng.integers(0, 16, size=(2 * inter, hidden), dtype=np.uint8)
            s1 = (rng.uniform(0.01, 0.05, size=(2 * inter, hidden // 32))).astype(np.float16)
            b1 = np.zeros((2 * inter,), dtype=np.float16)
            w2 = rng.integers(0, 16, size=(hidden, inter), dtype=np.uint8)
            s2 = (rng.uniform(0.01, 0.05, size=(hidden, inter // 32))).astype(np.float16)
            b2 = np.zeros((hidden,), dtype=np.float16)
            blocks_list.append(ref.assemble_block(hidden, inter, w1, s1, b1, w2, s2, b2))
            biases_list.append(b2)
        blocks = np.stack(blocks_list, axis=0)
        bias = np.stack(biases_list, axis=0)

        node = helper.make_node(
            "QMoEOffload",
            inputs=["X", "R", "expert_blocks", "fc2_bias"],
            outputs=["Y"],
            domain="com.microsoft",
            k=top_k,
            normalize_routing_weights=1,
            layout_version=1,
            layout_id="ort.qmoe.w4a16.block32.shared.v1",
            weights_prepacked=1,
            hidden_size=hidden,
            inter_size=inter,
            layer_index=0,
        )
        graph = helper.make_graph(
            [node],
            "p1_graph",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT16, ["T", hidden]),
                helper.make_tensor_value_info("R", TensorProto.FLOAT16, ["T", experts]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["T", hidden])],
            [
                numpy_helper.from_array(blocks, name="expert_blocks"),
                numpy_helper.from_array(bias, name="fc2_bias"),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]
        )
        sess = make_plugin_session(model)
        x = rng.standard_normal((2, hidden)).astype(np.float16)
        r = rng.standard_normal((2, experts)).astype(np.float16)

        ro = ort.RunOptions()
        ro.add_run_config_entry("disable_synchronize_execution_providers", "1")
        old_strict = os.environ.get("ORT_QMOE_REJECT_DISABLE_SYNC")
        try:
            os.environ["ORT_QMOE_REJECT_DISABLE_SYNC"] = "1"
            with self.assertRaises(Exception) as ctx:
                sess.run(None, {"X": x, "R": r}, run_options=ro)
            self.assertIn("disable_synchronize_execution_providers", str(ctx.exception))
        finally:
            if old_strict is None:
                os.environ.pop("ORT_QMOE_REJECT_DISABLE_SYNC", None)
            else:
                os.environ["ORT_QMOE_REJECT_DISABLE_SYNC"] = old_strict
        del sess
        gc.collect()

    def test_phase2_budget_group_pinned_ledger_and_concurrent_unequal_sessions(self):
        """Verifies Phase 2: BudgetGroup, HostPinnedBudgetLedger deduplication, and concurrent unequal session limits."""
        cpp_src = r"""
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>
#include "core/providers/cuda/cuda_device_memory_budget.h"
#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"
using namespace onnxruntime;
using namespace onnxruntime::contrib::qmoe_shared;
int main() {
    auto phys = std::make_shared<CudaDeviceMemoryBudget>();
    phys->ConfigurePhysicalCeiling(8192);
    auto grp = std::make_shared<BudgetGroup>(phys, 4096);
    SessionBudgetAccount target_acc(phys, 3072, grp);
    SessionBudgetAccount head_acc(phys, 2048, grp);

    auto r1 = target_acc.Reserve(2500);
    r1.Commit();
    // Group limit is 4096 (1596 remaining), so head_acc cannot reserve 1700 even though its own session limit is 2048!
    bool group_blocked = false;
    try {
        auto r2 = head_acc.Reserve(1700);
    } catch (const std::runtime_error&) {
        group_blocked = true;
    }
    assert(group_blocked);
    auto r2 = head_acc.Reserve(1500);
    r2.Commit();
    assert(grp->GetGroupStats().committed == 4000);
    assert(phys->GetStats().committed == 4000);

    target_acc.BeginFree(2500);
    target_acc.FinishFree(2500, true);
    head_acc.BeginFree(1500);
    head_acc.FinishFree(1500, true);

    // HostPinnedBudgetLedger deduplicates shared pointer ranges across target and head sessions.
    auto pinned = GetCudaHostPinnedMemoryBudget();
    const size_t base_committed = pinned->GetStats().committed;
    alignas(256) uint8_t fake_weights[4096]{};
    assert(pinned->RegisterHostRange(fake_weights, sizeof(fake_weights)));
    assert(pinned->GetStats().committed == base_committed + sizeof(fake_weights));
    // Second registration of the same pointer increments refcount without double-charging!
    assert(!pinned->RegisterHostRange(fake_weights, sizeof(fake_weights)));
    assert(pinned->GetStats().committed == base_committed + sizeof(fake_weights));
    assert(!pinned->UnregisterHostRange(fake_weights, true));
    assert(pinned->GetStats().committed == base_committed + sizeof(fake_weights));
    assert(pinned->UnregisterHostRange(fake_weights, true));
    assert(pinned->GetStats().committed == base_committed);

    // TensorExpertStore lease verification.
    ExpertBlockLayout layout{};
    assert(MakeExpertBlockLayout(128, 64, &layout));
    std::vector<uint8_t> store_buf(4 * layout.stride, 7);
    TensorExpertStore store;
    assert(store.Initialize(store_buf.data(), store_buf.size(), 4, layout.stride, 2));
    assert(store.ContentId() != 0);
    ExpertBlockLease l0{};
    assert(store.AcquireReady(0, &l0));
    assert(l0.data == store_buf.data() && l0.length == layout.stride);
    assert(!store.AcquireReady(4, &l0));
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "test_p2"
            subprocess.run(
                [
                    "g++",
                    "-O2",
                    "-std=c++20",
                    f"-I{ORT_ROOT / 'onnxruntime'}",
                    f"-I{ORT_ROOT / 'include/onnxruntime'}",
                    "-x",
                    "c++",
                    "-",
                    str(ORT_ROOT / "onnxruntime/contrib_ops/cpu/moe/qmoe_shared_layout.cc"),
                    "-o",
                    str(bin_path),
                ],
                input=cpp_src.encode(),
                check=True,
            )
            subprocess.run([str(bin_path)], check=True)

        # Also test two concurrent plugin sessions on the same GPU with unequal gpu_device_mem_limit values!
        rng = np.random.default_rng(202)
        experts, hidden, inter, top_k = 16, 256, 128, 4
        fc1_w = rng.integers(0, 256, size=(experts, hidden, inter), dtype=np.uint8)
        fc1_s = rng.integers(0x34, 0x44, size=(experts, 2 * inter, hidden // 16), dtype=np.uint8)
        fc2_w = rng.integers(0, 256, size=(experts, inter, hidden // 2), dtype=np.uint8)
        fc2_s = rng.integers(0x34, 0x44, size=(experts, hidden, inter // 16), dtype=np.uint8)
        fc1_g = np.ones((experts,), dtype=np.float32)
        fc2_g = np.ones((experts,), dtype=np.float32)

        inputs = ["X", "R", "fc1_w", "fc1_s", "", "fc2_w", "fc2_s", "", "", "", "", "", "", "", "", "fc1_g", "fc2_g"]
        node = helper.make_node(
            "QMoEOffload",
            inputs=inputs,
            outputs=["Y"],
            domain="com.microsoft",
            activation_type="swiglu",
            normalize_routing_weights=1,
            k=top_k,
            expert_weight_bits=4,
            block_size=16,
            swiglu_fusion=1,
            quant_type="nvfp4",
            weights_prepacked=1,
            layer_index=0,
            hidden_size=hidden,
            inter_size=inter,
            resident_slots=4,
            token_tile_size=4,
            gpu_working_bytes=16 << 20,
            pinned_staging_bytes=4 << 20,
        )
        inits = [
            numpy_helper.from_array(fc1_w, name="fc1_w"),
            helper.make_tensor("fc1_s", TensorProto.FLOAT8E4M3FN, fc1_s.shape, fc1_s.tobytes(), raw=True),
            numpy_helper.from_array(fc2_w, name="fc2_w"),
            helper.make_tensor("fc2_s", TensorProto.FLOAT8E4M3FN, fc2_s.shape, fc2_s.tobytes(), raw=True),
            numpy_helper.from_array(fc1_g, name="fc1_g"),
            numpy_helper.from_array(fc2_g, name="fc2_g"),
        ]
        graph = helper.make_graph(
            [node],
            "p2_concurrent",
            [
                helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [1, "S", hidden]),
                helper.make_tensor_value_info("R", TensorProto.BFLOAT16, ["S", experts]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [1, "S", hidden])],
            inits,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]
        )

        # Concurrent sessions on the same GPU with different limits (64 MiB and 96 MiB) both succeed!
        sess_a = make_plugin_session(model, device_limit_bytes=64 << 20)
        sess_b = make_plugin_session(model, device_limit_bytes=96 << 20)
        del sess_a, sess_b
        gc.collect()

    def test_phase3_fast_paths_and_deterministic_replay(self):
        """Verifies Phase 3: statically full (active_residents == E) and empty (active_residents == 0) fast paths + deterministic replay."""
        rng = np.random.default_rng(303)
        experts, hidden, inter, top_k = 8, 256, 128, 2
        fc1_w = rng.integers(0, 256, size=(experts, hidden, inter), dtype=np.uint8)
        fc1_s = rng.integers(0x34, 0x44, size=(experts, 2 * inter, hidden // 16), dtype=np.uint8)
        fc2_w = rng.integers(0, 256, size=(experts, inter, hidden // 2), dtype=np.uint8)
        fc2_s = rng.integers(0x34, 0x44, size=(experts, hidden, inter // 16), dtype=np.uint8)
        fc1_g = rng.uniform(0.01, 0.03, size=(experts,)).astype(np.float32)
        fc2_g = rng.uniform(0.01, 0.03, size=(experts,)).astype(np.float32)

        inputs = ["X", "R", "fc1_w", "fc1_s", "", "fc2_w", "fc2_s", "", "", "", "", "", "", "", "", "fc1_g", "fc2_g"]
        node = helper.make_node(
            "QMoEOffload",
            inputs=inputs,
            outputs=["Y"],
            domain="com.microsoft",
            activation_type="swiglu",
            normalize_routing_weights=1,
            k=top_k,
            expert_weight_bits=4,
            block_size=16,
            swiglu_fusion=1,
            quant_type="nvfp4",
            weights_prepacked=1,
            layer_index=0,
            hidden_size=hidden,
            inter_size=inter,
            token_tile_size=4,
            gpu_working_bytes=32 << 20,
            pinned_staging_bytes=8 << 20,
        )
        inits = [
            numpy_helper.from_array(fc1_w, name="fc1_w"),
            helper.make_tensor("fc1_s", TensorProto.FLOAT8E4M3FN, fc1_s.shape, fc1_s.tobytes(), raw=True),
            numpy_helper.from_array(fc2_w, name="fc2_w"),
            helper.make_tensor("fc2_s", TensorProto.FLOAT8E4M3FN, fc2_s.shape, fc2_s.tobytes(), raw=True),
            numpy_helper.from_array(fc1_g, name="fc1_g"),
            numpy_helper.from_array(fc2_g, name="fc2_g"),
        ]
        graph = helper.make_graph(
            [node],
            "p3_fastpath",
            [
                helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [1, "S", hidden]),
                helper.make_tensor_value_info("R", TensorProto.BFLOAT16, ["S", experts]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [1, "S", hidden])],
            inits,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]
        )

        x_bf16 = float32_to_bf16_uint16((rng.standard_normal((1, 2, hidden)) * 0.5).astype(np.float32))
        r_bf16 = float32_to_bf16_uint16(rng.standard_normal((2, experts)).astype(np.float32))
        ort_x = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(x_bf16, TensorProto.BFLOAT16)
        ort_r = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r_bf16, TensorProto.BFLOAT16)

        env_keys = (
            "ORT_QMOE_SHARED_ASYNC",
            "ORT_QMOE_SHARED_PINNED_REUSE",
            "ORT_QMOE_SHARED_RESIDENT_SLOTS",
            "ORT_QMOE_SHARED_PREPOPULATE",
            "ORT_QMOE_DETERMINISTIC_REPLAY",
            "ORT_QMOE_SHARED_RESIDENT_PROMOTIONS",
        )
        saved = {k: os.environ.get(k) for k in env_keys}
        try:
            # 1) Statically full async fast path (all 8 experts resident, no CPU callback)
            os.environ.update(
                ORT_QMOE_SHARED_ASYNC="1",
                ORT_QMOE_SHARED_PINNED_REUSE="1",
                ORT_QMOE_SHARED_RESIDENT_SLOTS=str(experts),
                ORT_QMOE_SHARED_PREPOPULATE="1",
                ORT_QMOE_DETERMINISTIC_REPLAY="1",
            )
            sess_full = make_plugin_session(model)
            out_full_ort = sess_full.run_with_ort_values(["Y"], {"X": ort_x, "R": ort_r})[0]
            out_full = bf16_uint16_to_float32(
                np.frombuffer(ctypes.string_at(out_full_ort.data_ptr(), x_bf16.size * 2), dtype=np.uint16).reshape(
                    x_bf16.shape
                )
            )
            del sess_full
            gc.collect()

            # 2) Host-known empty async fast path (resident_slots=4, prepopulate=0, promotions=0)
            os.environ.update(
                ORT_QMOE_SHARED_ASYNC="1",
                ORT_QMOE_SHARED_PINNED_REUSE="1",
                ORT_QMOE_SHARED_RESIDENT_SLOTS="4",
                ORT_QMOE_SHARED_PREPOPULATE="0",
                ORT_QMOE_DETERMINISTIC_REPLAY="1",
                ORT_QMOE_SHARED_RESIDENT_PROMOTIONS="0",
            )
            sess_empty = make_plugin_session(model)
            out_empty_ort1 = sess_empty.run_with_ort_values(["Y"], {"X": ort_x, "R": ort_r})[0]
            out_empty1 = bf16_uint16_to_float32(
                np.frombuffer(ctypes.string_at(out_empty_ort1.data_ptr(), x_bf16.size * 2), dtype=np.uint16).reshape(
                    x_bf16.shape
                )
            )
            out_empty_ort2 = sess_empty.run_with_ort_values(["Y"], {"X": ort_x, "R": ort_r})[0]
            out_empty2 = bf16_uint16_to_float32(
                np.frombuffer(ctypes.string_at(out_empty_ort2.data_ptr(), x_bf16.size * 2), dtype=np.uint16).reshape(
                    x_bf16.shape
                )
            )
            del sess_empty
            gc.collect()

            np.testing.assert_array_equal(out_empty1, out_empty2)
            rel_l2 = np.linalg.norm(out_full - out_empty1) / np.linalg.norm(out_full)
            self.assertLess(rel_l2, 5e-4)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_phase4_zero_copy_migration_tool_and_inblock_bias_validation(self):
        """Verifies Phase 4: migrate_qmoe_offload_v3.py zero-copy 26->5 migration and in-block fc2_bias check."""
        sys.path.insert(0, str(ORT_ROOT / "onnxruntime/test/python/transformers"))
        sys.path.insert(0, str(ORT_ROOT / "tools/python"))
        import migrate_qmoe_offload_v3 as mig
        import qmoe_shared_layout_ref as ref

        rng = np.random.default_rng(404)
        experts, hidden, inter, top_k = 4, 128, 64, 2
        blocks_list, biases_list = [], []
        for _ in range(experts):
            w1 = rng.integers(0, 16, size=(2 * inter, hidden), dtype=np.uint8)
            s1 = (rng.uniform(0.01, 0.05, size=(2 * inter, hidden // 32))).astype(np.float16)
            b1 = (rng.uniform(-0.02, 0.02, size=(2 * inter,))).astype(np.float16)
            w2 = rng.integers(0, 16, size=(hidden, inter), dtype=np.uint8)
            s2 = (rng.uniform(0.01, 0.05, size=(hidden, inter // 32))).astype(np.float16)
            b2 = (rng.uniform(-0.02, 0.02, size=(hidden,))).astype(np.float16)
            blocks_list.append(ref.assemble_block(hidden, inter, w1, s1, b1, w2, s2, b2))
            biases_list.append(b2)
        blocks = np.stack(blocks_list, axis=0)
        bias = np.stack(biases_list, axis=0)

        legacy_inputs = ["X", "R"] + [""] * 19 + ["expert_blocks", "", "", "fc2_bias", "phase"]
        legacy_node = helper.make_node(
            "QMoEOffload",
            inputs=legacy_inputs,
            outputs=["Y", "rows", "calls", "counts"],
            domain="com.microsoft",
            name="legacy_qmoe",
            k=top_k,
            normalize_routing_weights=1,
            layout_version=1,
            weights_prepacked=1,
            hidden_size=hidden,
            inter_size=inter,
            layer_index=0,
        )
        legacy_graph = helper.make_graph(
            [legacy_node],
            "legacy_graph",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT16, ["T", hidden]),
                helper.make_tensor_value_info("R", TensorProto.FLOAT16, ["T", experts]),
                helper.make_tensor_value_info("phase", TensorProto.INT8, ["T"]),
            ],
            [
                helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["T", hidden]),
                helper.make_tensor_value_info("rows", TensorProto.INT32, [5, experts]),
                helper.make_tensor_value_info("calls", TensorProto.INT32, [5, experts]),
                helper.make_tensor_value_info("counts", TensorProto.INT32, [6]),
            ],
            [
                numpy_helper.from_array(blocks, name="expert_blocks"),
                numpy_helper.from_array(bias, name="fc2_bias"),
            ],
        )
        legacy_model = helper.make_model(
            legacy_graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]
        )

        with tempfile.TemporaryDirectory() as tmp:
            legacy_path = Path(tmp) / "legacy.onnx"
            migrated_path = Path(tmp) / "sub" / "migrated_v3.onnx"
            onnx.save_model(
                legacy_model,
                str(legacy_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.bin",
                size_threshold=0,
            )
            _, manifest = mig.migrate_qmoe_offload_model(
                legacy_path,
                migrated_path,
                strip_diagnostics=True,
                drop_phase_input=True,
            )
            self.assertEqual(manifest["migrated_nodes"], 1)
            self.assertGreater(manifest["verified_bias_slices"], 0)
            self.assertEqual(manifest["nodes"][0]["inputs"], ["X", "R", "expert_blocks", "fc2_bias"])
            self.assertEqual(manifest["nodes"][0]["outputs"], ["Y"])

            sess_v3 = make_plugin_session(migrated_path)
            x = rng.standard_normal((3, hidden)).astype(np.float16)
            r = rng.standard_normal((3, experts)).astype(np.float16)
            y_v3 = sess_v3.run(None, {"X": x, "R": r})[0]
            self.assertEqual(y_v3.shape, (3, hidden))
            del sess_v3
            gc.collect()

        # Verify that corrupting in-block fc2_bias to a non-zero mismatch against separate fc2_bias is rejected!
        bad_bias = bias.copy()
        bad_bias[0, 0] += np.float16(0.5)
        bad_node = helper.make_node(
            "QMoEOffload",
            inputs=["X", "R", "expert_blocks", "fc2_bias"],
            outputs=["Y"],
            domain="com.microsoft",
            k=top_k,
            normalize_routing_weights=1,
            layout_version=1,
            layout_id="ort.qmoe.w4a16.block32.shared.v1",
            weights_prepacked=1,
            hidden_size=hidden,
            inter_size=inter,
            layer_index=0,
        )
        bad_graph = helper.make_graph(
            [bad_node],
            "bad_bias_graph",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT16, ["T", hidden]),
                helper.make_tensor_value_info("R", TensorProto.FLOAT16, ["T", experts]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["T", hidden])],
            [
                numpy_helper.from_array(blocks, name="expert_blocks"),
                numpy_helper.from_array(bad_bias, name="fc2_bias"),
            ],
        )
        bad_model = helper.make_model(
            bad_graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]
        )
        with self.assertRaises(Exception) as ctx:
            make_plugin_session(bad_model)
        self.assertIn("in-block fc2_bias disagrees", str(ctx.exception))

    def test_phase5_cpu_budget_group_and_nonuniform_layer_capacities(self):
        """Verifies Phase 5: CpuBudgetGroup phase spin suppression and nonuniform per-layer slot allocation."""
        cpp_src = r"""
#include <cassert>
#include "contrib_ops/cuda/moe/qmoe_spin_pool.h"
using namespace onnxruntime::contrib::cuda::qmoe;
int main() {
    SpinPool pool(4, 2000);
    auto& group = CpuBudgetGroup::Default();
    group.ConfigureDecodeSpinUs(2000);
    assert(pool.SpinDurationUs() == 2000);
    group.EnterPrefillOrAuxPhase(&pool);
    assert(pool.SpinDurationUs() == 0);
    assert(group.CurrentPhase() == CpuBudgetGroup::Phase::kPrefillOrAux);
    group.EnterDecodePhase(&pool);
    assert(pool.SpinDurationUs() == 2000);
    assert(group.CurrentPhase() == CpuBudgetGroup::Phase::kDecode);
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "test_p5"
            subprocess.run(
                [
                    "g++",
                    "-O2",
                    "-std=c++20",
                    "-pthread",
                    f"-I{ORT_ROOT / 'onnxruntime'}",
                    "-x",
                    "c++",
                    "-",
                    "-o",
                    str(bin_path),
                ],
                input=cpp_src.encode(),
                check=True,
            )
            subprocess.run([str(bin_path)], check=True)

        sys.path.insert(0, str(ORT_ROOT / "tools/python"))
        import qmoe_expert_distribution as dist

        # Layer 0 has highly concentrated routing (top 2 experts take 90%), Layer 1 has flatter routing over 6 experts.
        trace = {
            0: [450.0, 450.0] + [100.0 / 14.0] * 14,
            1: [150.0] * 6 + [10.0] * 10,
        }
        caps, init_exps, u_hit, nu_hit = dist.compute_nonuniform_layer_capacities(
            trace, total_slots=8, num_experts=16, min_slots_per_layer=2, max_slots_per_layer=16
        )
        self.assertEqual(sum(caps.values()), 8)
        self.assertEqual(caps[0], 2)
        self.assertEqual(caps[1], 6)
        self.assertGreater(nu_hit, u_hit)
        self.assertEqual(len(init_exps[0]), 2)
        self.assertEqual(len(init_exps[1]), 6)

    def _make_nvfp4_test_model(
        self,
        rng,
        experts=8,
        hidden=128,
        inter=64,
        top_k=2,
        resident_slots=0,
        spare_slots=0,
        initial_experts=None,
        extra_attrs=None,
        banks=None,
    ):
        if banks is None:
            fc1_w = rng.integers(0, 256, size=(experts, hidden, inter), dtype=np.uint8)
            fc1_s = rng.integers(0x34, 0x44, size=(experts, 2 * inter, hidden // 16), dtype=np.uint8)
            fc2_w = rng.integers(0, 256, size=(experts, inter, hidden // 2), dtype=np.uint8)
            fc2_s = rng.integers(0x34, 0x44, size=(experts, hidden, inter // 16), dtype=np.uint8)
            fc1_g = rng.uniform(0.01, 0.03, size=(experts,)).astype(np.float32)
            fc2_g = rng.uniform(0.01, 0.03, size=(experts,)).astype(np.float32)
            banks = (fc1_w, fc1_s, fc1_g, fc2_w, fc2_s, fc2_g)
        else:
            fc1_w, fc1_s, fc1_g, fc2_w, fc2_s, fc2_g = banks
        inputs = ["X", "R", "fc1_w", "fc1_s", "", "fc2_w", "fc2_s", "", "", "", "", "", "", "", "", "fc1_g", "fc2_g"]
        kwargs = dict(
            activation_type="swiglu",
            normalize_routing_weights=1,
            k=top_k,
            expert_weight_bits=4,
            block_size=16,
            swiglu_fusion=1,
            quant_type="nvfp4",
            weights_prepacked=1,
            layer_index=0,
            hidden_size=hidden,
            inter_size=inter,
            token_tile_size=8,
            gpu_working_bytes=64 << 20,
            pinned_staging_bytes=16 << 20,
            resident_slots=resident_slots,
            spare_slots=spare_slots,
        )
        if initial_experts is not None:
            kwargs["initial_experts"] = initial_experts
        if extra_attrs:
            kwargs.update(extra_attrs)
        node = helper.make_node("QMoEOffload", inputs=inputs, outputs=["Y"], domain="com.microsoft", **kwargs)
        inits = [
            numpy_helper.from_array(fc1_w, name="fc1_w"),
            helper.make_tensor("fc1_s", TensorProto.FLOAT8E4M3FN, fc1_s.shape, fc1_s.tobytes(), raw=True),
            numpy_helper.from_array(fc2_w, name="fc2_w"),
            helper.make_tensor("fc2_s", TensorProto.FLOAT8E4M3FN, fc2_s.shape, fc2_s.tobytes(), raw=True),
            numpy_helper.from_array(fc1_g, name="fc1_g"),
            numpy_helper.from_array(fc2_g, name="fc2_g"),
        ]
        graph = helper.make_graph(
            [node],
            "qmoe_nvfp4_test",
            [
                helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [1, "S", hidden]),
                helper.make_tensor_value_info("R", TensorProto.BFLOAT16, ["S", experts]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [1, "S", hidden])],
            inits,
        )
        return (
            helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]
            ),
            banks,
        )

    @staticmethod
    def _run_bf16_session(sess, x_bf16, r_bf16):
        ort_x = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(x_bf16, TensorProto.BFLOAT16)
        ort_r = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r_bf16, TensorProto.BFLOAT16)
        out_ort = sess.run_with_ort_values(["Y"], {"X": ort_x, "R": ort_r})[0]
        return bf16_uint16_to_float32(
            np.frombuffer(ctypes.string_at(out_ort.data_ptr(), x_bf16.size * 2), dtype=np.uint16).reshape(x_bf16.shape)
        )

    def test_numa_striping_and_worker_pinning(self):
        """Verifies NUMA page striping (ApplyNumaInterleavePolicy) and SpinPool socket-balanced worker pinning."""
        cpp_src = r"""
#include <cassert>
#include <thread>
#include <vector>
#include "contrib_ops/cuda/moe/qmoe_spin_pool.h"
using namespace onnxruntime::contrib::cuda::qmoe;
int main() {
    std::vector<uint8_t> buf(1 << 20, 0);
    // ApplyNumaInterleavePolicy safely succeeds on multi-node Linux hosts and never crashes.
    (void)ApplyNumaInterleavePolicy(buf.data(), buf.size());
    SpinPool pool(8, 1000);
    if (pool.NumaNodeCount() > 0) {
        while (pool.PinnedWorkerCount() < 7) {
            std::this_thread::yield();
        }
        assert(pool.PinnedWorkerCount() == 7);
    }
    std::atomic<int> sum{0};
    auto fn = [&](ptrdiff_t idx) { sum.fetch_add(static_cast<int>(idx + 1)); };
    pool.ParallelFor(16, fn, 1);
    assert(sum.load() == 16 * 17 / 2);
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "test_numa"
            subprocess.run(
                [
                    "g++",
                    "-O2",
                    "-std=c++20",
                    "-pthread",
                    f"-I{ORT_ROOT / 'onnxruntime'}",
                    "-x",
                    "c++",
                    "-",
                    "-o",
                    str(bin_path),
                ],
                input=cpp_src.encode(),
                check=True,
            )
            subprocess.run([str(bin_path)], check=True)

    def test_multi_row_rt1_to_rt4_cpu_reuse(self):
        """Verifies RT=1..4 contiguous token zero-copy activation reuse and Fp4Group8Avx512<RT> accuracy."""
        rng = np.random.default_rng(701)
        experts, hidden, inter, top_k = 8, 128, 64, 2
        m_cpu, banks = self._make_nvfp4_test_model(rng, experts, hidden, inter, top_k, resident_slots=0)
        m_gpu, _ = self._make_nvfp4_test_model(
            rng, experts, hidden, inter, top_k, resident_slots=experts, initial_experts=list(range(experts)), banks=banks
        )
        old_env = os.environ.get("ORT_QMOE_SHARED_ASYNC")
        try:
            os.environ["ORT_QMOE_SHARED_ASYNC"] = "1"
            sess_cpu = make_plugin_session(m_cpu)
            sess_gpu = make_plugin_session(m_gpu)
            for rt in (1, 2, 3, 4):
                x_f32 = rng.standard_normal((1, rt, hidden)).astype(np.float32) * 0.25
                # Route all rt tokens to the same top-2 experts (0 and 1) so cnt == rt triggers Fp4RowTileAvx512<rt>!
                r_f32 = np.full((rt, experts), -5.0, dtype=np.float32)
                r_f32[:, 0] = 4.0
                r_f32[:, 1] = 3.0
                x_bf16 = float32_to_bf16_uint16(x_f32)
                r_bf16 = float32_to_bf16_uint16(r_f32)
                y_cpu = self._run_bf16_session(sess_cpu, x_bf16, r_bf16)
                y_gpu = self._run_bf16_session(sess_gpu, x_bf16, r_bf16)
                np.testing.assert_allclose(y_cpu, y_gpu, rtol=2e-2, atol=2e-2)
            del sess_cpu, sess_gpu
            gc.collect()
        finally:
            if old_env is None:
                os.environ.pop("ORT_QMOE_SHARED_ASYNC", None)
            else:
                os.environ["ORT_QMOE_SHARED_ASYNC"] = old_env

    def test_doorbell_and_cuda_graph_capture(self):
        """Verifies Device-Doorbell Host Worker Protocol and CUDA Graph Capture/Replay on QMoEOffload."""
        rng = np.random.default_rng(702)
        experts, hidden, inter, top_k = 8, 128, 64, 2
        m_ref, banks = self._make_nvfp4_test_model(
            rng, experts, hidden, inter, top_k, resident_slots=4, spare_slots=2, initial_experts=[0, 1, 2, 3]
        )
        m_doorbell, _ = self._make_nvfp4_test_model(
            rng,
            experts,
            hidden,
            inter,
            top_k,
            resident_slots=4,
            spare_slots=2,
            initial_experts=[0, 1, 2, 3],
            extra_attrs={"allow_cuda_graph": 1},
            banks=banks,
        )

        x_f32 = rng.standard_normal((1, 2, hidden)).astype(np.float32) * 0.25
        r_f32 = rng.standard_normal((2, experts)).astype(np.float32)
        x_bf16 = float32_to_bf16_uint16(x_f32)
        r_bf16 = float32_to_bf16_uint16(r_f32)

        old_doorbell = os.environ.get("ORT_QMOE_DOORBELL")
        old_allow_cg = os.environ.get("ORT_QMOE_ALLOW_CUDA_GRAPH")
        try:
            os.environ["ORT_QMOE_DOORBELL"] = "0"
            sess_ref = make_plugin_session(m_ref)
            y_ref = self._run_bf16_session(sess_ref, x_bf16, r_bf16)
            del sess_ref
            gc.collect()

            os.environ["ORT_QMOE_DOORBELL"] = "1"
            os.environ["ORT_QMOE_ALLOW_CUDA_GRAPH"] = "1"
            ep_name = ensure_plugin_registered()
            devices = [
                d for d in ort.get_ep_devices() if d.ep_name == ep_name and d.device.type == ort.OrtHardwareDeviceType.GPU
            ]
            devices.sort(key=lambda d: int(d.ep_options.get("device_id", d.ep_metadata.get("cuda_device_id", "0"))))
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            opts.add_provider_for_devices(
                [devices[0]], {"enable_cuda_graph": "1", "min_num_runs_before_cuda_graph_capture": "2"}
            )
            # Build a model with Cast(BFLOAT16 -> FLOAT) output so OrtValue(..., "cuda", 0) and .numpy() work directly on GPU!
            cast_out = helper.make_node("Cast", inputs=["Y"], outputs=["Y_f32"], to=TensorProto.FLOAT)
            m_doorbell.graph.node.append(cast_out)
            del m_doorbell.graph.output[:]
            m_doorbell.graph.output.append(helper.make_tensor_value_info("Y_f32", TensorProto.FLOAT, [1, "S", hidden]))
            cast_in_x = helper.make_node("Cast", inputs=["X_f32"], outputs=["X"], to=TensorProto.BFLOAT16)
            cast_in_r = helper.make_node("Cast", inputs=["R_f32"], outputs=["R"], to=TensorProto.BFLOAT16)
            m_doorbell.graph.node.insert(0, cast_in_r)
            m_doorbell.graph.node.insert(0, cast_in_x)
            del m_doorbell.graph.input[:]
            m_doorbell.graph.input.extend(
                [
                    helper.make_tensor_value_info("X_f32", TensorProto.FLOAT, [1, "S", hidden]),
                    helper.make_tensor_value_info("R_f32", TensorProto.FLOAT, ["S", experts]),
                ]
            )

            sess_cg = ort.InferenceSession(m_doorbell.SerializeToString(), sess_options=opts)
            io_binding = sess_cg.io_binding()
            io_binding.bind_cpu_input("X_f32", x_f32)
            io_binding.bind_cpu_input("R_f32", r_f32)
            io_binding.bind_output("Y_f32", "cuda", 0)
            # Run 6 times: runs 1-2 eager warmup with Device-Doorbell, run 3 capture + replay, runs 4-6 CUDA Graph replay!
            for _ in range(6):
                io_binding.synchronize_inputs()
                sess_cg.run_with_iobinding(io_binding)
                y_cg = io_binding.copy_outputs_to_cpu()[0]
                np.testing.assert_allclose(y_cg, y_ref, rtol=1e-3, atol=1e-3)
            del sess_cg
            gc.collect()
        finally:
            if old_doorbell is None:
                os.environ.pop("ORT_QMOE_DOORBELL", None)
            else:
                os.environ["ORT_QMOE_DOORBELL"] = old_doorbell
            if old_allow_cg is None:
                os.environ.pop("ORT_QMOE_ALLOW_CUDA_GRAPH", None)
            else:
                os.environ["ORT_QMOE_ALLOW_CUDA_GRAPH"] = old_allow_cg

    def test_disjoint_host_complement_mode_and_ssd_tier(self):
        """Verifies Disjoint Host Complement Mode (D2H eviction writeback) and SSD FileExpertStore tier."""
        rng = np.random.default_rng(703)
        experts, hidden, inter, top_k = 8, 128, 64, 2
        m_ref, banks = self._make_nvfp4_test_model(rng, experts, hidden, inter, top_k, resident_slots=0)
        m_comp, _ = self._make_nvfp4_test_model(
            rng,
            experts,
            hidden,
            inter,
            top_k,
            resident_slots=3,
            spare_slots=1,
            initial_experts=[0, 1, 2],
            extra_attrs={"host_complement_mode": 1},
            banks=banks,
        )
        m_ssd, _ = self._make_nvfp4_test_model(
            rng,
            experts,
            hidden,
            inter,
            top_k,
            resident_slots=2,
            spare_slots=1,
            initial_experts=[0, 1],
            extra_attrs={"ssd_cache_slots": 3},
            banks=banks,
        )

        sess_ref = make_plugin_session(m_ref)
        sess_comp = make_plugin_session(m_comp)
        sess_ssd = make_plugin_session(m_ssd)
        try:
            # Cycle routes across cold experts (3..7) and previously resident experts (0..2) to trigger
            # D2H eviction writebacks and subsequent CPU execution of evicted experts from complement_host_slab_!
            for step in range(8):
                x_f32 = rng.standard_normal((1, 2, hidden)).astype(np.float32) * 0.25
                r_f32 = np.full((2, experts), -5.0, dtype=np.float32)
                e0 = (3 + step) % experts
                e1 = (4 + step) % experts
                r_f32[:, e0] = 5.0
                r_f32[:, e1] = 4.0
                x_bf16 = float32_to_bf16_uint16(x_f32)
                r_bf16 = float32_to_bf16_uint16(r_f32)
                y_ref = self._run_bf16_session(sess_ref, x_bf16, r_bf16)
                y_comp = self._run_bf16_session(sess_comp, x_bf16, r_bf16)
                y_ssd = self._run_bf16_session(sess_ssd, x_bf16, r_bf16)
                np.testing.assert_allclose(y_comp, y_ref, rtol=2e-2, atol=2e-2)
                np.testing.assert_allclose(y_ssd, y_ref, rtol=2e-2, atol=2e-2)
        finally:
            del sess_ref, sess_comp, sess_ssd
            gc.collect()


if __name__ == "__main__":
    unittest.main()

