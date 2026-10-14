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


if __name__ == "__main__":
    unittest.main()
