#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Unit tests for Portable QMoE Offload v4 (standard com.microsoft::QMoE + session.moe.offload.*)."""

import contextlib
import ctypes
import faulthandler
import gc
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx

_WIN_DLL_HANDLES = []
if os.name == "nt":
    for _root in (os.environ.get("CUDNN_HOME", ""), os.environ.get("CUDA_PATH", "")):
        if _root:
            for _sub in (os.path.join("bin", "x64"), "bin"):
                _d = os.path.join(_root, _sub)
                if os.path.isdir(_d):
                    _WIN_DLL_HANDLES.append(os.add_dll_directory(_d))
                    if _d not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort  # noqa: E402

TensorProto = onnx.TensorProto
helper = onnx.helper

_E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _decode_e4m3fn_table() -> np.ndarray:
    tbl = np.zeros(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if (code & 0x80) else 1.0
        exp = (code >> 3) & 0x0F
        mant = code & 0x07
        if exp == 0:
            val = (mant / 8.0) * (2.0**-6)
        elif exp == 15 and mant == 7:
            val = np.nan
        else:
            val = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        tbl[code] = sign * val
    return tbl


_E4M3FN_TABLE = _decode_e4m3fn_table()


def _float32_to_bfloat16_u16(arr: np.ndarray) -> np.ndarray:
    u32 = arr.astype(np.float32).view(np.uint32)
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)


def _bfloat16_u16_to_float32(arr: np.ndarray) -> np.ndarray:
    u32 = arr.astype(np.uint32) << 16
    return u32.view(np.float32)


def _round_to_act_dtype(arr: np.ndarray, use_fp16: bool = False) -> np.ndarray:
    if use_fp16:
        return arr.astype(np.float16).astype(np.float32)
    return _bfloat16_u16_to_float32(_float32_to_bfloat16_u16(arr))


def _encode_act_dtype_u16(arr: np.ndarray, use_fp16: bool = False) -> np.ndarray:
    """Returns the raw 16-bit encoding of `arr` in the activation dtype (FP16 or BF16)."""
    if use_fp16:
        return arr.astype(np.float16).view(np.uint16)
    return _float32_to_bfloat16_u16(arr)


def _cuda_device_compute_capability(ordinal: int = 0) -> tuple[int, int] | None:
    """Returns (major, minor) from the CUDA driver API, or None when the device cannot be queried."""
    lib_names = ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
    cuda = None
    for lib_name in lib_names:
        with contextlib.suppress(OSError):
            cuda = ctypes.CDLL(lib_name)
            break
    if cuda is None:
        return None
    try:
        device = ctypes.c_int()
        major = ctypes.c_int()
        minor = ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR
        if (
            cuda.cuInit(0) != 0
            or cuda.cuDeviceGet(ctypes.byref(device), ctypes.c_int(ordinal)) != 0
            or cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device) != 0
            or cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device) != 0
        ):
            return None
    except (AttributeError, OSError):
        return None
    return int(major.value), int(minor.value)


def _load_cudart():
    for lib_name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so", "cudart64_13.dll", "cudart64_12.dll"):
        with contextlib.suppress(OSError):
            return ctypes.CDLL(lib_name)
    return None


def _cuda_malloc(cudart, nbytes: int, device_id: int = 0) -> ctypes.c_void_p:
    with contextlib.suppress(AttributeError):
        cudart.cudaSetDevice(ctypes.c_int(device_id))
    ptr = ctypes.c_void_p()
    status = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if status != 0:
        raise RuntimeError(f"cudaMalloc({nbytes}) failed with status {status}")
    return ptr


def _cuda_upload(cudart, device_ptr: ctypes.c_void_p, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    status = cudart.cudaMemcpy(device_ptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, 1)  # cudaMemcpyHostToDevice
    if status != 0:
        raise RuntimeError(f"cudaMemcpy failed with status {status}")


def _wrap_qmoe_model_with_cast_io(model_path: str, out_path: str, hidden_size: int, num_experts: int) -> str:
    """Wraps a single-router canonical model in FLOAT <-> BFLOAT16 casts so CUDA Graph tests can bind GPU buffers."""
    m = onnx.load_model(model_path, load_external_data=False)
    m.graph.node.insert(
        0, helper.make_node("Cast", inputs=["R_f32"], outputs=["router_probs"], to=TensorProto.BFLOAT16)
    )
    m.graph.node.insert(0, helper.make_node("Cast", inputs=["X_f32"], outputs=["X"], to=TensorProto.BFLOAT16))
    m.graph.node.append(helper.make_node("Cast", inputs=["Y"], outputs=["Y_f32"], to=TensorProto.FLOAT))
    del m.graph.input[:]
    m.graph.input.extend(
        [
            helper.make_tensor_value_info("X_f32", TensorProto.FLOAT, ["T", hidden_size]),
            helper.make_tensor_value_info("R_f32", TensorProto.FLOAT, ["T", num_experts]),
        ]
    )
    del m.graph.output[:]
    m.graph.output.append(helper.make_tensor_value_info("Y_f32", TensorProto.FLOAT, ["T", hidden_size]))
    onnx.save_model(m, out_path)
    return out_path


@contextlib.contextmanager
def _capture_native_stderr():
    """Captures file-descriptor 2, which is where the C++ backend prints its diagnostics."""
    sys.stderr.flush()
    saved_fd = os.dup(2)
    captured = []
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield captured
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            tmp.seek(0)
            captured.append(tmp.read().decode(errors="replace"))


_RESIDENT_LOG_RE = re.compile(r"resident layer=(\d+) slots=(\d+) hit_routes=(\d+) routes=(\d+)")


def _resident_log_rows(log_text: str) -> dict[int, tuple[int, int, int]]:
    """Parses the per-layer (slots, hit_routes, routes) totals that the backend prints when it is destroyed.

    The backend prints them only when ORT_QMOE_SHARED_RESIDENT_LOG=1 is set before the session is created.
    """
    rows = {
        int(m.group(1)): (int(m.group(2)), int(m.group(3)), int(m.group(4)))
        for m in _RESIDENT_LOG_RE.finditer(log_text)
    }
    if not rows:
        raise AssertionError(
            "No 'QMoEOffload resident layer=...' lines were captured from native stderr; the backend prints them "
            f"when it is destroyed. Captured output:\n{log_text[-2000:]}"
        )
    return rows


def _resident_slots_per_layer(log_text: str) -> dict[int, int]:
    return {layer: slots for layer, (slots, _, _) in _resident_log_rows(log_text).items()}


def _resident_route_stats(log_text: str) -> dict[int, tuple[int, int]]:
    return {layer: (hit_routes, routes) for layer, (_, hit_routes, routes) in _resident_log_rows(log_text).items()}


def _resident_routes_per_layer(log_text: str) -> dict[int, int]:
    return {layer: routes for layer, (_, _, routes) in _resident_log_rows(log_text).items()}


_RESIDENT_PROMOTIONS_RE = re.compile(r"resident layer=(\d+) slots=\d+ hit_routes=\d+ routes=\d+ promotions=(\d+)")
_CPU_PATH_LOG_RE = re.compile(r"cpu layer=(\d+) calls=(\d+) routes=(\d+) cold_routes=(\d+)")


def _resident_promotions_per_layer(log_text: str) -> dict[int, int]:
    """Parses how many experts each layer promoted into its resident set (same log as _resident_log_rows)."""
    return {int(m.group(1)): int(m.group(2)) for m in _RESIDENT_PROMOTIONS_RE.finditer(log_text)}


def _cpu_path_route_rows(log_text: str) -> dict[int, tuple[int, int, int]]:
    """Parses the per-layer (calls, routes, cold_routes) totals of the CPU expert path.

    cold_routes counts the routes whose expert was not resident when the CPU path classified them, so it observes the
    placement at the moment each route executed. The backend prints the totals when it is destroyed, but only if
    ORT_QMOE_SHARED_CPU_TIMING=1 was set before the session was created, and it counts only runs of at most 4 rows.
    """
    rows = {
        int(m.group(1)): (int(m.group(2)), int(m.group(3)), int(m.group(4)))
        for m in _CPU_PATH_LOG_RE.finditer(log_text)
    }
    if not rows:
        raise AssertionError(
            "No 'QMoEOffload cpu layer=...' lines were captured from native stderr; the backend prints them when it "
            f"is destroyed. Captured output:\n{log_text[-2000:]}"
        )
    return rows


_STREAM_TILES_RE = re.compile(r"stream layer=(\d+) tiles=(\d+)")
_ASYNC_SERVICE_RE = re.compile(r"async layer=(\d+) doorbell_runs=(\d+) callback_runs=(\d+)")


def _async_service_runs_per_layer(log_text: str) -> dict[int, tuple[int, int]]:
    """Parses how many uncaptured runs each layer served through the doorbell and the callback (same log as above)."""
    return {int(m.group(1)): (int(m.group(2)), int(m.group(3))) for m in _ASYNC_SERVICE_RE.finditer(log_text)}


def _stream_tiles_per_layer(log_text: str) -> dict[int, int]:
    """Parses how many tiles each layer streamed through the synchronous GPU path (same log as _resident_log_rows)."""
    return {int(m.group(1)): int(m.group(2)) for m in _STREAM_TILES_RE.finditer(log_text)}


def _run_qmoe_bf16(sess, x_f32: np.ndarray, r_f32: np.ndarray, run_options=None) -> np.ndarray:
    x_bf16 = _float32_to_bfloat16_u16(x_f32)
    r_bf16 = _float32_to_bfloat16_u16(r_f32)
    ort_x = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(x_bf16, TensorProto.BFLOAT16)
    ort_r = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r_bf16, TensorProto.BFLOAT16)
    out = sess.run_with_ort_values(["Y"], {"X": ort_x, "router_probs": ort_r}, run_options)[0]
    raw_bytes = bytes((ctypes.c_uint16 * int(np.prod(x_f32.shape))).from_address(out.data_ptr()))
    return _bfloat16_u16_to_float32(np.frombuffer(raw_bytes, dtype=np.uint16).reshape(x_f32.shape))


def _qmoe_background_thread_names() -> list[str]:
    names = []
    for task_dir in Path("/proc/self/task").iterdir():
        with contextlib.suppress(OSError):
            names.append((task_dir / "comm").read_text().strip())
    return sorted(name for name in names if name.startswith("qmoe-"))


def _dequant_nvfp4_row_major(w_u8: np.ndarray, s_u8: np.ndarray, g_f32: np.ndarray, n: int, k: int) -> np.ndarray:
    ne = w_u8.shape[0]
    w_flat = w_u8.reshape(ne, n, k // 2)
    lo = _E2M1_TABLE[w_flat & 0x0F]
    hi = _E2M1_TABLE[(w_flat >> 4) & 0x0F]
    w_f32 = np.empty((ne, n, k), dtype=np.float32)
    w_f32[:, :, 0::2] = lo
    w_f32[:, :, 1::2] = hi
    s_f32 = _E4M3FN_TABLE[s_u8.reshape(ne, n, k // 16)] * g_f32.reshape(ne, 1, 1)
    return w_f32 * np.repeat(s_f32, 16, axis=2)


def _numpy_nvfp4_qmoe_reference(
    model_path: str,
    x_f32: np.ndarray,
    r_f32: np.ndarray | list[np.ndarray],
    use_fp16: bool = False,
    dropped_rank: int | None = None,
    dropped_route: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Reference output for the canonical NVFP4 QMoE chain.

    dropped_rank removes that top-k rank for every row of every layer. dropped_route=(layer, row, rank) removes a
    single expert contribution, which lets tests prove that their tolerances can detect one lost route.
    """
    model = onnx.load_model(model_path, load_external_data=True)
    init_map = {}
    for t in model.graph.initializer:
        raw = t.raw_data
        dims = tuple(int(d) for d in t.dims)
        if t.data_type == TensorProto.UINT8 or t.data_type == TensorProto.FLOAT8E4M3FN:
            init_map[t.name] = np.frombuffer(raw, dtype=np.uint8).reshape(dims).copy()
        elif t.data_type == TensorProto.FLOAT:
            init_map[t.name] = np.frombuffer(raw, dtype=np.float32).reshape(dims).copy()
        elif t.data_type == TensorProto.BFLOAT16:
            init_map[t.name] = _bfloat16_u16_to_float32(np.frombuffer(raw, dtype=np.uint16).reshape(dims)).copy()
        elif t.data_type == TensorProto.FLOAT16:
            init_map[t.name] = np.frombuffer(raw, dtype=np.float16).reshape(dims).astype(np.float32).copy()
        else:
            init_map[t.name] = onnx.numpy_helper.to_array(t).astype(np.float32)

    cur = _round_to_act_dtype(x_f32, use_fp16=use_fp16)
    qmoe_idx = 0
    for node in model.graph.node:
        if node.op_type == "Add":
            cur = _round_to_act_dtype(cur + init_map[node.input[1]], use_fp16=use_fp16)
            continue
        if node.op_type != "QMoE":
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        top_k = int(attrs.get("k", 4))
        norm_weights = int(attrs.get("normalize_routing_weights", 1)) != 0

        r_in = r_f32[qmoe_idx] if isinstance(r_f32, (list, tuple)) else r_f32
        layer_idx = qmoe_idx
        qmoe_idx += 1
        logits = _round_to_act_dtype(r_in, use_fp16=use_fp16)
        logits_shift = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits_shift) / np.sum(np.exp(logits_shift), axis=-1, keepdims=True)

        fc1_w_u8 = init_map[node.input[2]]
        fc1_s_u8 = init_map[node.input[3]]
        fc1_b = init_map.get(node.input[4]) if len(node.input) > 4 and node.input[4] else None
        fc2_w_u8 = init_map[node.input[5]]
        fc2_s_u8 = init_map[node.input[6]]
        fc2_b = init_map.get(node.input[7]) if len(node.input) > 7 and node.input[7] else None
        fc1_g = init_map[node.input[15]]
        fc2_g = init_map[node.input[16]]

        ne = fc1_w_u8.shape[0]
        hid = cur.shape[1]
        # FC1 scales are [E, 2*I, H/blk] in every weight layout, unlike the weight dims that differ between layouts.
        inter = fc1_s_u8.shape[1] // 2
        fc1_w = _round_to_act_dtype(_dequant_nvfp4_row_major(fc1_w_u8, fc1_s_u8, fc1_g, 2 * inter, hid), use_fp16)
        fc2_w = _round_to_act_dtype(_dequant_nvfp4_row_major(fc2_w_u8, fc2_s_u8, fc2_g, hid, inter), use_fp16)

        num_rows = cur.shape[0]
        out = np.zeros((num_rows, hid), dtype=np.float32)
        for row in range(num_rows):
            # Stable argsort matching TopK tie-breaking (highest probability, lowest index first)
            order = np.lexsort((np.arange(ne), -probs[row]))[:top_k]
            weights = probs[row, order].copy()
            if norm_weights:
                weights = weights / np.sum(weights)
            row_acc = np.zeros((hid,), dtype=np.float32)
            for rank_idx, (e_id, w_scale) in enumerate(zip(order, weights, strict=False)):
                if dropped_rank is not None and rank_idx == dropped_rank:
                    continue
                if dropped_route is not None and dropped_route == (layer_idx, row, rank_idx):
                    continue
                proj1 = fc1_w[e_id] @ cur[row]
                if fc1_b is not None:
                    proj1 = proj1 + fc1_b[e_id]
                gate = proj1[0::2]
                up = proj1[1::2]
                swiglu = (gate / (1.0 + np.exp(-gate))) * up
                swiglu = _round_to_act_dtype(swiglu, use_fp16=use_fp16)
                proj2 = fc2_w[e_id] @ swiglu
                if fc2_b is not None:
                    proj2 = proj2 + fc2_b[e_id]
                row_acc = row_acc + float(w_scale) * proj2
            out[row] = _round_to_act_dtype(row_acc, use_fp16=use_fp16)
        cur = out
    return cur


def _build_canonical_nvfp4_qmoe_model(
    model_dir: str,
    num_layers: int = 2,
    num_experts: int = 16,
    hidden_size: int = 128,
    inter_size: int = 64,
    top_k: int = 4,
    seed: int = 42,
    with_bias: bool = False,
    short_profile_pairs: int | None = None,
    embedded_initializers: bool = False,
    non_moe_gpu_init_elements: int = 0,
    layer_experts: list[int] | None = None,
    quant_type: str = "nvfp4",
    use_fp16: bool = False,
    weight_layout: str = "legacy",
) -> str:
    """Builds a chain of QMoE layers with random NVFP4 weights.

    weight_layout="legacy" emits FC1 [E, H, I] and FC2 [E, I, H/2]; "standard" emits FC1 [E, 2*I, H/2] and
    FC2 [E, H, I/2]. Both layouts carry identical bytes for a given seed.
    """
    if weight_layout not in ("legacy", "standard"):
        raise ValueError(f"unknown weight_layout {weight_layout!r}")
    rng = np.random.default_rng(seed)
    ext_filename = "weights.data"
    ext_path = os.path.join(model_dir, ext_filename)
    is_mxfp4 = quant_type in ("fp4", "mxfp4")
    blk_size = 32 if is_mxfp4 else 16
    scale_dtype = TensorProto.FLOAT8E8M0 if is_mxfp4 else TensorProto.FLOAT8E4M3FN
    act_onnx_dtype = TensorProto.FLOAT16 if use_fp16 else TensorProto.BFLOAT16

    nodes = []
    initializers = []
    offset = 0

    def add_tensor(wf, name, arr, onnx_dtype, force_embedded=False):
        nonlocal offset
        raw = arr.tobytes()
        if embedded_initializers or force_embedded:
            t = helper.make_tensor(
                name=name,
                data_type=onnx_dtype,
                dims=list(arr.shape),
                vals=raw,
                raw=True,
            )
            initializers.append(t)
            return
        pad = (4096 - (offset % 4096)) % 4096
        if pad:
            wf.write(b"\x00" * pad)
            offset += pad
        wf.write(raw)
        t = helper.make_tensor(
            name=name,
            data_type=onnx_dtype,
            dims=list(arr.shape),
            vals=raw,
            raw=True,
        )
        onnx.external_data_helper.set_external_data(t, location=ext_filename, offset=offset, length=len(raw))
        t.data_location = TensorProto.EXTERNAL
        t.ClearField("raw_data")
        offset += len(raw)
        initializers.append(t)

    with open(ext_path, "wb") as wf:
        cur_in = "X"
        if non_moe_gpu_init_elements > 0:
            non_moe_arr = np.zeros((1, hidden_size), dtype=np.uint16)
            add_tensor(wf, "non_moe_gpu_bias_init", non_moe_arr, act_onnx_dtype, force_embedded=True)
            nodes.append(
                helper.make_node(
                    "Add",
                    inputs=[cur_in, "non_moe_gpu_bias_init"],
                    outputs=["X_after_non_moe"],
                    name="model/non_moe_pre_add",
                )
            )
            cur_in = "X_after_non_moe"

        for l_idx in range(num_layers):
            ne = layer_experts[l_idx] if layer_experts is not None else num_experts
            fc1_w = rng.integers(0, 256, size=(ne, hidden_size, inter_size), dtype=np.uint8)
            fc2_w = rng.integers(0, 256, size=(ne, inter_size, hidden_size // 2), dtype=np.uint8)
            if weight_layout == "standard":
                fc1_w = fc1_w.reshape(ne, 2 * inter_size, hidden_size // 2)
                fc2_w = fc2_w.reshape(ne, hidden_size, inter_size // 2)
            if is_mxfp4:
                # E8M0 scale codes: 125 = 2^-2 = 0.25, 126 = 2^-1 = 0.5
                scale_choices = np.array([124, 125, 126], dtype=np.uint8)
                fc1_g = np.full((ne,), 0.05, dtype=np.float32)
                fc2_g = np.full((ne,), 0.05, dtype=np.float32)
            else:
                # E4M3 scale bytes: 0x4C = 3.0, 0x50 = 4.0, 0x54 = 6.0; paired with exact power-of-2 global_scale = 2^-7 = 0.0078125
                # (0.0078125 * 448 = 3.5 < 3.98 fast-decode bound) so block*global is exact in BF16/FP16/FP32 and output signal ~ O(1..10).
                scale_choices = np.array([0x4C, 0x50, 0x54], dtype=np.uint8)
                fc1_g = np.full((ne,), 0.0078125, dtype=np.float32)
                fc2_g = np.full((ne,), 0.0078125, dtype=np.float32)
            fc1_s = rng.choice(scale_choices, size=(ne, 2 * inter_size, hidden_size // blk_size))
            fc2_s = rng.choice(scale_choices, size=(ne, hidden_size, inter_size // blk_size))

            w1_name = f"model.layers.{l_idx}.mlp.fc1_experts_weights"
            s1_name = f"model.layers.{l_idx}.mlp.fc1_scales"
            b1_name = ""
            w2_name = f"model.layers.{l_idx}.mlp.fc2_experts_weights"
            s2_name = f"model.layers.{l_idx}.mlp.fc2_scales"
            b2_name = ""
            g1_name = f"model.layers.{l_idx}.mlp.fc1_global_scales"
            g2_name = f"model.layers.{l_idx}.mlp.fc2_global_scales"

            add_tensor(wf, w1_name, fc1_w, TensorProto.UINT8)
            add_tensor(wf, s1_name, fc1_s, scale_dtype)
            if with_bias:
                b1_name = f"model.layers.{l_idx}.mlp.fc1_bias"
                b2_name = f"model.layers.{l_idx}.mlp.fc2_bias"
                b1_f32 = rng.uniform(-0.1, 0.1, size=(ne, 2 * inter_size)).astype(np.float32)
                b2_f32 = rng.uniform(-0.1, 0.1, size=(ne, hidden_size)).astype(np.float32)
                if use_fp16:
                    fc1_b = b1_f32.astype(np.float16).view(np.uint16)
                    fc2_b = b2_f32.astype(np.float16).view(np.uint16)
                else:
                    fc1_b = _float32_to_bfloat16_u16(b1_f32)
                    fc2_b = _float32_to_bfloat16_u16(b2_f32)
                add_tensor(wf, b1_name, fc1_b, act_onnx_dtype)
                add_tensor(wf, b2_name, fc2_b, act_onnx_dtype)
            add_tensor(wf, w2_name, fc2_w, TensorProto.UINT8)
            add_tensor(wf, s2_name, fc2_s, scale_dtype)
            add_tensor(wf, g1_name, fc1_g, TensorProto.FLOAT)
            add_tensor(wf, g2_name, fc2_g, TensorProto.FLOAT)

            r_in_name = "router_probs" if layer_experts is None else f"router_probs_{l_idx}"
            out_name = "Y" if l_idx == num_layers - 1 else f"H_{l_idx}"
            node = helper.make_node(
                "QMoE",
                inputs=[
                    cur_in,
                    r_in_name,
                    w1_name,
                    s1_name,
                    b1_name,
                    w2_name,
                    s2_name,
                    b2_name,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    g1_name,
                    g2_name,
                ],
                outputs=[out_name],
                name=f"model/layers.{l_idx}/mlp/QMoE_node_{l_idx}",
                domain="com.microsoft",
                activation_type="swiglu",
                normalize_routing_weights=1,
                k=min(top_k, ne),
                expert_weight_bits=4,
                block_size=blk_size,
                swiglu_fusion=1,
                quant_type=quant_type,
                weights_prepacked=1,
            )
            nodes.append(node)
            cur_in = out_name

    if embedded_initializers and os.path.exists(ext_path):
        os.remove(ext_path)

    # Also write a STRP v1 popularity profile (expert-profile.bin)
    prof_path = os.path.join(model_dir, "expert-profile.bin")
    pairs = []
    for e in range(num_experts):
        for l_idx in range(num_layers):
            ne = layer_experts[l_idx] if layer_experts is not None else num_experts
            if e < ne:
                pairs.append((l_idx, (e * 3 + l_idx) % ne))
    if short_profile_pairs is not None:
        pairs = pairs[:short_profile_pairs]
    with open(prof_path, "wb") as pf:
        pf.write(struct.pack("<6I", 0x50525453, 1, num_layers, num_experts, num_layers * num_experts, len(pairs)))
        pf.writelines(struct.pack("<2H", l_id, e_id) for l_id, e_id in pairs)

    graph_inputs = [helper.make_tensor_value_info("X", act_onnx_dtype, ["T", hidden_size])]
    if layer_experts is None:
        graph_inputs.append(helper.make_tensor_value_info("router_probs", act_onnx_dtype, ["T", num_experts]))
    else:
        for l_idx, ne in enumerate(layer_experts):
            graph_inputs.append(helper.make_tensor_value_info(f"router_probs_{l_idx}", act_onnx_dtype, ["T", ne]))

    graph = helper.make_graph(
        nodes,
        "qmoe_v4_test",
        graph_inputs,
        [
            helper.make_tensor_value_info("Y", act_onnx_dtype, ["T", hidden_size]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 20),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model_path = os.path.join(model_dir, "model.onnx")
    onnx.save_model(model, model_path)
    return model_path


class TestQMoEOffloadV4(unittest.TestCase):
    _use_plugin_ep = False
    _nvfp4_supported = True
    _nvfp4_skip_reason = ""
    _int4_bf16_runner_supported = True

    @staticmethod
    def _cuda_plugin_ep_is_registered() -> bool:
        """Whether a CUDA plugin EP library is already registered, which is the wheel's bundled copy after import.

        Never replace it with ORT_CUDA_PLUGIN_PATH. The bundled library stays mapped, so a second copy makes the CUDA
        runtime see every kernel twice ("Duplicate entry kernels"), after which kernel launches fail with "invalid
        argument" or hang. The bundled copy cannot be matched by content: a wheel is repaired after the build, so it
        differs from the library in the build directory even though both come from the same build.
        """
        with contextlib.suppress(Exception):
            return any(
                device.ep_name == "CUDAExecutionProvider" and device.ep_metadata.get("library_path")
                for device in ort.get_ep_devices()
            )
        return False

    @classmethod
    def setUpClass(cls):
        for env_key in ("ORT_CUDA_PLUGIN_PATH", "QWEN_PROVIDER"):
            plugin_path = os.environ.get(env_key)
            if plugin_path and os.path.exists(plugin_path):
                if cls._cuda_plugin_ep_is_registered():
                    cls._use_plugin_ep = True
                else:
                    with contextlib.suppress(Exception):
                        ort.register_execution_provider_library("CUDAExecutionProvider", plugin_path)
                        cls._use_plugin_ep = True
                break
        if not cls._use_plugin_ep and "CUDAExecutionProvider" not in ort.get_available_providers():
            transformers_dir = Path(__file__).resolve().parent / "transformers"
            if str(transformers_dir) not in sys.path:
                sys.path.insert(0, str(transformers_dir))
            with contextlib.suppress(Exception):
                from cuda_plugin_ep_helper import ensure_cuda_plugin_ep_registered  # noqa: PLC0415

                if ensure_cuda_plugin_ep_registered():
                    cls._use_plugin_ep = True
        if not cls._use_plugin_ep and "CUDAExecutionProvider" not in ort.get_available_providers():
            lib_name = (
                "onnxruntime_providers_cuda.dll" if sys.platform == "win32" else "libonnxruntime_providers_cuda.so"
            )
            ort_capi = Path(ort.__file__).resolve().parent / "capi" / lib_name
            if ort_capi.exists():
                with contextlib.suppress(Exception):
                    ort.register_execution_provider_library("CUDAExecutionProvider", str(ort_capi))
                    cls._use_plugin_ep = True
        cls._skip_unless_nvfp4_offload_supported()
        cls._probe_int4_bf16_runner_support()

    @classmethod
    def _skip_unless_nvfp4_offload_supported(cls):
        """Skips the suite where CUDA offload cannot work (no CUDA EP or SM < 80) and probes NVFP4 support."""
        plugin_dev = cls._selected_plugin_device()
        if "CUDAExecutionProvider" not in ort.get_available_providers() and plugin_dev is None:
            raise unittest.SkipTest("The CUDA execution provider is not available")

        capability = _cuda_device_compute_capability(cls._device_id(plugin_dev))
        if capability is not None and capability < (8, 0):
            raise unittest.SkipTest(
                f"QMoE offload requires an SM80 or newer GPU, found SM{capability[0]}{capability[1]}"
            )

        # USE_FP4_QMOE is a build-time option, so ask the kernel itself. When disabled, NVFP4 tests skip in setUp()
        # while INT4/INT3 offload tests still execute.
        cls._nvfp4_supported = True
        cls._nvfp4_skip_reason = ""
        with tempfile.TemporaryDirectory() as probe_dir:
            probe_path = _build_canonical_nvfp4_qmoe_model(probe_dir, num_layers=1, embedded_initializers=True)
            try:
                cls._make_session(probe_path, {"session.moe.offload.mode": "disabled"})
            except Exception as e:
                if "USE_FP4_QMOE" in str(e):
                    cls._nvfp4_supported = False
                    cls._nvfp4_skip_reason = "This build does not enable USE_FP4_QMOE (NVFP4 QMoE)"

    @classmethod
    def _probe_int4_bf16_runner_support(cls):
        """Quick builds (ORT_QUICK_BUILD) do not compile the native BF16 INT4/INT8 QMoE runner, so ask the kernel."""
        cls._int4_bf16_runner_supported = True
        with tempfile.TemporaryDirectory() as probe_dir:
            probe_path = cls._build_int_qmoe_model(probe_dir, layer_specs=[(4, 32, 1)])
            try:
                cls._make_session(probe_path, {"session.moe.offload.mode": "disabled"})
            except Exception as e:
                if "failed to construct MoE runner" in str(e):
                    cls._int4_bf16_runner_supported = False

    def _skip_unless_int4_bf16_runner(self):
        if not self._int4_bf16_runner_supported:
            self.skipTest("The native BF16 INT4 QMoE runner is not compiled into this build (e.g. ORT_QUICK_BUILD)")

    def setUp(self):
        int_only_prefixes = (
            "test_v4_int4",
            "test_v4_int3",
            "test_v4_int2",
            "test_v4_zero_cold_doorbell_",
            "test_v4_skip_zero_cold_doorbell_",
            "test_v4_tail_substitution_",
        )
        if not self._nvfp4_supported and not self._testMethodName.startswith(int_only_prefixes):
            self.skipTest(self._nvfp4_skip_reason or "This build does not enable USE_FP4_QMOE (NVFP4 QMoE)")

    @classmethod
    def _get_plugin_cuda_device(cls):
        with contextlib.suppress(Exception):
            devices = [
                d
                for d in ort.get_ep_devices()
                if d.ep_name == "CUDAExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.GPU
            ]
            if devices:
                devices.sort(key=lambda d: int(d.ep_options.get("device_id", d.ep_metadata.get("cuda_device_id", "0"))))
                return devices[0]
        return None

    @classmethod
    def _selected_plugin_device(cls, prefer_plugin: bool = False):
        """The plugin EP device _make_session binds, or None when it uses the ordinary CUDAExecutionProvider."""
        plugin_dev = cls._get_plugin_cuda_device()
        if (prefer_plugin or "CUDAExecutionProvider" not in ort.get_available_providers()) and plugin_dev is not None:
            return plugin_dev
        return None

    @staticmethod
    def _device_id(plugin_dev) -> int:
        if plugin_dev is None:
            return 0
        return int(plugin_dev.ep_options.get("device_id", plugin_dev.ep_metadata.get("cuda_device_id", "0")))

    @classmethod
    def _make_session(
        cls,
        model_path: str,
        config_entries: dict,
        cuda_options: dict | None = None,
        prefer_plugin: bool = False,
        profile_file_prefix: str | None = None,
    ):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if profile_file_prefix is not None:
            so.enable_profiling = True
            so.profile_file_prefix = profile_file_prefix
        for k, v in config_entries.items():
            so.add_session_config_entry(k, str(v))
        plugin_dev = cls._selected_plugin_device(prefer_plugin)
        if plugin_dev is not None:
            so.add_provider_for_devices([plugin_dev], cuda_options or {})
            return ort.InferenceSession(model_path, sess_options=so)
        opts = dict(cuda_options or {})
        opts.setdefault("device_id", "0")
        providers = [("CUDAExecutionProvider", opts)]
        return ort.InferenceSession(model_path, sess_options=so, providers=providers)

    def _assert_parity(
        self,
        y_ref: np.ndarray,
        y_cand: np.ndarray,
        label: str,
        rtol: float = 5e-2,
        atol: float = 2.5e-2,
        min_signal: float = 0.25,
    ):
        max_ref = float(np.max(np.abs(y_ref)))
        norm_ref = float(np.linalg.norm(y_ref))
        self.assertGreaterEqual(
            max_ref,
            min_signal,
            f"Reference signal too small in {label}: max_ref={max_ref} < min_signal={min_signal}",
        )
        max_diff = float(np.max(np.abs(y_ref - y_cand)))
        rel_l2 = float(np.linalg.norm(y_ref - y_cand)) / max(norm_ref, 1e-6)
        rel_max = max_diff / max(max_ref, 1e-6)
        self.assertLess(
            max_diff,
            max(atol, rtol * max_ref),
            f"Max abs diff mismatch in {label}: max_diff={max_diff}, max_ref={max_ref}, rel_max={rel_max}",
        )
        self.assertLess(
            rel_l2,
            min(rtol, 2.5e-2),
            f"Relative L2 mismatch in {label}: rel_l2={rel_l2}, max_diff={max_diff}, max_ref={max_ref}",
        )

    def _run_bf16(self, sess, x_f32: np.ndarray, r_f32: np.ndarray) -> np.ndarray:
        return _run_qmoe_bf16(sess, x_f32, r_f32)

    def test_discarded_prototype_features_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(tmpdir)
            # 1. Unknown session.moe.offload.* key must fail with INVALID_ARGUMENT
            with self.assertRaisesRegex(Exception, "Unknown session.moe.offload"):
                self._make_session(model_path, {"session.moe.offload.unknown_knob": "1"})

            # 2. Discarded prototype moe.cpu_offload.* key must fail with INVALID_ARGUMENT
            with self.assertRaisesRegex(Exception, "discarded and not supported"):
                self._make_session(model_path, {"moe.cpu_offload.experts": "4"})

            # 3. Discarded prototype placement strategy 'basic' must fail with INVALID_ARGUMENT
            with self.assertRaisesRegex(Exception, "expected 'static' or 'adaptive'"):
                self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "basic",
                    },
                )

            # 4. Separate QMoEOffload operator schema must no longer exist
            m = onnx.load_model(model_path, load_external_data=False)
            m.graph.node[0].op_type = "QMoEOffload"
            legacy_path = os.path.join(tmpdir, "legacy_qmoe_offload.onnx")
            onnx.save_model(m, legacy_path)
            with self.assertRaisesRegex(Exception, "QMoEOffload"):
                self._make_session(legacy_path, {})

    def test_v4_portable_qmoe_modes_and_numerical_parity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir, num_layers=2, num_experts=16, hidden_size=128, inter_size=64, top_k=4
            )
            # Per-expert slot bytes for H=128, I=64, block=16:
            # 2*64*64 + 128*32 + 2*64*8 + 128*4 + 8 = 8192 + 4096 + 1024 + 512 + 8 = 13,832 B
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4  # 128 B per layer

            # 1. Baseline: mode=disabled (Full GPU resident)
            sess_base = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})

            # 2. mode=auto with ample VRAM (Full GPU residency admitted via Stage 1->4)
            sess_auto_full = self._make_session(model_path, {"session.moe.offload.mode": "auto"})

            # 3. mode=required, static placement (6 resident slots/layer = 12 total slots across 2 layers, spare=0)
            budget_12_slots = 2 * fixed_table_bytes + 12 * slot_bytes
            sess_static = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(budget_12_slots),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )

            # 4. mode=required, adaptive placement (6 resident + 2 spare slots/layer = 16 total physical slots)
            budget_adaptive = 2 * fixed_table_bytes + 16 * slot_bytes
            sess_adaptive = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )

            # 5. mode=required with bounded host_expert_bytes (FileExpertStore bounded host cache)
            # Aligned host stride for H=128, I=64 is 14,336 B (8192 + 1024 + 256 + 4096 + 512 + 256).
            host_slot_stride = 14336
            sess_bounded_host = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.host_expert_bytes": str(2 * 4 * host_slot_stride),  # 4 cold slots/layer
                    "session.moe.offload.allow_file_backing": "1",
                    "session.moe.offload.budget_group": "",
                },
            )

            # 6. mode=required with cpu_execution="disabled" (GPU landing-ring weight streaming for all row counts)
            sess_gpu_stream_only = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.cpu_execution": "disabled",
                    "session.moe.offload.device_expert_bytes": str(budget_12_slots),
                },
            )

            rng = np.random.default_rng(123)
            # Test both decode regime (T=2 <= async_max_tokens=4) and prefill regime (T=16 > async_max_tokens=4)
            for num_rows in [1, 2, 4, 16]:
                x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 16), dtype=np.float32)

                y_np_ref = _numpy_nvfp4_qmoe_reference(model_path, x, r, use_fp16=False)
                y_np_dropped = _numpy_nvfp4_qmoe_reference(model_path, x, r, use_fp16=False, dropped_rank=3)
                y_base_f32 = self._run_bf16(sess_base, x, r)

                # Verify that baseline GPU QMoE matches the independent NumPy reference and has strong signal
                self._assert_parity(y_np_ref, y_base_f32, f"numpy_vs_gpu_base(T={num_rows})")
                # Verify that an all-zero output and a single-dropped-expert output are decisively rejected
                self.assertGreater(float(np.max(np.abs(y_base_f32))), 0.4)
                self.assertGreater(
                    float(np.linalg.norm(y_base_f32 - y_np_dropped)) / float(np.linalg.norm(y_base_f32)),
                    0.10,
                    f"Single dropped expert contribution must change output significantly at T={num_rows}",
                )

                for name, s in [
                    ("auto_full", sess_auto_full),
                    ("static_hybrid", sess_static),
                    ("adaptive_hybrid", sess_adaptive),
                    ("bounded_host", sess_bounded_host),
                    ("gpu_stream_only", sess_gpu_stream_only),
                ]:
                    y_cand_f32 = self._run_bf16(s, x, r)
                    self._assert_parity(y_base_f32, y_cand_f32, f"mode={name}(T={num_rows})")
                    self._assert_parity(y_np_ref, y_cand_f32, f"numpy_vs_mode={name}(T={num_rows})")

            # Also verify FP16 activation parity across baseline, offload, and independent NumPy reference
            dir_fp16 = os.path.join(tmpdir, "fp16")
            os.makedirs(dir_fp16, exist_ok=True)
            model_fp16 = _build_canonical_nvfp4_qmoe_model(
                dir_fp16, num_layers=2, num_experts=16, hidden_size=128, inter_size=64, top_k=4, use_fp16=True
            )
            sess_fp16_base = self._make_session(model_fp16, {"session.moe.offload.mode": "disabled"})
            sess_fp16_off = self._make_session(
                model_fp16,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )
            for num_rows in [1, 4, 16]:
                x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 16), dtype=np.float32)
                y_fp16_np = _numpy_nvfp4_qmoe_reference(model_fp16, x, r, use_fp16=True)
                y_fp16_base = self._run_fp16(sess_fp16_base, x, r)
                y_fp16_off = self._run_fp16(sess_fp16_off, x, r)
                self._assert_parity(y_fp16_np, y_fp16_base, f"fp16_numpy_vs_base(T={num_rows})")
                self._assert_parity(y_fp16_base, y_fp16_off, f"fp16_base_vs_off(T={num_rows})", rtol=2e-2, atol=1.5e-2)
            del sess_fp16_off, sess_fp16_base

    def test_v4_eligibility_biases_boundaries_and_multi_session_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a = os.path.join(tmpdir, "model_a")
            dir_b = os.path.join(tmpdir, "model_b")
            dir_bias = os.path.join(tmpdir, "model_bias")
            os.makedirs(dir_a, exist_ok=True)
            os.makedirs(dir_b, exist_ok=True)
            os.makedirs(dir_bias, exist_ok=True)

            model_a = _build_canonical_nvfp4_qmoe_model(dir_a, seed=101, short_profile_pairs=4)
            model_b = _build_canonical_nvfp4_qmoe_model(dir_b, seed=202, short_profile_pairs=4)
            model_bias = _build_canonical_nvfp4_qmoe_model(dir_bias, seed=303, with_bias=True)

            slot_bytes = 13832
            host_slot_stride = 14336
            fixed_table_bytes = 2 * 16 * 4
            budget_adaptive = 2 * fixed_table_bytes + 16 * slot_bytes

            # 1. Exact 1-slot host cache boundary (F07, F10):
            # 0 bytes or 1 byte below 2 * host_slot_stride must fail fast instead of falling back to unbounded mmap.
            with self.assertRaisesRegex(Exception, "host_expert_bytes"):
                self._make_session(
                    model_a,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                        "session.moe.offload.host_expert_bytes": "0",
                    },
                )
            with self.assertRaisesRegex(Exception, "host_expert_bytes"):
                self._make_session(
                    model_a,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                        "session.moe.offload.host_expert_bytes": str(2 * host_slot_stride - 1),
                    },
                )
            sess_one_slot_host = self._make_session(
                model_a,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.host_expert_bytes": str(2 * host_slot_stride),
                },
            )
            del sess_one_slot_host

            # 2. Missing explicit popularity_file and invalid node_overrides must fail fast (F13, F14).
            with self.assertRaisesRegex(Exception, "popularity_file"):
                self._make_session(
                    model_a,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.popularity_file": "nonexistent_profile.bin",
                    },
                )
            with self.assertRaisesRegex(Exception, "node_overrides"):
                self._make_session(
                    model_a,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.node_overrides": "malformed_override_without_colon",
                    },
                )

            # 3. Unsupported swiglu_fusion=2 must be rejected before relocation in mode=required (F03, F04).
            m_bad = onnx.load_model(model_a, load_external_data=False)
            for attr in m_bad.graph.node[0].attribute:
                if attr.name == "swiglu_fusion":
                    attr.i = 2
            bad_path = os.path.join(dir_a, "model_bad_swiglu.onnx")
            onnx.save_model(m_bad, bad_path)
            with self.assertRaisesRegex(Exception, "swiglu_fusion"):
                self._make_session(bad_path, {"session.moe.offload.mode": "required"})

            # 4. Nonzero FC1/FC2 biases parity across cold, hot, and promoted execution (F05).
            fc1_bias_per_expert = 2 * 64 * 2  # 256 B
            fc2_bias_per_layer = 16 * 128 * 2  # 4096 B
            bias_slot_bytes = slot_bytes + fc1_bias_per_expert
            # The node keeps its own full FC1/FC2 bias banks besides the backend's per-slot FC1 bias and FC2 buffer.
            bias_fixed_bytes = fixed_table_bytes + 2 * fc2_bias_per_layer + 16 * fc1_bias_per_expert
            bias_budget = 2 * bias_fixed_bytes + 12 * bias_slot_bytes
            sess_bias_ref = self._make_session(model_bias, {"session.moe.offload.mode": "disabled"})
            sess_bias_off = self._make_session(
                model_bias,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(bias_budget),
                },
            )
            rng = np.random.default_rng(777)
            for step in range(4):
                x = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((2, 16), dtype=np.float32)
                y_ref = self._run_bf16(sess_bias_ref, x, r)
                y_off = self._run_bf16(sess_bias_off, x, r)
                self._assert_parity(y_ref, y_off, f"bias_parity(step={step})")

            # 5. Two same-shaped sessions with different weights on the same GPU (F11, F18).
            sess_a_ref = self._make_session(model_a, {"session.moe.offload.mode": "disabled"})
            sess_b_ref = self._make_session(model_b, {"session.moe.offload.mode": "disabled"})
            sess_a_off = self._make_session(
                model_a,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                    "session.moe.offload.node_overrides": "model/layers.0/mlp/QMoE_node_0:max_resident_slots=6",
                },
            )
            sess_b_off = self._make_session(
                model_b,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )
            for num_rows in [2, 16, 2]:
                x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 16), dtype=np.float32)
                ya_ref = self._run_bf16(sess_a_ref, x, r)
                yb_ref = self._run_bf16(sess_b_ref, x, r)
                ya_off = self._run_bf16(sess_a_off, x, r)
                yb_off = self._run_bf16(sess_b_off, x, r)
                self.assertGreater(float(np.max(np.abs(ya_ref - yb_ref))), 0.1)
                self._assert_parity(ya_ref, ya_off, f"sess_a_isolation(T={num_rows})")
                self._assert_parity(yb_ref, yb_off, f"sess_b_isolation(T={num_rows})")

    def test_v4_cuda_graph_capture_and_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir, num_layers=2, num_experts=16, hidden_size=128, inter_size=64, top_k=4
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            budget_adaptive = 2 * fixed_table_bytes + 16 * slot_bytes

            rng = np.random.default_rng(999)
            x = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
            r = rng.standard_normal((2, 16), dtype=np.float32)
            sess_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})

            # Wrap inputs/outputs in FLOAT <-> BFLOAT16 Cast on GPU so CUDA Graph capture has 100% GPU I/O
            cg_model_path = _wrap_qmoe_model_with_cast_io(model_path, os.path.join(tmpdir, "model_cg.onnx"), 128, 16)

            devices = [
                d
                for d in ort.get_ep_devices()
                if d.ep_name == "CUDAExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.GPU
            ]
            devices.sort(key=lambda d: int(d.ep_options.get("device_id", d.ep_metadata.get("cuda_device_id", "0"))))
            dev_id = (
                int(devices[0].ep_options.get("device_id", devices[0].ep_metadata.get("cuda_device_id", "0")))
                if devices
                else 0
            )

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            for k, v in {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "adaptive",
                "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                "session.moe.offload.popularity_file": "expert-profile.bin",
            }.items():
                so.add_session_config_entry(k, v)
            if devices:
                so.add_provider_for_devices(
                    [devices[0]], {"enable_cuda_graph": "1", "min_num_runs_before_cuda_graph_capture": "2"}
                )
                sess_graph = ort.InferenceSession(cg_model_path, sess_options=so)
            else:
                sess_graph = ort.InferenceSession(
                    cg_model_path, sess_options=so, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})]
                )

            def round_bf16(a: np.ndarray) -> np.ndarray:
                return _bfloat16_u16_to_float32(_float32_to_bfloat16_u16(a))

            cudart = _load_cudart()
            if cudart is None:
                self.skipTest("CUDA runtime library is required to update captured GPU inputs in place")

            def device_buffer(nbytes: int) -> ctypes.c_void_p:
                ptr = ctypes.c_void_p()
                self.assertEqual(cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), 0)
                return ptr

            def upload(ptr: ctypes.c_void_p, arr: np.ndarray):
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                self.assertEqual(cudart.cudaMemcpy(ptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, 1), 0)

            # Update the captured GPU inputs in place with different activations and routes on every replay so a
            # stale doorbell, host-result or scratch buffer cannot match the reference by accident.
            x_ptr = device_buffer(x.nbytes)
            r_ptr = device_buffer(r.nbytes)
            try:
                x_in = round_bf16(x)
                r_in = round_bf16(r)
                upload(x_ptr, x_in)
                upload(r_ptr, r_in)
                binding = sess_graph.io_binding()
                binding.bind_input("X_f32", "cuda", dev_id, np.float32, list(x.shape), x_ptr.value)
                binding.bind_input("R_f32", "cuda", dev_id, np.float32, list(r.shape), r_ptr.value)
                binding.bind_output("Y_f32", "cuda", dev_id)

                replay_rng = np.random.default_rng(31337)
                for step in range(8):
                    if step > 0:
                        x_in = round_bf16(replay_rng.standard_normal((2, 128), dtype=np.float32) * 0.5)
                        r_in = round_bf16(replay_rng.standard_normal((2, 16), dtype=np.float32) * 2.0)
                        upload(x_ptr, x_in)
                        upload(r_ptr, r_in)
                    y_ref = self._run_bf16(sess_ref, x_in, r_in)
                    sess_graph.run_with_iobinding(binding)
                    y_cand = binding.copy_outputs_to_cpu()[0]
                    self._assert_parity(y_ref, y_cand, f"cuda_graph_replay(step={step})")
            finally:
                cudart.cudaFree(x_ptr)
                cudart.cudaFree(r_ptr)

    def _run_fp16(self, sess, x_f32: np.ndarray, r_f32: np.ndarray) -> np.ndarray:
        x_f16 = x_f32.astype(np.float16)
        r_f16 = r_f32.astype(np.float16)
        out = sess.run(["Y"], {"X": x_f16, "router_probs": r_f16})[0]
        return out.astype(np.float32)

    def test_v4_mxfp4_offload_is_rejected_when_required_and_ignored_when_auto(self):
        # Only NVFP4 with weights_prepacked=1 defines the K-packed row-major layout the offload kernels read, so
        # MXFP4 nodes must neither be relocated nor reinterpreted.
        with tempfile.TemporaryDirectory() as tmpdir:
            for use_fp16 in (False, True):
                sub_dir = os.path.join(tmpdir, "fp16" if use_fp16 else "bf16")
                os.makedirs(sub_dir, exist_ok=True)
                model_path = _build_canonical_nvfp4_qmoe_model(
                    sub_dir,
                    num_layers=2,
                    num_experts=16,
                    hidden_size=128,
                    inter_size=64,
                    top_k=4,
                    seed=515 + int(use_fp16),
                    with_bias=True,
                    quant_type="fp4",
                    use_fp16=use_fp16,
                )
                with self.assertRaisesRegex(Exception, "expected 'nvfp4'"):
                    self._make_session(model_path, {"session.moe.offload.mode": "required"})

                sess_base = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})
                sess_auto = self._make_session(model_path, {"session.moe.offload.mode": "auto"})
                rng = np.random.default_rng(808 + int(use_fp16))
                runner = self._run_fp16 if use_fp16 else self._run_bf16
                for num_rows in (1, 4, 16):
                    x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                    r = rng.standard_normal((num_rows, 16), dtype=np.float32)
                    np.testing.assert_array_equal(runner(sess_base, x, r), runner(sess_auto, x, r))

            # Also verify unsupported quant_type="int" configurations (raw INT4 weights_prepacked=0,
            # default/omitted INT4 weights_prepacked=-1, and prepacked INT3 weights_prepacked=1) are rejected
            # in mode="required", while supported prepacked INT4 (bits=4, weights_prepacked=1, block_size=32)
            # and raw INT3 (bits=3, weights_prepacked=0, block_size=32) are admitted.
            nv_dir = os.path.join(tmpdir, "nv_for_int")
            os.makedirs(nv_dir, exist_ok=True)
            base_nv = _build_canonical_nvfp4_qmoe_model(
                nv_dir,
                num_layers=1,
                num_experts=16,
                hidden_size=128,
                inter_size=128,
                seed=520,
                embedded_initializers=True,
            )
            for bits, prepacked, should_succeed, tag in (
                (4, 0, False, "int4_raw"),
                (4, -1, False, "int4_default_prepacked"),
                (3, 1, False, "int3_prepacked"),
                (4, 1, True, "int4_prepacked"),
                (3, 0, True, "int3_raw"),
            ):
                if should_succeed and bits == 4 and not self._int4_bf16_runner_supported:
                    continue  # Admitting this BF16 INT4 node builds the runner that quick builds do not compile.
                m_int = onnx.load_model(base_nv)
                del m_int.graph.node[0].input[15:]
                for init in m_int.graph.initializer:
                    if init.name.endswith("fc1_experts_weights"):
                        w_shape = (16, 256, (128 * 3) // 8) if bits == 3 else (16, 128, 128)
                        del init.dims[:]
                        init.dims.extend(w_shape)
                        init.raw_data = np.zeros(w_shape, dtype=np.uint8).tobytes()
                    elif init.name.endswith("fc2_experts_weights"):
                        w_shape = (16, 128, (128 * 3) // 8) if bits == 3 else (16, 128, 64)
                        del init.dims[:]
                        init.dims.extend(w_shape)
                        init.raw_data = np.zeros(w_shape, dtype=np.uint8).tobytes()
                    elif init.name.endswith("fc1_scales"):
                        s_shape = (16, 256, 128 // 32)
                        init.data_type = TensorProto.BFLOAT16
                        del init.dims[:]
                        init.dims.extend(s_shape)
                        init.raw_data = np.zeros(s_shape, dtype=np.uint16).tobytes()
                    elif init.name.endswith("fc2_scales"):
                        s_shape = (16, 128, 128 // 32)
                        init.data_type = TensorProto.BFLOAT16
                        del init.dims[:]
                        init.dims.extend(s_shape)
                        init.raw_data = np.zeros(s_shape, dtype=np.uint16).tobytes()
                for attr in m_int.graph.node[0].attribute:
                    if attr.name == "quant_type":
                        attr.s = b"int"
                    elif attr.name == "expert_weight_bits":
                        attr.i = bits
                    elif attr.name == "weights_prepacked":
                        attr.i = prepacked
                    elif attr.name == "block_size":
                        attr.i = 32
                int_path = os.path.join(nv_dir, f"model_{tag}.onnx")
                onnx.save_model(m_int, int_path)
                if should_succeed:
                    s_ok = self._make_session(int_path, {"session.moe.offload.mode": "required"})
                    del s_ok
                else:
                    with self.assertRaisesRegex(
                        Exception, "quant_type='int' offload requires|rejected node|INT3 requires.*weights_prepacked=0"
                    ):
                        self._make_session(int_path, {"session.moe.offload.mode": "required"})

    def test_v4_embedded_heap_weights_bounded_cache_eviction_safety(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Embedded initializers (heap-backed, 2 pages/expert for fc1_w) with bounded host cache (1 cold slot/layer).
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=2,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=4,
                seed=606,
                embedded_initializers=True,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            host_slot_stride = 14336
            budget_small = 2 * fixed_table_bytes + 8 * slot_bytes  # 4 slots/layer (2 resident + 2 spare)

            sess_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})
            sess_bounded_embedded = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_small),
                    "session.moe.offload.host_expert_bytes": str(2 * host_slot_stride),  # 1 cold slot/layer
                    "session.moe.offload.allow_file_backing": "1",
                },
            )
            rng = np.random.default_rng(607)
            # Route specifically to cold uncached experts (experts 12..15) across decode, prefill, and follow-up decode.
            for num_rows in (2, 16, 2):
                x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                r = np.full((num_rows, 16), -10.0, dtype=np.float32)
                for row in range(num_rows):
                    for k_idx in range(4):
                        r[row, 15 - ((row + k_idx) % 8)] = 5.0 - 0.5 * k_idx
                y_ref = self._run_bf16(sess_ref, x, r)
                y_got = self._run_bf16(sess_bounded_embedded, x, r)
                self._assert_parity(y_ref, y_got, f"embedded_heap_bounded_cache(T={num_rows})")

    def test_v4_named_budget_group_with_non_moe_gpu_initializers_and_failed_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mixed model with non-MoE GPU initializers allocated during SaveInitializedTensors before CreateKernels.
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=2,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=4,
                seed=707,
                non_moe_gpu_init_elements=128 * 256,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            budget_adaptive = 2 * fixed_table_bytes + 16 * slot_bytes

            sess_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})
            sess_grouped = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.budget_group": "test_mixed_group_v4",
                    "session.moe.offload.gpu_device_mem_limit": str(64 * 1024 * 1024),
                },
            )
            rng = np.random.default_rng(708)
            x = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
            r = rng.standard_normal((2, 16), dtype=np.float32)
            y_ref = self._run_bf16(sess_ref, x, r)
            y_grp = self._run_bf16(sess_grouped, x, r)
            self._assert_parity(y_ref, y_grp, "named_budget_group_parity")

            # A failed initialization in a named budget group after non-MoE GPU initializers were allocated
            # must clean up cleanly without underflowing group accounting or terminating the process.
            with self.assertRaisesRegex(Exception, "host_expert_bytes"):
                self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                        "session.moe.offload.host_expert_bytes": "0",
                        "session.moe.offload.budget_group": "test_mixed_group_v4_fail",
                        "session.moe.offload.gpu_device_mem_limit": str(32 * 1024 * 1024),
                    },
                )

    def test_v4_prefill_histogram_across_incompatible_expert_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_e16 = os.path.join(tmpdir, "e16")
            dir_e64 = os.path.join(tmpdir, "e64")
            dir_het = os.path.join(tmpdir, "het")
            os.makedirs(dir_e16, exist_ok=True)
            os.makedirs(dir_e64, exist_ok=True)
            os.makedirs(dir_het, exist_ok=True)

            m_e16 = _build_canonical_nvfp4_qmoe_model(dir_e16, num_layers=2, num_experts=16, seed=811)
            m_e64 = _build_canonical_nvfp4_qmoe_model(dir_e64, num_layers=2, num_experts=64, seed=812)
            m_het = _build_canonical_nvfp4_qmoe_model(
                dir_het, num_layers=2, num_experts=32, layer_experts=[16, 32], seed=813
            )

            slot_bytes = 13832
            rng = np.random.default_rng(814)

            for m_path, ne in ((m_e16, 16), (m_e64, 64)):
                fixed_bytes = 2 * ne * 4
                budget = 2 * fixed_bytes + 12 * slot_bytes
                s_ref = self._make_session(m_path, {"session.moe.offload.mode": "disabled"})
                s_off = self._make_session(
                    m_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.device_expert_bytes": str(budget),
                    },
                )
                for num_rows in (16, 2):
                    x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                    r = rng.standard_normal((num_rows, ne), dtype=np.float32)
                    y_ref = self._run_bf16(s_ref, x, r)
                    y_off = self._run_bf16(s_off, x, r)
                    self._assert_parity(y_ref, y_off, f"hist_ne{ne}(T={num_rows})")
                del s_off
                del s_ref

            # Heterogeneous layers (layer 0 has E=16, layer 1 has E=32)
            het_budget = (2 * 16 * 4 + 2 * 32 * 4) + 12 * slot_bytes
            s_het_ref = self._make_session(m_het, {"session.moe.offload.mode": "disabled"})
            s_het_off = self._make_session(
                m_het,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(het_budget),
                },
            )
            for num_rows in (16, 2):
                x_bf16 = _float32_to_bfloat16_u16(rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5)
                r0_bf16 = _float32_to_bfloat16_u16(rng.standard_normal((num_rows, 16), dtype=np.float32))
                r1_bf16 = _float32_to_bfloat16_u16(rng.standard_normal((num_rows, 32), dtype=np.float32))
                feeds = {
                    "X": ort.OrtValue.ortvalue_from_numpy_with_onnx_type(x_bf16, TensorProto.BFLOAT16),
                    "router_probs_0": ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r0_bf16, TensorProto.BFLOAT16),
                    "router_probs_1": ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r1_bf16, TensorProto.BFLOAT16),
                }
                out_ref = s_het_ref.run_with_ort_values(["Y"], feeds)[0]
                out_off = s_het_off.run_with_ort_values(["Y"], feeds)[0]
                b_ref = bytes((ctypes.c_uint16 * (num_rows * 128)).from_address(out_ref.data_ptr()))
                b_off = bytes((ctypes.c_uint16 * (num_rows * 128)).from_address(out_off.data_ptr()))
                y_ref = _bfloat16_u16_to_float32(np.frombuffer(b_ref, dtype=np.uint16))
                y_off = _bfloat16_u16_to_float32(np.frombuffer(b_off, dtype=np.uint16))
                self._assert_parity(y_ref, y_off, f"hist_het(T={num_rows})")

    def test_v4_pinned_bytes_exact_rounded_and_multilayer_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_1l = os.path.join(tmpdir, "l1")
            dir_2l = os.path.join(tmpdir, "l2")
            os.makedirs(dir_1l, exist_ok=True)
            os.makedirs(dir_2l, exist_ok=True)
            m_1l = _build_canonical_nvfp4_qmoe_model(dir_1l, num_layers=1, num_experts=16, seed=901)
            m_2l = _build_canonical_nvfp4_qmoe_model(dir_2l, num_layers=2, num_experts=16, seed=902)

            slot_bytes = 13832
            fixed_bytes = 2 * 16 * 4
            budget_1l = fixed_bytes + 6 * slot_bytes
            budget_2l = 2 * fixed_bytes + 12 * slot_bytes
            # Actual rounded persistent decode host buffer for E=16:
            # max(min_host_bytes, 256 KiB) + 512 + align256(16 * 4) = 262,144 + 512 + 256 = 262,912 B
            per_layer_decode_pinned = 262912

            rng = np.random.default_rng(903)
            x_dec = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
            r_dec = rng.standard_normal((2, 16), dtype=np.float32)

            # 1-layer below-256KiB (64 KiB) and 1-byte-below boundary (262,911 B) must be rejected before allocation.
            for bad_pinned in (65536, per_layer_decode_pinned - 1):
                s_bad = self._make_session(
                    m_1l,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(budget_1l),
                        "session.moe.offload.pinned_bytes": str(bad_pinned),
                    },
                )
                with self.assertRaisesRegex(Exception, "pinned"):
                    self._run_bf16(s_bad, x_dec, r_dec)
                del s_bad

            # Exact 1-layer boundary (262,912 B) must succeed.
            s_ok_1l = self._make_session(
                m_1l,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.device_expert_bytes": str(budget_1l),
                    "session.moe.offload.pinned_bytes": str(per_layer_decode_pinned),
                },
            )
            _ = self._run_bf16(s_ok_1l, x_dec, r_dec)
            del s_ok_1l

            # 2-layer aggregate limit: 300 KiB or (2 * 262,912 - 1) must reject on the second layer.
            for bad_2l_pinned in (300000, 2 * per_layer_decode_pinned - 1):
                s_bad_2l = self._make_session(
                    m_2l,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(budget_2l),
                        "session.moe.offload.pinned_bytes": str(bad_2l_pinned),
                    },
                )
                with self.assertRaisesRegex(Exception, "pinned"):
                    self._run_bf16(s_bad_2l, x_dec, r_dec)
                del s_bad_2l

            # Exact 2-layer aggregate boundary (525,824 B) must succeed for decode.
            s_ok_2l = self._make_session(
                m_2l,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.device_expert_bytes": str(budget_2l),
                    "session.moe.offload.pinned_bytes": str(2 * per_layer_decode_pinned),
                },
            )
            _ = self._run_bf16(s_ok_2l, x_dec, r_dec)
            del s_ok_2l

            # Prefill host staging (2 * host_pass_bytes = 2 * 111,360 = 222,720 B for E=16, H=128, I=64, wave=8):
            # 222,719 B must be rejected before cudaHostAlloc, while 222,720 B must succeed for prefill.
            prefill_staging_total = 2 * 111360
            s_pref_bad = self._make_session(
                m_1l,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.device_expert_bytes": str(budget_1l),
                    "session.moe.offload.pinned_bytes": str(prefill_staging_total - 1),
                },
            )
            x_pref = rng.standard_normal((16, 128), dtype=np.float32) * 0.5
            r_pref = rng.standard_normal((16, 16), dtype=np.float32)
            with self.assertRaisesRegex(Exception, "pinned"):
                self._run_bf16(s_pref_bad, x_pref, r_pref)
            del s_pref_bad

            s_pref_ok = self._make_session(
                m_1l,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.device_expert_bytes": str(budget_1l),
                    "session.moe.offload.pinned_bytes": str(prefill_staging_total),
                },
            )
            _ = self._run_bf16(s_pref_ok, x_pref, r_pref)

    def test_v4_ordinary_cuda_ep_and_plugin_controller_budget_enforcement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir, num_layers=2, num_experts=16, hidden_size=128, inter_size=64, top_k=4, seed=950
            )
            slot_bytes = 13832
            fixed_bytes = 2 * 16 * 4
            budget_partial = 2 * fixed_bytes + 12 * slot_bytes

            # 1. Explicit CUDAExecutionProvider partial-residency hybrid decode (cpu_execution="auto", T=1, 2, 4),
            # including plugin EP device binding when available.
            sess_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})
            sess_ord = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.cpu_execution": "auto",
                    "session.moe.offload.device_expert_bytes": str(budget_partial),
                },
                prefer_plugin=True,
            )

            rng = np.random.default_rng(951)
            for t_rows in (1, 2, 4):
                x = rng.standard_normal((t_rows, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((t_rows, 16), dtype=np.float32)
                y_ref = self._run_bf16(sess_ref, x, r)
                y_ord = self._run_bf16(sess_ord, x, r)
                self._assert_parity(y_ref, y_ord, f"ordinary_or_plugin_decode(T={t_rows})")
            del sess_ord
            del sess_ref

            # 2. Verify QMoESessionController holds the session's BudgetAccount and enforces gpu_device_mem_limit
            # in both session planning and arena/landing-ring allocations (Finding 6).
            with self.assertRaisesRegex(Exception, "MoE Offload session admission failed"):
                self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_working_bytes": str(8 * 1024 * 1024),
                        "session.moe.offload.gpu_device_mem_limit": str(4 * 1024 * 1024),
                    },
                )
            with self.assertRaisesRegex(Exception, "(?i)budget|memory|allocation"):
                sess_tight = self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.cpu_execution": "disabled",
                        "session.moe.offload.device_expert_bytes": str(budget_partial),
                        "session.moe.offload.gpu_device_mem_limit": str(budget_partial + 64 * 1024),
                    },
                )
                self._run_bf16(
                    sess_tight,
                    np.ones((1, 128), dtype=np.float32),
                    np.ones((1, 16), dtype=np.float32),
                )

    @unittest.skipIf(sys.platform == "win32", "Requires POSIX g++ toolchain for standalone header/allocator probe.")
    def test_v4_budget_group_migration_and_concurrent_allocator_address_reuse(self):
        repo_root = Path(__file__).resolve().parents[3]
        budget_hdr = repo_root / "onnxruntime/core/providers/cuda/cuda_device_memory_budget.h"
        if not budget_hdr.is_file():
            self.skipTest(f"Header not found at {budget_hdr}")
        probe_src = r"""
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "core/providers/cuda/cuda_device_memory_budget.h"

int main() {
  using namespace onnxruntime;
  // 1. Committed allocation attached to a new BudgetGroup before BeginFree/FinishFree.
  {
    auto phys = std::make_shared<CudaDeviceMemoryBudget>();
    SessionBudgetAccount acct(phys, 1024 * 1024);
    acct.Reserve(4096).Commit();
    auto grp = std::make_shared<BudgetGroup>(phys, 512 * 1024);
    acct.SetBudgetGroup(grp);
    assert(acct.GetAccountStats().committed == 4096);
    assert(acct.GetGroupStats().committed == 4096);
    acct.BeginFree(4096);
    acct.FinishFree(4096, true);
    assert(acct.GetAccountStats().committed == 0);
    assert(acct.GetGroupStats().committed == 0);
    assert(phys->GetStats().committed == 0);
  }
  // 2. Pending reservation and in-flight freeing migrated across SetBudgetGroup.
  {
    auto phys = std::make_shared<CudaDeviceMemoryBudget>();
    SessionBudgetAccount acct(phys, 1024 * 1024);
    auto res_pending = acct.Reserve(2048);
    acct.Reserve(8192).Commit();
    acct.BeginFree(8192);
    auto grp = std::make_shared<BudgetGroup>(phys, 512 * 1024);
    acct.SetBudgetGroup(grp);
    res_pending.Commit();
    acct.FinishFree(8192, true);
    assert(acct.GetGroupStats().committed == 2048);
    acct.BeginFree(2048);
    acct.FinishFree(2048, true);
    assert(acct.GetAccountStats().committed == 0);
    assert(acct.GetGroupStats().committed == 0);
    assert(phys->GetStats().committed == 0);
  }
  // 3. Simulated concurrent address reuse between driver free and metadata cleanup.
  {
    enum class State : uint8_t { kCommitted = 0, kFreeing = 1, kQuarantined = 2 };
    struct Rec { size_t bytes; uint64_t generation; State state; };
    std::mutex mu;
    uint64_t next_gen = 0;
    std::unordered_map<void*, Rec> allocs;
    auto phys = std::make_shared<CudaDeviceMemoryBudget>();
    SessionBudgetAccount acct(phys, 1024 * 1024);
    void* reused_addr = reinterpret_cast<void*>(uintptr_t{0xdeadbeef0000ULL});

    auto alloc_at = [&](void* ptr, size_t sz) {
      auto r = acct.Reserve(sz);
      std::lock_guard<std::mutex> lk(mu);
      const uint64_t g = ++next_gen;
      auto [it, ins] = allocs.emplace(ptr, Rec{sz, g, State::kCommitted});
      if (!ins) {
        assert(it->second.state != State::kCommitted);
        it->second = Rec{sz, g, State::kCommitted};
      }
      r.Commit();
    };

    alloc_at(reused_addr, 4096);
    // Begin Free(reused_addr) on thread A
    uint64_t freeing_gen = 0;
    size_t freeing_bytes = 0;
    {
      std::lock_guard<std::mutex> lk(mu);
      auto it = allocs.find(reused_addr);
      assert(it != allocs.end() && it->second.state == State::kCommitted);
      it->second.state = State::kFreeing;
      freeing_bytes = it->second.bytes;
      freeing_gen = it->second.generation;
      acct.BeginFree(freeing_bytes);
    }
    // Driver frees reused_addr; before thread A reacquires mu, thread B allocates the same address!
    alloc_at(reused_addr, 16384);
    // Thread A reacquires mu to finish its free:
    {
      std::lock_guard<std::mutex> lk(mu);
      acct.FinishFree(freeing_bytes, true);
      auto it = allocs.find(reused_addr);
      if (it != allocs.end() && it->second.generation == freeing_gen) {
        allocs.erase(it);
      }
    }
    // The new 16384-byte allocation at reused_addr must still be tracked and committed!
    assert(acct.GetAccountStats().committed == 16384);
    assert(allocs.count(reused_addr) == 1 && allocs[reused_addr].bytes == 16384);
    // Freeing the second allocation must discharge 100% of the budget to 0.
    {
      std::lock_guard<std::mutex> lk(mu);
      auto it = allocs.find(reused_addr);
      assert(it != allocs.end() && it->second.state == State::kCommitted);
      const size_t b2 = it->second.bytes;
      const uint64_t g2 = it->second.generation;
      it->second.state = State::kFreeing;
      acct.BeginFree(b2);
      acct.FinishFree(b2, true);
      if (it->second.generation == g2) allocs.erase(it);
    }
    assert(acct.GetAccountStats().committed == 0);
    assert(phys->GetStats().committed == 0);
  }
  return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = Path(tmpdir) / "probe.cc"
            bin_file = Path(tmpdir) / "probe"
            src_file.write_text(probe_src)
            subprocess.check_call(
                ["g++", "-O2", "-std=c++17", f"-I{repo_root / 'onnxruntime'}", str(src_file), "-o", str(bin_file)]
            )
            subprocess.check_call([str(bin_file)])

    def test_v4_one_slot_bounded_host_cache_promotion_and_demand_cold_experts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=2,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=4,
                seed=1101,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            host_slot_stride = 14336
            # 4 resident + 2 spare slots per layer (12 total slots across 2 layers), plus 1 host cache slot per layer
            budget_small = 2 * fixed_table_bytes + 12 * slot_bytes
            sess = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_small),
                    "session.moe.offload.host_expert_bytes": str(2 * host_slot_stride),
                    "session.moe.offload.allow_file_backing": "1",
                },
            )
            rng = np.random.default_rng(1102)
            # Step 0..5: repeatedly route to cold experts [12, 13, 14, 15] to trigger background promotions
            for step in range(6):
                x = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
                r = np.full((2, 16), -10.0, dtype=np.float32)
                for row in range(2):
                    r[row, [12, 13, 14, 15]] = [6.0, 5.0, 4.0, 3.0]
                y_np = _numpy_nvfp4_qmoe_reference(model_path, x, r)
                y_got = self._run_bf16(sess, x, r)
                self._assert_parity(y_np, y_got, f"one_slot_promo_phase1(step={step})")

            # Step 6..11: route to different cold experts [8, 9, 10, 11] to verify demand cold reads and further
            # promotions never starve on a 1-slot FileExpertStore.
            for step in range(6):
                x = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
                r = np.full((2, 16), -10.0, dtype=np.float32)
                for row in range(2):
                    r[row, [8, 9, 10, 11]] = [6.0, 5.0, 4.0, 3.0]
                y_np = _numpy_nvfp4_qmoe_reference(model_path, x, r)
                y_got = self._run_bf16(sess, x, r)
                self._assert_parity(y_np, y_got, f"one_slot_promo_phase2(step={step})")

    def test_v4_concurrent_sessions_and_zero_residency_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a = os.path.join(tmpdir, "conc_a")
            dir_b = os.path.join(tmpdir, "conc_b")
            os.makedirs(dir_a, exist_ok=True)
            os.makedirs(dir_b, exist_ok=True)
            model_a = _build_canonical_nvfp4_qmoe_model(dir_a, num_layers=2, num_experts=16, seed=1201)
            model_b = _build_canonical_nvfp4_qmoe_model(dir_b, num_layers=2, num_experts=16, seed=1202)

            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            budget_adaptive = 2 * fixed_table_bytes + 12 * slot_bytes

            # 1. Two eager hybrid sessions with async_max_tokens=8 and T=5 rows (nonpersistent tiles) running concurrently
            sess_a = self._make_session(
                model_a,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.async_max_tokens": "8",
                },
            )
            sess_b = self._make_session(
                model_b,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.async_max_tokens": "8",
                },
            )
            rng_a = np.random.default_rng(1203)
            rng_b = np.random.default_rng(1204)
            inputs_a = [
                (
                    rng_a.standard_normal((5, 128), dtype=np.float32) * 0.5,
                    rng_a.standard_normal((5, 16), dtype=np.float32),
                )
                for _ in range(6)
            ]
            inputs_b = [
                (
                    rng_b.standard_normal((5, 128), dtype=np.float32) * 0.5,
                    rng_b.standard_normal((5, 16), dtype=np.float32),
                )
                for _ in range(6)
            ]
            refs_a = [_numpy_nvfp4_qmoe_reference(model_a, x, r) for x, r in inputs_a]
            refs_b = [_numpy_nvfp4_qmoe_reference(model_b, x, r) for x, r in inputs_b]
            errors = []

            def worker(sess, inputs, refs, label):
                try:
                    for idx, ((x, r), y_ref) in enumerate(zip(inputs, refs, strict=False)):
                        y_got = self._run_bf16(sess, x, r)
                        self._assert_parity(y_ref, y_got, f"{label}_iter_{idx}")
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=worker, args=(sess_a, inputs_a, refs_a, "conc_sess_a"))
            t2 = threading.Thread(target=worker, args=(sess_b, inputs_b, refs_b, "conc_sess_b"))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if errors:
                raise errors[0]
            del sess_a, sess_b

            # 2. Zero-residency cpu_execution="disabled" session must never borrow another session's resident weights
            sess_res_owner = self._make_session(
                model_a,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                },
            )
            sess_zero_res = self._make_session(
                model_b,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.cpu_execution": "disabled",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.node_overrides": (
                        "model/layers.0/mlp/QMoE_node_0:max_resident_slots=0;"
                        "model/layers.1/mlp/QMoE_node_1:max_resident_slots=0"
                    ),
                },
            )
            x_test = rng_a.standard_normal((2, 128), dtype=np.float32) * 0.5
            r_test = rng_a.standard_normal((2, 16), dtype=np.float32)
            ya_np = _numpy_nvfp4_qmoe_reference(model_a, x_test, r_test)
            yb_np = _numpy_nvfp4_qmoe_reference(model_b, x_test, r_test)
            ya_got = self._run_bf16(sess_res_owner, x_test, r_test)
            yb_got = self._run_bf16(sess_zero_res, x_test, r_test)
            self._assert_parity(ya_np, ya_got, "res_owner_a")
            self._assert_parity(yb_np, yb_got, "zero_res_b")
            self.assertGreater(float(np.max(np.abs(ya_got - yb_got))), 0.1)

    def test_v4_decode_capture_prefill_reclaim_and_recapture_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=1,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=4,
                seed=1301,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            budget_1l = fixed_table_bytes + 8 * slot_bytes
            per_layer_decode_pinned = 262912

            cg_model_path = _wrap_qmoe_model_with_cast_io(
                model_path, os.path.join(tmpdir, "model_cg_reclaim.onnx"), 128, 16
            )

            cudart = _load_cudart()
            if cudart is None:
                self.skipTest("CUDA runtime library is required to update captured GPU inputs in place")

            dev_id = self._device_id(self._selected_plugin_device(prefer_plugin=True))

            old_proactive = os.environ.get("ORT_QMOE_PREFILL_PROACTIVE_LEND")
            os.environ["ORT_QMOE_PREFILL_PROACTIVE_LEND"] = "0"
            try:
                sess_graph = self._make_session(
                    cg_model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.device_expert_bytes": str(budget_1l),
                        # Tight pinned budget fits either persistent decode (262,912 B) OR prefill staging (222,720 B),
                        # but NOT both simultaneously (262,912 + 222,720 = 485,632 B), forcing prefill to reclaim
                        # decode pinned buffers and invalidate the captured decode graph.
                        "session.moe.offload.pinned_bytes": str(per_layer_decode_pinned),
                    },
                    cuda_options={
                        "enable_cuda_graph": "1",
                        "min_num_runs_before_cuda_graph_capture": "2",
                    },
                    prefer_plugin=True,
                )

                def device_buffer(nbytes: int) -> ctypes.c_void_p:
                    ptr = ctypes.c_void_p()
                    self.assertEqual(cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), 0)
                    return ptr

                def upload(ptr: ctypes.c_void_p, arr: np.ndarray):
                    arr = np.ascontiguousarray(arr, dtype=np.float32)
                    self.assertEqual(
                        cudart.cudaMemcpy(ptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, 1),
                        0,
                    )

                rng = np.random.default_rng(1302)
                x_dec = _round_to_act_dtype(rng.standard_normal((2, 128), dtype=np.float32) * 0.5)
                r_dec = _round_to_act_dtype(rng.standard_normal((2, 16), dtype=np.float32) * 2.0)
                x_ptr = device_buffer(x_dec.nbytes)
                r_ptr = device_buffer(r_dec.nbytes)
                try:
                    binding = sess_graph.io_binding()
                    binding.bind_input("X_f32", "cuda", dev_id, np.float32, list(x_dec.shape), x_ptr.value)
                    binding.bind_input("R_f32", "cuda", dev_id, np.float32, list(r_dec.shape), r_ptr.value)
                    binding.bind_output("Y_f32", "cuda", dev_id)

                    # Phase 1: warmup, capture, and replay decode graph (T=2)
                    for step in range(5):
                        x_in = _round_to_act_dtype(rng.standard_normal((2, 128), dtype=np.float32) * 0.5)
                        r_in = _round_to_act_dtype(rng.standard_normal((2, 16), dtype=np.float32) * 2.0)
                        upload(x_ptr, x_in)
                        upload(r_ptr, r_in)
                        y_np = _numpy_nvfp4_qmoe_reference(model_path, x_in, r_in)
                        sess_graph.run_with_iobinding(binding)
                        y_got = binding.copy_outputs_to_cpu()[0]
                        self._assert_parity(y_np, y_got, f"pre_reclaim_decode(step={step})")

                    # Phase 2: run prefill (T=16) with annotation -1 (skip capture) which reclaims decode pinned state
                    ro_skip = ort.RunOptions()
                    ro_skip.add_run_config_entry("gpu_graph_id", "-1")
                    x_pref = _round_to_act_dtype(rng.standard_normal((16, 128), dtype=np.float32) * 0.5)
                    r_pref = _round_to_act_dtype(rng.standard_normal((16, 16), dtype=np.float32) * 2.0)
                    xp_ptr = device_buffer(x_pref.nbytes)
                    rp_ptr = device_buffer(r_pref.nbytes)
                    try:
                        upload(xp_ptr, x_pref)
                        upload(rp_ptr, r_pref)
                        pref_binding = sess_graph.io_binding()
                        pref_binding.bind_input("X_f32", "cuda", dev_id, np.float32, list(x_pref.shape), xp_ptr.value)
                        pref_binding.bind_input("R_f32", "cuda", dev_id, np.float32, list(r_pref.shape), rp_ptr.value)
                        pref_binding.bind_output("Y_f32", "cuda", dev_id)
                        sess_graph.run_with_iobinding(pref_binding, ro_skip)
                        y_pref_got = pref_binding.copy_outputs_to_cpu()[0]
                        y_pref_np = _numpy_nvfp4_qmoe_reference(model_path, x_pref, r_pref)
                        self._assert_parity(y_pref_np, y_pref_got, "prefill_reclaim_step")
                    finally:
                        cudart.cudaFree(xp_ptr)
                        cudart.cudaFree(rp_ptr)

                    # Phase 3: decode again (T=2) — must recreate decode state, recapture, and replay with fresh inputs
                    post_binding = sess_graph.io_binding()
                    post_binding.bind_input("X_f32", "cuda", dev_id, np.float32, list(x_dec.shape), x_ptr.value)
                    post_binding.bind_input("R_f32", "cuda", dev_id, np.float32, list(r_dec.shape), r_ptr.value)
                    post_binding.bind_output("Y_f32", "cuda", dev_id)
                    for step in range(5):
                        x_in = _round_to_act_dtype(rng.standard_normal((2, 128), dtype=np.float32) * 0.5)
                        r_in = _round_to_act_dtype(rng.standard_normal((2, 16), dtype=np.float32) * 2.0)
                        upload(x_ptr, x_in)
                        upload(r_ptr, r_in)
                        y_np = _numpy_nvfp4_qmoe_reference(model_path, x_in, r_in)
                        sess_graph.run_with_iobinding(post_binding)
                        y_got = post_binding.copy_outputs_to_cpu()[0]
                        self._assert_parity(y_np, y_got, f"post_reclaim_decode(step={step})")
                finally:
                    cudart.cudaFree(x_ptr)
                    cudart.cudaFree(r_ptr)
            finally:
                if old_proactive is None:
                    os.environ.pop("ORT_QMOE_PREFILL_PROACTIVE_LEND", None)
                else:
                    os.environ["ORT_QMOE_PREFILL_PROACTIVE_LEND"] = old_proactive

    def test_v4_prefill_reclaim_right_after_decode_does_not_race_the_doorbell_worker(self):
        # The doorbell worker keeps polling for a couple of milliseconds after the last decode doorbell. A prefill
        # that reclaims the decode pinned buffer inside that window must not let the worker pick up the buffer's stale
        # request counter as a new doorbell. All inputs are prepared up front so the prefill run starts right after
        # the decode run, which is what puts the reclaim inside the window.
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=1,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=4,
                seed=1301,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            budget_1l = fixed_table_bytes + 8 * slot_bytes
            per_layer_decode_pinned = 262912

            cg_model_path = _wrap_qmoe_model_with_cast_io(
                model_path, os.path.join(tmpdir, "model_cg_reclaim_race.onnx"), 128, 16
            )

            cudart = _load_cudart()
            if cudart is None:
                self.skipTest("CUDA runtime library is required to update captured GPU inputs in place")

            dev_id = self._device_id(self._selected_plugin_device(prefer_plugin=True))

            old_proactive = os.environ.get("ORT_QMOE_PREFILL_PROACTIVE_LEND")
            os.environ["ORT_QMOE_PREFILL_PROACTIVE_LEND"] = "0"
            try:
                sess_graph = self._make_session(
                    cg_model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.device_expert_bytes": str(budget_1l),
                        # Fits persistent decode pinned (262,912 B) or prefill staging (222,720 B), not both, so every
                        # prefill reclaims the decode pinned buffer.
                        "session.moe.offload.pinned_bytes": str(per_layer_decode_pinned),
                    },
                    cuda_options={
                        "enable_cuda_graph": "1",
                        "min_num_runs_before_cuda_graph_capture": "2",
                    },
                    prefer_plugin=True,
                )

                def device_buffer(nbytes: int) -> ctypes.c_void_p:
                    ptr = ctypes.c_void_p()
                    self.assertEqual(cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), 0)
                    return ptr

                def upload(ptr: ctypes.c_void_p, arr: np.ndarray):
                    arr = np.ascontiguousarray(arr, dtype=np.float32)
                    self.assertEqual(
                        cudart.cudaMemcpy(ptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, 1),
                        0,
                    )

                def make_binding(x_ptr: ctypes.c_void_p, r_ptr: ctypes.c_void_p, x: np.ndarray, r: np.ndarray):
                    binding = sess_graph.io_binding()
                    binding.bind_input("X_f32", "cuda", dev_id, np.float32, list(x.shape), x_ptr.value)
                    binding.bind_input("R_f32", "cuda", dev_id, np.float32, list(r.shape), r_ptr.value)
                    binding.bind_output("Y_f32", "cuda", dev_id)
                    return binding

                rng = np.random.default_rng(1302)
                x_dec = _round_to_act_dtype(rng.standard_normal((2, 128), dtype=np.float32) * 0.5)
                r_dec = _round_to_act_dtype(rng.standard_normal((2, 16), dtype=np.float32) * 2.0)
                x_pref = _round_to_act_dtype(rng.standard_normal((16, 128), dtype=np.float32) * 0.5)
                r_pref = _round_to_act_dtype(rng.standard_normal((16, 16), dtype=np.float32) * 2.0)
                y_dec_np = _numpy_nvfp4_qmoe_reference(model_path, x_dec, r_dec)
                y_pref_np = _numpy_nvfp4_qmoe_reference(model_path, x_pref, r_pref)

                x_ptr = device_buffer(x_dec.nbytes)
                r_ptr = device_buffer(r_dec.nbytes)
                xp_ptr = device_buffer(x_pref.nbytes)
                rp_ptr = device_buffer(r_pref.nbytes)
                try:
                    upload(x_ptr, x_dec)
                    upload(r_ptr, r_dec)
                    upload(xp_ptr, x_pref)
                    upload(rp_ptr, r_pref)

                    ro_skip = ort.RunOptions()
                    ro_skip.add_run_config_entry("gpu_graph_id", "-1")
                    pref_binding = make_binding(xp_ptr, rp_ptr, x_pref, r_pref)

                    for cycle in range(4):
                        dec_binding = make_binding(x_ptr, r_ptr, x_dec, r_dec)
                        for _ in range(5):
                            sess_graph.run_with_iobinding(dec_binding)
                        # No Python work between the last decode and the reclaiming prefill.
                        sess_graph.run_with_iobinding(pref_binding, ro_skip)
                        y_pref_got = pref_binding.copy_outputs_to_cpu()[0]
                        self._assert_parity(y_pref_np, y_pref_got, f"prefill_reclaim(cycle={cycle})")

                    # Decode must recreate its state, recapture and replay after the last reclaim.
                    post_binding = make_binding(x_ptr, r_ptr, x_dec, r_dec)
                    for step in range(5):
                        sess_graph.run_with_iobinding(post_binding)
                        y_got = post_binding.copy_outputs_to_cpu()[0]
                        self._assert_parity(y_dec_np, y_got, f"post_reclaim_decode(step={step})")
                finally:
                    for ptr in (x_ptr, r_ptr, xp_ptr, rp_ptr):
                        cudart.cudaFree(ptr)
            finally:
                if old_proactive is None:
                    os.environ.pop("ORT_QMOE_PREFILL_PROACTIVE_LEND", None)
                else:
                    os.environ["ORT_QMOE_PREFILL_PROACTIVE_LEND"] = old_proactive

    def test_v4_device_working_bytes_enforced_across_planned_prefill_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=1,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=4,
                seed=1401,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            budget_partial = fixed_table_bytes + 6 * slot_bytes
            budget_full = fixed_table_bytes + 16 * slot_bytes
            # 1 expert dequantized weight size for H=128, I=64 (BF16) is 3 * 128 * 64 * 2 = 49,152 B.
            # Setting device_working_bytes = 16,384 B cannot fit even a 1-expert wave and must fail on prefill
            # in both partial-residency and full-residency planned modes.
            x_pref = np.ones((16, 128), dtype=np.float32) * 0.25
            r_pref = np.ones((16, 16), dtype=np.float32)
            for _label, dev_budget in (("partial", budget_partial), ("full", budget_full)):
                sess = self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(dev_budget),
                        "session.moe.offload.device_working_bytes": "16384",
                    },
                )
                with self.assertRaisesRegex(Exception, "device_working_bytes"):
                    self._run_bf16(sess, x_pref, r_pref)

    def test_v4_one_slot_cold_routes_each_dominant_and_a_lost_route_is_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=2,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=4,
                seed=1101,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            host_slot_stride = 14336
            budget_small = 2 * fixed_table_bytes + 12 * slot_bytes
            rng = np.random.default_rng(1103)
            steps_per_cold_set = 6
            total_rows = 0
            # The 12-slot budget plans 4 resident experts (plus 2 spare slots) per layer: the head of each layer's
            # ranking in the profile written by _build_canonical_nvfp4_qmoe_model, i.e. {0, 3, 6, 9} in layer 0 and
            # {1, 4, 7, 10} in layer 1. Both cold sets below avoid the resident experts of both layers.
            cold_sets = ([12, 13, 14, 15], [2, 5, 8, 11])
            # Every row is dominated by one cold expert (softmax weight ~0.996) and the dominant expert rotates through
            # the whole cold set. Losing any single dominant contribution then moves the output far outside the parity
            # tolerance, which dropped_route proves below for every layer and row. That only shows the dominant route
            # was merged, so the placement is pinned and observed as well: promotions are disabled, which keeps the
            # cold experts cold for the whole test, and ORT_QMOE_SHARED_CPU_TIMING makes the CPU path count, per
            # layer, how many of the routes it executed were non-resident when they ran.
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "ORT_QMOE_SHARED_RESIDENT_LOG": "1",
                        "ORT_QMOE_SHARED_CPU_TIMING": "1",
                        "ORT_QMOE_SHARED_RESIDENT_PROMOTIONS": "0",
                    },
                ),
                _capture_native_stderr() as captured,
            ):
                sess = self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.device_expert_bytes": str(budget_small),
                        "session.moe.offload.host_expert_bytes": str(2 * host_slot_stride),
                        "session.moe.offload.allow_file_backing": "1",
                    },
                )
                for phase, cold_set in enumerate(cold_sets):
                    for step in range(steps_per_cold_set):
                        rows = 1 + step % 2
                        total_rows += rows
                        x = rng.standard_normal((rows, 128), dtype=np.float32) * 0.5
                        r = np.full((rows, 16), -10.0, dtype=np.float32)
                        for row in range(rows):
                            dominant = cold_set[(step + row) % len(cold_set)]
                            r[row, [e for e in cold_set if e != dominant]] = [2.0, 1.0, 0.0]
                            r[row, dominant] = 8.0
                        y_ref = _numpy_nvfp4_qmoe_reference(model_path, x, r)
                        y_got = self._run_bf16(sess, x, r)
                        label = f"one_slot_dominant_cold(phase={phase},step={step})"
                        self._assert_parity(y_ref, y_got, label)
                        for layer in range(2):
                            for row in range(rows):
                                y_lost = _numpy_nvfp4_qmoe_reference(model_path, x, r, dropped_route=(layer, row, 0))
                                with self.assertRaisesRegex(AssertionError, "mismatch"):
                                    self._assert_parity(
                                        y_ref, y_lost, f"{label}, lost route (layer={layer}, row={row})"
                                    )
                del sess
                gc.collect()

            # Both layers must report, and in each of them every executed route, including each amplified dominant
            # one, must have run on a non-resident expert. The placement did not change either: the planned residents
            # were still the only ones and nothing was promoted.
            cpu_rows = _cpu_path_route_rows(captured[0])
            self.assertEqual(sorted(cpu_rows), [0, 1], "both layers must run their cold routes on the CPU path")
            for layer, (_calls, routes, cold_routes) in cpu_rows.items():
                self.assertEqual(routes, total_rows * 4, f"layer {layer} observes every executed row once")
                self.assertEqual(
                    cold_routes,
                    routes,
                    f"layer {layer}: every route must have run on a non-resident expert, but only {cold_routes} of "
                    f"{routes} did, so a cold set overlaps the planned resident experts",
                )
            self.assertEqual(_resident_slots_per_layer(captured[0]), {0: 4, 1: 4})
            self.assertEqual(_resident_promotions_per_layer(captured[0]), {0: 0, 1: 0})

    def test_v4_cuda_graph_capture_and_immediate_replay_observe_each_execution_once(self):
        cudart = _load_cudart()
        if cudart is None:
            self.skipTest("CUDA runtime library is required to update captured GPU inputs in place")
        rows, top_k, steps, min_runs_before_capture = 2, 4, 4, 2
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=2,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=top_k,
                seed=1501,
            )
            cg_model_path = _wrap_qmoe_model_with_cast_io(
                model_path, os.path.join(tmpdir, "model_cg_routes.onnx"), 128, 16
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            offload_config = {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "adaptive",
                # 6 resident slots per layer, so every route is classified against a partial resident set.
                "session.moe.offload.device_expert_bytes": str(2 * fixed_table_bytes + 16 * slot_bytes),
                "session.moe.offload.popularity_file": "expert-profile.bin",
            }

            def observed_routes_per_layer(prefer_plugin: bool, enable_cuda_graph: bool) -> dict[int, int]:
                cuda_options = (
                    {
                        "enable_cuda_graph": "1",
                        "min_num_runs_before_cuda_graph_capture": str(min_runs_before_capture),
                    }
                    if enable_cuda_graph
                    else {}
                )
                dev_id = self._device_id(self._selected_plugin_device(prefer_plugin))
                rng = np.random.default_rng(1502)
                x_ptr = _cuda_malloc(cudart, rows * 128 * 4)
                r_ptr = _cuda_malloc(cudart, rows * 16 * 4)
                try:
                    # The backend prints its route totals when it is destroyed, so the session has to go away
                    # inside the capture window.
                    with (
                        mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1"}),
                        _capture_native_stderr() as captured,
                    ):
                        sess = self._make_session(
                            cg_model_path, offload_config, cuda_options=cuda_options, prefer_plugin=prefer_plugin
                        )
                        binding = sess.io_binding()
                        binding.bind_input("X_f32", "cuda", dev_id, np.float32, [rows, 128], x_ptr.value)
                        binding.bind_input("R_f32", "cuda", dev_id, np.float32, [rows, 16], r_ptr.value)
                        binding.bind_output("Y_f32", "cuda", dev_id)
                        for _ in range(steps):
                            _cuda_upload(cudart, x_ptr, rng.standard_normal((rows, 128), dtype=np.float32) * 0.5)
                            _cuda_upload(cudart, r_ptr, rng.standard_normal((rows, 16), dtype=np.float32) * 2.0)
                            sess.run_with_iobinding(binding)
                            binding.copy_outputs_to_cpu()
                        del binding
                        del sess
                        gc.collect()
                finally:
                    cudart.cudaFree(x_ptr)
                    cudart.cudaFree(r_ptr)
                return _resident_routes_per_layer(captured[0])

            routes_per_execution = rows * top_k
            plugin_flavors = (False, True) if self._get_plugin_cuda_device() is not None else (False,)
            for prefer_plugin in plugin_flavors:
                with self.subTest(prefer_plugin=prefer_plugin):
                    self.assertEqual(
                        observed_routes_per_layer(prefer_plugin, enable_cuda_graph=False),
                        dict.fromkeys(range(2), steps * routes_per_execution),
                        "without CUDA Graph every Run observes its routes once",
                    )
                    # ORT keeps re-running the first Run() until the graph is captured, so that Run executes
                    # min_num_runs_before_cuda_graph_capture regular runs plus the capture run (which is replayed
                    # right away); each later Run replays once. Counting the capture run and its immediate replay as
                    # two observations of one execution would add another rows * top_k routes per layer.
                    self.assertEqual(
                        observed_routes_per_layer(prefer_plugin, enable_cuda_graph=True),
                        dict.fromkeys(range(2), (steps + min_runs_before_capture) * routes_per_execution),
                        "with CUDA Graph every executed run, capture included, is observed exactly once",
                    )

    def test_v4_standard_weight_layout_with_hidden_size_other_than_128(self):
        hidden_size, inter_size, num_experts, num_layers = 256, 64, 16, 2
        slot_bytes = (
            (2 * inter_size * hidden_size // 2)
            + (hidden_size * inter_size // 2)
            + (2 * inter_size * hidden_size // 16)
            + (hidden_size * inter_size // 16)
            + 8
        )
        fixed_table_bytes = 2 * num_experts * 4
        planned_slots = {}
        # Both layouts hold identical bytes, so one numpy reference serves them. H=256/I=64 is the shape whose
        # dimensions are mixed up when H and I are read from the legacy weight layout only (H=128, I=256).
        for weight_layout in ("standard", "legacy"):
            with self.subTest(weight_layout=weight_layout), tempfile.TemporaryDirectory() as tmpdir:
                model_path = _build_canonical_nvfp4_qmoe_model(
                    tmpdir,
                    num_layers=num_layers,
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    inter_size=inter_size,
                    top_k=4,
                    seed=1201,
                    weight_layout=weight_layout,
                )
                with (
                    mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1"}),
                    _capture_native_stderr() as captured,
                ):
                    sess = self._make_session(
                        model_path,
                        {
                            "session.moe.offload.mode": "required",
                            "session.moe.offload.placement": "adaptive",
                            # Partial residency for decode and for the prefill waves.
                            "session.moe.offload.device_expert_bytes": str(
                                num_layers * fixed_table_bytes + 10 * slot_bytes
                            ),
                            "session.moe.offload.popularity_file": "expert-profile.bin",
                        },
                    )
                    rng = np.random.default_rng(1202)
                    for t_rows in (1, 4, 16):
                        x = rng.standard_normal((t_rows, hidden_size), dtype=np.float32) * 0.5
                        r = rng.standard_normal((t_rows, num_experts), dtype=np.float32)
                        y_ref = _numpy_nvfp4_qmoe_reference(model_path, x, r)
                        y_got = self._run_bf16(sess, x, r)
                        self._assert_parity(y_ref, y_got, f"{weight_layout}_layout_h{hidden_size}(T={t_rows})")
                    del sess
                    gc.collect()
                planned_slots[weight_layout] = _resident_slots_per_layer(captured[0])

        # The admission planner charges every expert slot from H and I. Both layouts have the same H, I and bytes, so
        # they must buy the same slots; reading the standard layout as H=128/I=256 doubles the charge per slot.
        if len(planned_slots) == 2:
            self.assertEqual(
                planned_slots["standard"],
                planned_slots["legacy"],
                "the weight layout must not change the number of resident slots the device budget buys",
            )

    def test_v4_per_run_controls_are_validated_and_scoped_to_their_run(self):
        rows, top_k = 2, 4
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=2,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=top_k,
                seed=1601,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            offload_config = {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "adaptive",
                "session.moe.offload.device_expert_bytes": str(2 * fixed_table_bytes + 16 * slot_bytes),
                "session.moe.offload.popularity_file": "expert-profile.bin",
            }
            rng = np.random.default_rng(1602)

            def step_inputs():
                return (
                    rng.standard_normal((rows, 128), dtype=np.float32) * 0.5,
                    rng.standard_normal((rows, 16), dtype=np.float32),
                )

            warmup = ort.RunOptions()
            warmup.add_run_config_entry("ep.cuda.qmoe_in_warmup", "1")
            malformed = ort.RunOptions()
            malformed.add_run_config_entry("ep.cuda.qmoe_prefill_total_tokens", "abc")

            # ORT_QMOE_IN_WARMUP used to switch the whole process into warmup; it is not read any more, and the
            # controls travel with the Run that carries them instead of through process-wide state.
            with (
                mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1", "ORT_QMOE_IN_WARMUP": "1"}),
                _capture_native_stderr() as captured,
            ):
                sess = self._make_session(model_path, offload_config)
                sess_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})

                for _ in range(3):
                    x, r = step_inputs()
                    self._assert_parity(self._run_bf16(sess_ref, x, r), self._run_bf16(sess, x, r), "plain_run")

                for _ in range(2):
                    x, r = step_inputs()
                    self._assert_parity(
                        self._run_bf16(sess_ref, x, r), _run_qmoe_bf16(sess, x, r, warmup), "warmup_run"
                    )

                with self.assertRaisesRegex(Exception, "(?i)INVALID_ARGUMENT|qmoe_prefill_total_tokens"):
                    x, r = step_inputs()
                    _run_qmoe_bf16(sess, x, r, malformed)

                # Neither the warmup flag nor the rejected value sticks to the session.
                x, r = step_inputs()
                self._assert_parity(self._run_bf16(sess_ref, x, r), self._run_bf16(sess, x, r), "run_after_controls")
                del sess
                del sess_ref
                gc.collect()

            self.assertEqual(
                _resident_routes_per_layer(captured[0]),
                dict.fromkeys(range(2), 4 * rows * top_k),
                "only the 3 plain runs and the final run may be observed",
            )

    def test_v4_synchronous_streaming_path_ignores_warmup_runs(self):
        rows, top_k = 2, 4
        cold_set = [12, 13, 14, 15]
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=2,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=top_k,
                seed=1701,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            # The 12-slot budget keeps 4 resident experts per layer ({0, 3, 6, 9} and {1, 4, 7, 10}, the head of each
            # layer's profile), none of them in cold_set. With the CPU path disabled, the synchronous GPU streaming
            # path serves every cold expert and the async CPU path is never involved.
            offload_config = {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "adaptive",
                "session.moe.offload.cpu_execution": "disabled",
                "session.moe.offload.device_expert_bytes": str(2 * fixed_table_bytes + 12 * slot_bytes),
                "session.moe.offload.popularity_file": "expert-profile.bin",
            }
            warmup = ort.RunOptions()
            warmup.add_run_config_entry("ep.cuda.qmoe_in_warmup", "1")

            def run_session(warmup_runs: int, plain_runs: int) -> str:
                rng = np.random.default_rng(1702)
                with (
                    mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1"}),
                    _capture_native_stderr() as captured,
                ):
                    sess = self._make_session(model_path, offload_config)
                    for step, run_options in enumerate([warmup] * warmup_runs + [None] * plain_runs):
                        x = rng.standard_normal((rows, 128), dtype=np.float32) * 0.5
                        r = np.full((rows, 16), -10.0, dtype=np.float32)
                        for row in range(rows):
                            r[row, cold_set] = np.roll([8.0, 2.0, 1.0, 0.0], step + row)
                        # Ignoring a warmup run must not change what it computes.
                        self._assert_parity(
                            _numpy_nvfp4_qmoe_reference(model_path, x, r),
                            _run_qmoe_bf16(sess, x, r, run_options),
                            f"sync_stream(warmup_runs={warmup_runs},plain_runs={plain_runs},step={step})",
                        )
                    del sess
                    gc.collect()
                return captured[0]

            def assert_streamed(log_text: str, label: str):
                tiles = _stream_tiles_per_layer(log_text)
                self.assertEqual(sorted(tiles), [0, 1], label)
                for layer, count in tiles.items():
                    self.assertGreater(count, 0, f"{label}: layer {layer} must run the synchronous streaming path")

            # Control: plain runs are observed and, since every route is cold, they drive promotions. Without this
            # the checks below could pass just because nothing ever adapts in this configuration.
            control = run_session(warmup_runs=0, plain_runs=6)
            assert_streamed(control, "control")
            self.assertEqual(_resident_slots_per_layer(control), {0: 4, 1: 4})
            self.assertEqual(_resident_routes_per_layer(control), dict.fromkeys(range(2), 6 * rows * top_k))
            for layer, promotions in _resident_promotions_per_layer(control).items():
                self.assertGreater(promotions, 0, f"layer {layer}: plain cold routes must drive promotions")

            # Warmup runs execute the same streaming path but feed no route counters, scores or promotions.
            warmup_only = run_session(warmup_runs=6, plain_runs=0)
            assert_streamed(warmup_only, "warmup_only")
            self.assertEqual(_resident_log_rows(warmup_only), {0: (4, 0, 0), 1: (4, 0, 0)})
            self.assertEqual(_resident_promotions_per_layer(warmup_only), {0: 0, 1: 0})

            # The placement a real run starts from is the planned one: all of its routes still miss the resident set.
            after_warmup = run_session(warmup_runs=6, plain_runs=1)
            assert_streamed(after_warmup, "after_warmup")
            self.assertEqual(_resident_log_rows(after_warmup), dict.fromkeys(range(2), (4, 0, rows * top_k)))
            self.assertEqual(_resident_promotions_per_layer(after_warmup), {0: 0, 1: 0})

    def test_v4_nvfp4_cpu_disabled_streaming_wave_shrinks_to_fit_the_working_budget(self):
        rows, top_k, num_experts = 2, 2, 8
        slot_bytes = 13832
        fixed_table_bytes = 2 * num_experts * 4
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=1,
                num_experts=num_experts,
                hidden_size=128,
                inter_size=64,
                top_k=top_k,
                seed=33622,
            )
            # Four experts are resident and the CPU path is disabled, so cold routes stream. The double-buffered landing
            # slabs of the default 8-expert wave alone need 2 * 8 * slot_bytes = 216 KiB, above the 200 KiB working budget; a 4-expert wave fits.
            offload_config = {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "static",
                "session.moe.offload.cpu_execution": "disabled",
                "session.moe.offload.device_expert_bytes": str(fixed_table_bytes + 4 * slot_bytes),
                "session.moe.offload.device_working_bytes": str(200 << 10),
                "session.moe.offload.popularity_file": "expert-profile.bin",
            }
            self.assertGreater(2 * num_experts * slot_bytes, 200 << 10)
            rng = np.random.default_rng(33623)
            with (
                mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1"}),
                _capture_native_stderr() as captured,
            ):
                sess = self._make_session(model_path, offload_config)
                try:
                    for step in range(4):
                        x = rng.standard_normal((rows, 128), dtype=np.float32) * 0.5
                        r = rng.standard_normal((rows, num_experts), dtype=np.float32)
                        self._assert_parity(
                            _numpy_nvfp4_qmoe_reference(model_path, x, r),
                            _run_qmoe_bf16(sess, x, r),
                            f"nvfp4_cpu_disabled_wave_fit(step={step})",
                        )
                finally:
                    del sess
                    gc.collect()
            log_text = captured[0]
            self.assertRegex(log_text, r"stream budget layer=0 tile=\d+->\d+ wave=8->[1-7]\b")
            self.assertGreater(_stream_tiles_per_layer(log_text).get(0, 0), 0, log_text)

    @unittest.skipUnless(sys.platform.startswith("linux"), "QMoE background threads are enumerated through /proc")
    def test_v4_background_threads_are_joined_when_the_cuda_ep_library_is_unregistered(self):
        if self._get_plugin_cuda_device() is None:
            self.skipTest("Requires the CUDA plugin EP, whose library unload has to stop the process-wide QMoE threads")
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=2,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=4,
                seed=1701,
            )
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--background-thread-child", model_path],
                env={**os.environ, "ORT_QMOE_SHARED_CPU_THREADS": "4"},
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        self.assertEqual(
            result.returncode,
            0,
            f"child process failed with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr[-4000:]}",
        )

    @staticmethod
    def _build_int_qmoe_model(
        model_dir: str,
        layer_specs: list[tuple[int, int, int | None]],
        num_experts: int = 8,
        hidden_size: int = 256,
        inter_size: int = 128,
        top_k: int = 2,
        seed: int = 2026,
        with_bias: bool = False,
        use_fp16: bool = False,
        scale_range: tuple[float, float] = (0.04, 0.12),
        swiglu_fusion: int | None = 1,
        swiglu_limit: float = float("inf"),
        normalize_routing_weights: int = 1,
    ) -> str:
        """Build an INT4/INT3/INT2 model with layer specs (weight_bits, block_size, prepacked)."""
        rng = np.random.default_rng(seed)
        ext_filename = "int_weights.data"
        ext_path = os.path.join(model_dir, ext_filename)
        act_onnx_dtype = TensorProto.FLOAT16 if use_fp16 else TensorProto.BFLOAT16

        nodes = []
        initializers = []
        offset = 0

        def add_tensor(wf, name, arr, onnx_dtype):
            nonlocal offset
            raw = arr.tobytes()
            pad = (4096 - (offset % 4096)) % 4096
            if pad:
                wf.write(b"\x00" * pad)
                offset += pad
            wf.write(raw)
            t = helper.make_tensor(
                name=name,
                data_type=onnx_dtype,
                dims=list(arr.shape),
                vals=raw,
                raw=True,
            )
            onnx.external_data_helper.set_external_data(t, location=ext_filename, offset=offset, length=len(raw))
            t.data_location = TensorProto.EXTERNAL
            t.ClearField("raw_data")
            offset += len(raw)
            initializers.append(t)

        with open(ext_path, "wb") as wf:
            cur_in = "X"
            for l_idx, (bits, blk_size, prepacked) in enumerate(layer_specs):
                if bits == 4 and prepacked in (1, -1, None):
                    fc1_w = rng.integers(
                        0,
                        256,
                        size=(num_experts, hidden_size, inter_size),
                        dtype=np.uint8,
                    )
                    fc2_w = rng.integers(
                        0,
                        256,
                        size=(num_experts, inter_size, hidden_size // 2),
                        dtype=np.uint8,
                    )
                elif bits == 2 and prepacked == 1:
                    fc1_w = rng.integers(0, 256, size=(num_experts, hidden_size, inter_size // 2), dtype=np.uint8)
                    fc2_w = rng.integers(0, 256, size=(num_experts, inter_size, hidden_size // 4), dtype=np.uint8)
                elif bits == 3 and prepacked == 0:
                    fc1_w = rng.integers(
                        0, 256, size=(num_experts, 2 * inter_size, (hidden_size * 3) // 8), dtype=np.uint8
                    )
                    fc2_w = rng.integers(0, 256, size=(num_experts, hidden_size, (inter_size * 3) // 8), dtype=np.uint8)
                else:
                    pack_factor = 8 // bits
                    fc1_w = rng.integers(
                        0, 256, size=(num_experts, 2 * inter_size, hidden_size // pack_factor), dtype=np.uint8
                    )
                    fc2_w = rng.integers(
                        0, 256, size=(num_experts, hidden_size, inter_size // pack_factor), dtype=np.uint8
                    )

                s1_f32 = rng.uniform(
                    scale_range[0],
                    scale_range[1],
                    size=(num_experts, 2 * inter_size, hidden_size // blk_size),
                ).astype(np.float32)
                s2_f32 = rng.uniform(
                    scale_range[0],
                    scale_range[1],
                    size=(num_experts, hidden_size, inter_size // blk_size),
                ).astype(np.float32)
                if use_fp16:
                    fc1_s = s1_f32.astype(np.float16).view(np.uint16)
                    fc2_s = s2_f32.astype(np.float16).view(np.uint16)
                else:
                    fc1_s = _float32_to_bfloat16_u16(s1_f32)
                    fc2_s = _float32_to_bfloat16_u16(s2_f32)

                w1_name = f"model.layers.{l_idx}.mlp.fc1_experts_weights"
                s1_name = f"model.layers.{l_idx}.mlp.fc1_scales"
                b1_name = ""
                w2_name = f"model.layers.{l_idx}.mlp.fc2_experts_weights"
                s2_name = f"model.layers.{l_idx}.mlp.fc2_scales"
                b2_name = ""

                add_tensor(wf, w1_name, fc1_w, TensorProto.UINT8)
                add_tensor(wf, s1_name, fc1_s, act_onnx_dtype)
                if with_bias:
                    b1_name = f"model.layers.{l_idx}.mlp.fc1_bias"
                    b2_name = f"model.layers.{l_idx}.mlp.fc2_bias"
                    b1_f32 = rng.uniform(-0.05, 0.05, size=(num_experts, 2 * inter_size)).astype(np.float32)
                    b2_f32 = rng.uniform(-0.05, 0.05, size=(num_experts, hidden_size)).astype(np.float32)
                    if use_fp16:
                        fc1_b = b1_f32.astype(np.float16).view(np.uint16)
                        fc2_b = b2_f32.astype(np.float16).view(np.uint16)
                    else:
                        fc1_b = _float32_to_bfloat16_u16(b1_f32)
                        fc2_b = _float32_to_bfloat16_u16(b2_f32)
                    add_tensor(wf, b1_name, fc1_b, act_onnx_dtype)
                    add_tensor(wf, b2_name, fc2_b, act_onnx_dtype)
                add_tensor(wf, w2_name, fc2_w, TensorProto.UINT8)
                add_tensor(wf, s2_name, fc2_s, act_onnx_dtype)

                out_name = "Y" if l_idx == len(layer_specs) - 1 else f"H_{l_idx}"
                node_kwargs = {
                    "activation_type": "swiglu",
                    "normalize_routing_weights": normalize_routing_weights,
                    "k": min(top_k, max(1, num_experts)),
                    "expert_weight_bits": bits,
                    "block_size": blk_size,
                    "swiglu_limit": swiglu_limit,
                    "quant_type": "int",
                }
                if swiglu_fusion is not None:
                    node_kwargs["swiglu_fusion"] = swiglu_fusion
                if prepacked is not None:
                    node_kwargs["weights_prepacked"] = prepacked
                node = helper.make_node(
                    "QMoE",
                    inputs=[cur_in, "router_probs", w1_name, s1_name, b1_name, w2_name, s2_name, b2_name],
                    outputs=[out_name],
                    name=f"model/layers.{l_idx}/mlp/QMoE_node_{l_idx}",
                    domain="com.microsoft",
                    **node_kwargs,
                )
                nodes.append(node)
                cur_in = out_name

        if num_experts > 0:
            prof_path = os.path.join(model_dir, "expert-profile.bin")
            pairs = [
                (l_idx, (e * 3 + l_idx) % num_experts) for e in range(num_experts) for l_idx in range(len(layer_specs))
            ]
            with open(prof_path, "wb") as pf:
                pf.write(
                    struct.pack(
                        "<6I",
                        0x50525453,
                        1,
                        len(layer_specs),
                        num_experts,
                        len(layer_specs) * num_experts,
                        len(pairs),
                    )
                )
                pf.writelines(struct.pack("<2H", l_id, e_id) for l_id, e_id in pairs)

        graph = helper.make_graph(
            nodes,
            "qmoe_int_offload_v4_test",
            [
                helper.make_tensor_value_info("X", act_onnx_dtype, ["T", hidden_size]),
                helper.make_tensor_value_info("router_probs", act_onnx_dtype, ["T", max(1, num_experts)]),
            ],
            [helper.make_tensor_value_info("Y", act_onnx_dtype, ["T", hidden_size])],
            initializer=initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)],
        )
        model_path = os.path.join(model_dir, "model.onnx")
        onnx.save_model(model, model_path)
        return model_path

    def _assert_int_offload_parity(
        self,
        y_ref: np.ndarray,
        y_cand: np.ndarray,
        label: str,
        min_ref_peak: float = 0.25,
        max_rel_l2: float = 2.5e-2,
        max_rel_inf: float = 5e-2,
    ) -> None:
        self.assertEqual(y_ref.shape, y_cand.shape, f"{label}: shape mismatch")
        self.assertTrue(np.all(np.isfinite(y_ref)), f"{label}: non-finite reference")
        self.assertTrue(np.all(np.isfinite(y_cand)), f"{label}: non-finite candidate")
        ref_peak = float(np.max(np.abs(y_ref)))
        self.assertGreater(ref_peak, min_ref_peak, f"{label}: reference peak {ref_peak} too small")
        token_norms = np.linalg.norm(y_ref.astype(np.float64), axis=-1)
        self.assertGreater(
            float(np.min(token_norms)),
            0.25 * min_ref_peak * np.sqrt(y_ref.shape[-1]),
            f"{label}: per-token reference norm too small",
        )
        token_rel_l2 = np.linalg.norm((y_ref - y_cand).astype(np.float64), axis=-1) / token_norms
        self.assertLess(
            float(np.max(token_rel_l2)),
            max_rel_l2,
            f"{label}: max per-token relative L2 mismatch {float(np.max(token_rel_l2)):.5f}",
        )
        max_abs = float(np.max(np.abs(y_ref - y_cand)))
        self.assertLess(
            max_abs,
            max_rel_inf * ref_peak,
            f"{label}: max abs mismatch {max_abs:.5f} exceeds {max_rel_inf} * ref_peak ({ref_peak:.5f})",
        )
        # Verify an all-zero output would decisively fail both the relative L2 and max-abs checks
        zero_rel_l2 = float(np.max(np.linalg.norm(y_ref.astype(np.float64), axis=-1) / token_norms))
        self.assertGreater(zero_rel_l2, max_rel_l2 * 10.0)
        self.assertGreater(ref_peak, max_rel_inf * ref_peak * 10.0)

    def test_v4_int4_cutlass_resident_and_bounded_streaming_multichunk_parity(self):
        rows = 2304  # k=2 produces 4,608 routes in one tile, including a 512-route second chunk.
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(
                os.environ,
                {
                    "ORT_QMOE_SHARED_RESIDENT_LOG": "1",
                    "ORT_QMOE_INT4_PREFILL_GEMM": "1",
                    "ORT_QMOE_MAX_SYNC_TILE_TOKENS": "4096",
                    "ORT_QMOE_SHARED_STREAM_ROWS": "4096",
                    "ORT_QMOE_PREFILL_CHUNK_TOKENS": "4096",
                },
            ),
        ):
            for use_fp16 in (True, False):
                if not use_fp16:
                    self._skip_unless_int4_bf16_runner()
                model_dir = os.path.join(tmpdir, f"fp16_{use_fp16}")
                os.makedirs(model_dir)
                model = self._build_int_qmoe_model(model_dir, [(4, 32, 1)], use_fp16=use_fp16)
                ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
                run = self._run_fp16 if use_fp16 else self._run_bf16
                rng = np.random.default_rng(33622)
                x = rng.standard_normal((rows, 256), dtype=np.float32) * 0.5
                for streaming in (False, True):
                    with self.subTest(use_fp16=use_fp16, streaming=streaming):
                        router = np.full((rows, 8), -10.0, dtype=np.float32)
                        if streaming:
                            router[: rows // 2, 1:3] = 10.0
                            router[rows // 2 :, 4:6] = 10.0
                        else:
                            # The generated popularity profile places experts 0, 3, and 6 on GPU.
                            router[:, [0, 3]] = 10.0
                        expected = run(ref, x, router)
                        with _capture_native_stderr() as captured:
                            sess = self._make_session(
                                model,
                                {
                                    "session.moe.offload.mode": "required",
                                    "session.moe.offload.placement": "static",
                                    "session.moe.offload.popularity_file": "expert-profile.bin",
                                    "session.moe.offload.cpu_execution": "disabled" if streaming else "enabled",
                                    "session.moe.offload.device_expert_bytes": str(64 + 3 * 55304),
                                    # Two cache slots must serve a four-expert streaming wave.
                                    "session.moe.offload.host_expert_bytes": str(2 * 55808),
                                    "session.moe.offload.prefill_wave_experts": "4",
                                },
                            )
                            actual = run(sess, x, router)
                            del sess
                            gc.collect()
                        self._assert_int_offload_parity(expected, actual, f"cutlass({use_fp16=},{streaming=})")
                        dispatched = re.findall(r"INT4 CUTLASS layer=0 routes=(\d+) capacity=(\d+)", captured[0])
                        self.assertEqual(dispatched, [("4096", "4096"), ("512", "4096")], captured[0])
                        hits = _resident_log_rows(captured[0])[0][1]
                        self.assertEqual(hits, 0 if streaming else rows * 2, captured[0])
                del ref
                gc.collect()

    def test_v4_int4_cutlass_scratch_budget_and_wave_limit(self):
        self._skip_unless_int4_bf16_runner()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(
                os.environ,
                {
                    "ORT_QMOE_SHARED_RESIDENT_LOG": "1",
                    "ORT_QMOE_INT4_PREFILL_GEMM": "1",
                    "ORT_QMOE_MAX_SYNC_TILE_TOKENS": "512",
                },
            ),
        ):
            model = self._build_int_qmoe_model(tmpdir, [(4, 32, 1)])
            ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            rng = np.random.default_rng(33623)
            for rows, working_bytes in ((16, 2 << 20), (512, 3 << 20), (512, 9 << 18)):
                with self.subTest(rows=rows, working_bytes=working_bytes):
                    x = rng.standard_normal((rows, 256), dtype=np.float32) * 0.5
                    router = np.full((rows, 8), -10.0, dtype=np.float32)
                    router[:, 4:6] = 10.0
                    expected = self._run_bf16(ref, x, router)
                    with _capture_native_stderr() as captured:
                        sess = self._make_session(
                            model,
                            {
                                "session.moe.offload.mode": "required",
                                "session.moe.offload.placement": "static",
                                "session.moe.offload.popularity_file": "expert-profile.bin",
                                "session.moe.offload.cpu_execution": "disabled",
                                "session.moe.offload.device_expert_bytes": str(64 + 3 * 55304),
                                "session.moe.offload.device_working_bytes": str(working_bytes),
                                "session.moe.offload.host_expert_bytes": str(2 * 55808),
                                "session.moe.offload.prefill_wave_experts": "1" if rows == 16 else "2",
                            },
                        )
                        actual = self._run_bf16(sess, x, router)
                        del sess
                        gc.collect()
                    self._assert_int_offload_parity(expected, actual, f"scratch({rows=},{working_bytes=})")
                    dispatched = re.findall(r"INT4 CUTLASS layer=0 routes=(\d+) capacity=(\d+)", captured[0])
                    if rows == 16:
                        self.assertFalse(dispatched, captured[0])
                        self.assertRegex(
                            captured[0], r"stream layer=0 tiles=1 experts=2 routes=32 cpu_routes=0 waves=2\b"
                        )
                    elif working_bytes == 3 << 20:
                        self.assertTrue(dispatched, captured[0])
                        self.assertTrue(all(int(capacity) < 1024 for _, capacity in dispatched), captured[0])
                        self.assertEqual(sum(int(routes) for routes, _ in dispatched), 1024, captured[0])
                    else:
                        self.assertFalse(dispatched, captured[0])
            del ref
            gc.collect()

    def test_v4_int3_streaming_retries_tiles_with_implicit_workspace_and_small_pinned_budgets(self):
        rows = 8192
        for hidden, intermediate, experts, pinned_bytes in ((2048, 512, 128, None), (128, 128, 16, 1 << 20)):
            with (
                self.subTest(hidden=hidden, pinned_bytes=pinned_bytes),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.dict(
                    os.environ,
                    {
                        "ORT_QMOE_SHARED_RESIDENT_LOG": "1",
                        "ORT_QMOE_SHARED_PIN_EXPERTS": "0",
                        "ORT_QMOE_SHARED_STREAM_ROWS": str(rows),
                        "ORT_QMOE_MAX_SYNC_TILE_TOKENS": str(rows),
                        "ORT_QMOE_PREFILL_CHUNK_TOKENS": str(rows),
                    },
                ),
            ):
                model = self._build_int_qmoe_model(
                    directory,
                    [(3, 32, 0)],
                    num_experts=experts,
                    hidden_size=hidden,
                    inter_size=intermediate,
                    top_k=8,
                    scale_range=(0.06, 0.16),
                )
                slot_bytes = 3 * hidden * intermediate * 3 // 8 + 3 * hidden * intermediate // 32 * 2 + 8
                config = {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                    "session.moe.offload.device_expert_bytes": str(2 * experts * 4 + slot_bytes),
                    "session.moe.offload.cpu_threads": "4",
                }
                if pinned_bytes is not None:
                    config["session.moe.offload.pinned_bytes"] = str(pinned_bytes)
                rng = np.random.default_rng(33629)
                x = rng.standard_normal((rows, hidden), dtype=np.float32) * 0.5
                router = np.full((rows, experts), -10.0, dtype=np.float32)
                router[:, 1:9] = 10.0
                ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
                expected = self._run_bf16(ref, x, router)
                del ref
                for cpu_execution in ("enabled", "disabled"):
                    with self.subTest(cpu_execution=cpu_execution), _capture_native_stderr() as captured:
                        session = self._make_session(
                            model, {**config, "session.moe.offload.cpu_execution": cpu_execution}
                        )
                        actual = self._run_bf16(session, x, router)
                        del session
                        gc.collect()
                    self._assert_int_offload_parity(expected, actual, f"bounded_int3({hidden=},{cpu_execution=})")
                    stream_stats = re.search(r"stream layer=0 tiles=(\d+).* routes=(\d+)", captured[0])
                    self.assertIsNotNone(stream_stats, captured[0])
                    self.assertGreater(int(stream_stats[1]), 1, captured[0])
                    self.assertEqual(int(stream_stats[2]), rows * 8, captured[0])

    def test_v4_int3_fp16_grouped_resident_and_streamed_prefill_parity(self):
        rows = 33  # Each of the two selected experts has a one-route partial tile.
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1", "ORT_QMOE_SHARED_PIN_EXPERTS": "0"}),
        ):
            model = self._build_int_qmoe_model(directory, [(3, 32, 0)], use_fp16=True)
            rng = np.random.default_rng(33629)
            x = rng.standard_normal((rows, 256), dtype=np.float32) * 0.5
            reference = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            for streamed in (False, True):
                with self.subTest(streamed=streamed):
                    router = np.full((rows, 8), -10.0, dtype=np.float32)
                    router[:, [1, 2] if streamed else [0, 3]] = 10.0
                    expected = self._run_fp16(reference, x, router)
                    results = {}
                    for grouped in (False, True):
                        with _capture_native_stderr() as captured:
                            session = self._make_session(
                                model,
                                {
                                    "session.moe.offload.mode": "required",
                                    "session.moe.offload.placement": "static",
                                    "session.moe.offload.popularity_file": "expert-profile.bin",
                                    "session.moe.offload.cpu_execution": "disabled" if streamed else "enabled",
                                    "session.moe.offload.device_expert_bytes": str(64 + 3 * 43016),
                                    "session.moe.offload.prefill_wave_experts": "2",
                                    "ep.cuda.qmoe_disable_int3_tensorcore_prefill": "0" if grouped else "1",
                                },
                            )
                            results[grouped] = self._run_fp16(session, x, router)
                            del session
                            gc.collect()
                        dispatches = re.findall(
                            r"INT3 prefill layer=0 source=(\w+) routes=(\d+) grouped=(\d+)", captured[0]
                        )
                        self.assertEqual(
                            dispatches,
                            [("streamed" if streamed else "resident", str(rows * 2), str(int(grouped)))],
                            captured[0],
                        )
                        slots, hits, routes = _resident_log_rows(captured[0])[0]
                        self.assertEqual(slots, 3, captured[0])
                        self.assertEqual(hits, 0 if streamed else rows * 2, captured[0])
                        self.assertEqual(routes, rows * 2, captured[0])
                        self._assert_int_offload_parity(
                            expected, results[grouped], f"fp16_int3({streamed=},{grouped=})"
                        )
                    self._assert_int_offload_parity(
                        results[False],
                        results[True],
                        f"fp16_grouped_vs_ungrouped({streamed=})",
                        max_rel_l2=5e-3,
                        max_rel_inf=1e-2,
                    )
            del reference
            gc.collect()

    def test_v4_int4_streaming_reduces_scale_staging_wave_to_fit_pinned_budget(self):
        self._skip_unless_int4_bf16_runner()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1"}),
        ):
            model = self._build_int_qmoe_model(directory, [(4, 32, 1)], num_experts=16, top_k=8)
            rng = np.random.default_rng(33630)
            x = rng.standard_normal((64, 256), dtype=np.float32) * 0.5
            router = np.full((64, 16), -10.0, dtype=np.float32)
            router[:, 1:9] = 10.0
            ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            expected = self._run_bf16(ref, x, router)
            del ref
            with _capture_native_stderr() as captured:
                session = self._make_session(
                    model,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "static",
                        "session.moe.offload.popularity_file": "expert-profile.bin",
                        "session.moe.offload.cpu_execution": "disabled",
                        "session.moe.offload.device_expert_bytes": str(128 + 55304),
                        "session.moe.offload.pinned_bytes": str(64 << 10),
                        "session.moe.offload.prefill_wave_experts": "16",
                    },
                )
                actual = self._run_bf16(session, x, router)
                del session
                gc.collect()
            self._assert_int_offload_parity(expected, actual, "bounded_int4_scale_staging")
            stream_stats = re.search(r"stream layer=0 tiles=(\d+).* waves=(\d+)", captured[0])
            self.assertIsNotNone(stream_stats, captured[0])
            self.assertGreater(int(stream_stats[2]), int(stream_stats[1]), captured[0])

    def test_v4_int4_cpu_disabled_streaming_uses_full_pinned_budget(self):
        self._skip_unless_int4_bf16_runner()
        hidden, intermediate, experts, rows = 2048, 512, 16, 8
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1", "ORT_QMOE_SHARED_PIN_EXPERTS": "0"}),
        ):
            model = self._build_int_qmoe_model(
                directory, [(4, 32, 1)], num_experts=experts, hidden_size=hidden, inter_size=intermediate, top_k=8
            )
            rng = np.random.default_rng(33629)
            x = rng.standard_normal((rows, hidden), dtype=np.float32) * 0.5
            router = np.full((rows, experts), -10.0, dtype=np.float32)
            router[:, 1:9] = 10.0
            reference = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            expected = self._run_bf16(reference, x, router)
            del reference

            slot_bytes = 3 * hidden * intermediate // 2 + 3 * hidden * intermediate // 32 * 2 + 8
            with _capture_native_stderr() as captured:
                session = self._make_session(
                    model,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "static",
                        "session.moe.offload.popularity_file": "expert-profile.bin",
                        "session.moe.offload.cpu_execution": "disabled",
                        "session.moe.offload.device_expert_bytes": str(2 * experts * 4 + slot_bytes),
                        "session.moe.offload.pinned_bytes": str(512 << 10),
                        "session.moe.offload.prefill_wave_experts": "16",
                    },
                )
                actual = self._run_bf16(session, x, router)
                del session
                gc.collect()
            self._assert_int_offload_parity(expected, actual, "cpu_disabled_full_pinned_budget")
            stream_stats = re.search(
                r"stream layer=0 tiles=(\d+) experts=(\d+) routes=(\d+) cpu_routes=(\d+) waves=(\d+)", captured[0]
            )
            self.assertIsNotNone(stream_stats, captured[0])
            self.assertEqual(int(stream_stats[3]), rows * 8, captured[0])
            self.assertEqual(int(stream_stats[4]), 0, captured[0])
            self.assertGreater(int(stream_stats[5]), 1, captured[0])

    def test_v4_int4_tensorcore_offload_decode_prefill_and_cuda_graph_parity(self):
        self._skip_unless_int4_bf16_runner()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(
                os.environ,
                {
                    "ORT_QMOE_SHARED_RESIDENT_LOG": "1",
                    "ORT_QMOE_SHARED_CPU_TIMING": "1",
                    "ORT_QMOE_INT4_PREFILL_GEMM": "0",
                },
            ),
        ):
            # H=512 satisfies FC1's k % 512 == 0 Tensor-Core gate; I=128 satisfies 2*I % 64 == 0 and FC2's k % 128 == 0.
            # Per-expert GPU slot bytes: 2*128*256 + 512*64 + 2*128*16*2 + 512*4*2 + 8 = 110,600 B;
            # FC1 bias slot = 512 B; FC2 bias per layer = 8*512*2 = 8,192 B; fixed = 64 + 2*8192 + 8*512 = 20,544 B.
            model = self._build_int_qmoe_model(
                tmpdir,
                [(4, 32, 1)],
                num_experts=8,
                hidden_size=512,
                inter_size=128,
                top_k=8,
                with_bias=True,
                scale_range=(0.02, 0.06),
            )
            ref = self._make_session(
                model,
                {
                    "session.moe.offload.mode": "disabled",
                    "ep.cuda.disable_moe_gemv_tc": "1",
                },
            )
            offload_cfg = {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "static",
                "session.moe.offload.popularity_file": "expert-profile.bin",
                "session.moe.offload.cpu_execution": "enabled",
                "session.moe.offload.device_expert_bytes": str(20544 + 4 * (110600 + 512)),
                "session.moe.offload.prefill_wave_experts": "2",
            }
            rng = np.random.default_rng(33731)

            try:
                with self.subTest(case="hybrid_decode_and_sync_prefill"):
                    with _capture_native_stderr() as captured:
                        sess = self._make_session(model, offload_cfg)
                        try:
                            # 6 tokens x top_k=8 = 48 routes (33..256): async hybrid decode with unsorted hit_compact,
                            # source-row indirection via hit_ids, and 4 resident + 4 cold experts per token.
                            x_dec = _round_to_act_dtype(rng.standard_normal((6, 512), dtype=np.float32) * 0.5)
                            r_dec = rng.standard_normal((6, 8), dtype=np.float32)
                            self._assert_int_offload_parity(
                                self._run_bf16(ref, x_dec, r_dec),
                                self._run_bf16(sess, x_dec, r_dec),
                                "int4_tc_decode(T=6,routes=48)",
                            )
                            # 10 tokens x top_k=8 = 80 routes (> async_max_tokens=8): synchronous resident + cold path.
                            x_sync = _round_to_act_dtype(rng.standard_normal((10, 512), dtype=np.float32) * 0.5)
                            r_sync = rng.standard_normal((10, 8), dtype=np.float32)
                            self._assert_int_offload_parity(
                                self._run_bf16(ref, x_sync, r_sync),
                                self._run_bf16(sess, x_sync, r_sync),
                                "int4_tc_sync(T=10,routes=80)",
                            )
                        finally:
                            del sess
                            gc.collect()
                    self.assertIn("INT4 TC decode layer=0 routes=48", captured[0])
                    self.assertIn("INT4 TC prefill layer=0 source=resident routes=40", captured[0])
                    calls, total_routes, cold_routes = _cpu_path_route_rows(captured[0])[0]
                    self.assertGreater(calls, 0, captured[0])
                    self.assertGreater(cold_routes, 0, captured[0])
                    self.assertEqual(2 * cold_routes, total_routes, captured[0])

                with self.subTest(case="cuda_graph_replay"):
                    cudart = _load_cudart()
                    if cudart is None:
                        self.skipTest("CUDA runtime library is required to update captured GPU inputs in place")
                    graph_model = _wrap_qmoe_model_with_cast_io(
                        model, os.path.join(tmpdir, "graph_int4_tc.onnx"), 512, 8
                    )
                    with _capture_native_stderr() as captured:
                        sess_g = self._make_session(
                            graph_model,
                            offload_cfg,
                            cuda_options={"enable_cuda_graph": "1", "min_num_runs_before_cuda_graph_capture": "2"},
                        )
                        device_id = self._device_id(self._selected_plugin_device())
                        x_ptr = _cuda_malloc(cudart, 6 * 512 * 4, device_id)
                        r_ptr = _cuda_malloc(cudart, 6 * 8 * 4, device_id)
                        try:
                            binding = sess_g.io_binding()
                            binding.bind_input("X_f32", "cuda", device_id, np.float32, [6, 512], x_ptr.value)
                            binding.bind_input("R_f32", "cuda", device_id, np.float32, [6, 8], r_ptr.value)
                            binding.bind_output("Y_f32", "cuda", device_id)
                            for step in range(6):
                                x_step = _round_to_act_dtype(rng.standard_normal((6, 512), dtype=np.float32) * 0.5)
                                r_step = np.roll(
                                    np.linspace(4.0, -4.0, 8, dtype=np.float32).reshape(1, 8).repeat(6, axis=0),
                                    step,
                                    axis=1,
                                )
                                _cuda_upload(cudart, x_ptr, x_step)
                                _cuda_upload(cudart, r_ptr, r_step)
                                sess_g.run_with_iobinding(binding)
                                actual_g = binding.copy_outputs_to_cpu()[0]
                                self._assert_int_offload_parity(
                                    self._run_bf16(ref, x_step, r_step),
                                    actual_g,
                                    f"int4_tc_graph(step={step})",
                                )
                            del binding
                        finally:
                            cudart.cudaFree(x_ptr)
                            cudart.cudaFree(r_ptr)
                            del sess_g
                            gc.collect()
                    self.assertIn("INT4 TC decode layer=0 routes=48", captured[0])
            finally:
                del ref
                gc.collect()

    def test_v4_int3_tensorcore_offload_decode_and_grouped_prefill_parity(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(
                os.environ,
                {
                    "ORT_QMOE_SHARED_RESIDENT_LOG": "1",
                    "ORT_QMOE_SHARED_CPU_TIMING": "1",
                    "ORT_QMOE_SHARED_PIN_EXPERTS": "0",
                },
            ),
        ):
            # H=512, I=128, INT3 block_size=64:
            # GPU slot bytes: 2*128*(512*3//8) + 512*(128*3//8) + 2*128*8*2 + 512*2*2 + 8 = 79,880 B;
            # FC1 bias slot = 512 B; FC2 bias per layer = 8*512*2 = 8,192 B; fixed = 64 + 2*8192 + 8*512 = 20,544 B.
            dir_dec = os.path.join(tmpdir, "int3_tc_dec")
            os.makedirs(dir_dec, exist_ok=True)
            model_dec = self._build_int_qmoe_model(
                dir_dec,
                [(3, 64, 0)],
                num_experts=8,
                hidden_size=512,
                inter_size=128,
                top_k=8,
                with_bias=True,
                scale_range=(0.04, 0.12),
            )
            ref_dec = self._make_session(
                model_dec,
                {
                    "session.moe.offload.mode": "disabled",
                    "ep.cuda.disable_qmoe_int3_tc": "1",
                    "ep.cuda.qmoe_disable_int3_tensorcore_prefill": "1",
                },
            )
            rng = np.random.default_rng(33732)
            with self.subTest(case="mixed_resident_cold_decode"):
                with _capture_native_stderr() as captured:
                    sess_dec = self._make_session(
                        model_dec,
                        {
                            "session.moe.offload.mode": "required",
                            "session.moe.offload.placement": "static",
                            "session.moe.offload.popularity_file": "expert-profile.bin",
                            "session.moe.offload.cpu_execution": "enabled",
                            "session.moe.offload.device_expert_bytes": str(20544 + 4 * (79880 + 512)),
                        },
                    )
                    try:
                        # 6 tokens x top_k=8 = 48 routes (>= 33): async decode with source_route_ids=hit_ids,
                        # output=nullptr, and 4 resident + 4 cold experts per token.
                        x_d = _round_to_act_dtype(rng.standard_normal((6, 512), dtype=np.float32) * 0.5)
                        r_d = rng.standard_normal((6, 8), dtype=np.float32)
                        self._assert_int_offload_parity(
                            self._run_bf16(ref_dec, x_d, r_d),
                            self._run_bf16(sess_dec, x_d, r_d),
                            "int3_tc_decode(T=6,routes=48)",
                        )
                    finally:
                        del sess_dec
                        gc.collect()
                self.assertIn("INT3 decode layer=0 routes=48 tc=1", captured[0])
                calls, total_routes, cold_routes = _cpu_path_route_rows(captured[0])[0]
                self.assertGreater(calls, 0, captured[0])
                self.assertEqual(2 * cold_routes, total_routes, captured[0])
            del ref_dec

            dir_pre = os.path.join(tmpdir, "int3_tc_pre")
            os.makedirs(dir_pre, exist_ok=True)
            model_pre = self._build_int_qmoe_model(
                dir_pre,
                [(3, 64, 0)],
                num_experts=8,
                hidden_size=512,
                inter_size=128,
                top_k=2,
                with_bias=True,
                scale_range=(0.04, 0.12),
            )
            ref_pre = self._make_session(
                model_pre,
                {
                    "session.moe.offload.mode": "disabled",
                    "ep.cuda.disable_qmoe_int3_tc": "1",
                    "ep.cuda.qmoe_disable_int3_tensorcore_prefill": "1",
                },
            )
            with self.subTest(case="ungrouped_gathered_tc"):
                # 17 rows x top_k=2 = 34 routes (>= 33) with grouped prefill disabled exercises TryLaunchBf16 with
                # input_is_gathered=true and output=nullptr on both resident and streamed paths.
                x_u = _round_to_act_dtype(rng.standard_normal((17, 512), dtype=np.float32) * 0.5)
                r_u = np.full((17, 8), -10.0, dtype=np.float32)
                r_u[:, [0, 4]] = 10.0  # Expert 0 is resident; expert 4 streams when cpu_execution="disabled".
                expected_u = self._run_bf16(ref_pre, x_u, r_u)
                with _capture_native_stderr() as captured:
                    sess_u = self._make_session(
                        model_pre,
                        {
                            "session.moe.offload.mode": "required",
                            "session.moe.offload.placement": "static",
                            "session.moe.offload.popularity_file": "expert-profile.bin",
                            "session.moe.offload.cpu_execution": "disabled",
                            "session.moe.offload.device_expert_bytes": str(20544 + 2 * (79880 + 512)),
                            "session.moe.offload.prefill_wave_experts": "2",
                            "ep.cuda.qmoe_disable_int3_tensorcore_prefill": "1",
                            "ep.cuda.qmoe_int3_tc_min_routes": "16",
                        },
                    )
                    try:
                        actual_u = self._run_bf16(sess_u, x_u, r_u)
                    finally:
                        del sess_u
                        gc.collect()
                self._assert_int_offload_parity(expected_u, actual_u, "int3_tc_ungrouped_gathered(T=17)")
                dispatches_u = re.findall(
                    r"INT3 prefill layer=0 source=(\w+) routes=(\d+) grouped=(\d+) tc=(\d+)", captured[0]
                )
                self.assertEqual(
                    dispatches_u,
                    [("resident", "17", "0", "1"), ("streamed", "17", "0", "1")],
                    captured[0],
                )

            with self.subTest(case="grouped_streaming_prefill_above_255_routes"):
                # 165 rows x top_k=2 = 330 routes (> 255): 165 routes go to resident experts (0, 3) and 165 routes go
                # to streamed cold experts (1, 2). Each expert gets 82 or 83 routes (10 full 8-route tiles + a 2- or
                # 3-route partial tile), and the wave route IDs in sorted_routes span 0..164 (> 128) and when all 330
                # routes go to streamed cold experts (1, 2), sorted_routes spans 0..329 (> 255).
                x_g = _round_to_act_dtype(rng.standard_normal((165, 512), dtype=np.float32) * 0.5)
                for mode_name, target_cols, expected_dispatches in (
                    (
                        "mixed_resident_and_streamed",
                        ([0, 1], [3, 2]),
                        [("resident", "165", "1", "1"), ("streamed", "165", "1", "1")],
                    ),
                    (
                        "streamed_routes_above_255",
                        ([1, 2], [4, 5]),
                        [("streamed", "330", "1", "1")],
                    ),
                ):
                    r_g = np.full((165, 8), -10.0, dtype=np.float32)
                    r_g[0::2, target_cols[0]] = 10.0
                    r_g[1::2, target_cols[1]] = 10.0
                    expected_g = self._run_bf16(ref_pre, x_g, r_g)
                    with _capture_native_stderr() as captured:
                        sess_g = self._make_session(
                            model_pre,
                            {
                                "session.moe.offload.mode": "required",
                                "session.moe.offload.placement": "static",
                                "session.moe.offload.popularity_file": "expert-profile.bin",
                                "session.moe.offload.cpu_execution": "disabled",
                                "session.moe.offload.device_expert_bytes": str(20544 + 2 * (79880 + 512)),
                                "session.moe.offload.prefill_wave_experts": "4",
                                "ep.cuda.qmoe_disable_int3_tensorcore_prefill": "0",
                            },
                        )
                        try:
                            actual_g = self._run_bf16(sess_g, x_g, r_g)
                        finally:
                            del sess_g
                            gc.collect()
                    self._assert_int_offload_parity(expected_g, actual_g, f"int3_tc_grouped({mode_name})")
                    dispatches_g = re.findall(
                        r"INT3 prefill layer=0 source=(\w+) routes=(\d+) grouped=(\d+) tc=(\d+)", captured[0]
                    )
                    self.assertEqual(dispatches_g, expected_dispatches, captured[0])
            del ref_pre
            gc.collect()

    def test_v4_int4_int2_all_resident_unsorted_routes_above_256(self):
        rng = np.random.default_rng(33787)
        for bits, block, slot_bytes in ((4, 32, 110600), (2, 64, 55304)):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as directory:
                if bits == 4:
                    self._skip_unless_int4_bf16_runner()
                model = self._build_int_qmoe_model(directory, [(bits, block, 1)], hidden_size=512)
                reference = self._make_session(model, {"session.moe.offload.mode": "disabled"})
                resident = self._make_session(
                    model,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "static",
                        "session.moe.offload.device_expert_bytes": str(64 + 8 * slot_bytes),
                    },
                )
                for rows in (129, 257):
                    x = rng.standard_normal((rows, 512), dtype=np.float32) * 0.5
                    router = rng.standard_normal((rows, 8), dtype=np.float32)
                    expected = self._run_bf16(reference, x, router)
                    actual = self._run_bf16(resident, x, router)
                    self._assert_int_offload_parity(expected, actual, f"resident_int{bits}({rows=})")
                del resident, reference

    def test_v4_int4_int3_int2_streaming_stages_weights_with_small_explicit_pinned_budget(self):
        for bits, block, prepacked, slot_bytes in ((4, 32, 1, 110600), (3, 64, 0, 79880), (2, 64, 1, 55304)):
            with (
                self.subTest(bits=bits),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1"}),
            ):
                if bits == 4:
                    self._skip_unless_int4_bf16_runner()
                model = self._build_int_qmoe_model(directory, [(bits, block, prepacked)], hidden_size=512)
                x = np.random.default_rng(bits).standard_normal((16, 512), dtype=np.float32) * 0.5
                router = np.tile(np.linspace(-4.0, 4.0, 8, dtype=np.float32), (16, 1))
                reference = self._make_session(model, {"session.moe.offload.mode": "disabled"})
                expected = self._run_bf16(reference, x, router)
                del reference
                with _capture_native_stderr() as captured:
                    session = self._make_session(
                        model,
                        {
                            "session.moe.offload.mode": "required",
                            "session.moe.offload.placement": "static",
                            "session.moe.offload.cpu_execution": "disabled",
                            "session.moe.offload.device_expert_bytes": str(64 + slot_bytes),
                            "session.moe.offload.pinned_bytes": str(1 << 20),
                            "session.moe.offload.prefill_wave_experts": "1",
                        },
                    )
                    actual = self._run_bf16(session, x, router)
                    del session
                    gc.collect()
                self._assert_int_offload_parity(expected, actual, f"pinned_staging_int{bits}")
                staging = re.search(r"staging layer=0 weight_banks=(\d+) pinned_bytes=(\d+)", captured[0])
                self.assertIsNotNone(staging, captured[0])
                self.assertEqual(int(staging[1]), 2, captured[0])
                self.assertLessEqual(int(staging[2]), 1 << 20, captured[0])

    def test_v4_int2_fp16_cold_expert_swiglu_projection_overflow_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = self._build_int_qmoe_model(
                directory, [(2, 64, 1)], hidden_size=128, inter_size=128, top_k=1, use_fp16=True
            )
            model = onnx.load(model_path, load_external_data=False)
            banks = {value.name: value for value in model.graph.initializer}
            fc1_weights = np.full((8, 128 * 128 // 2), 0xAA, dtype=np.uint8)
            fc1_weights[:, 0] = 0xA9
            fc1_weights[:, 16] = 0xAB
            fc1_scales = np.ones((8, 256, 2), dtype=np.float16)
            # Overflow the negative gate projection, not the positive up projection.
            fc1_scales[:, 0, :] = 8192
            fc2_weights = np.full((8, 128 * 128 // 4), 0xAA, dtype=np.uint8)
            for n in range(128):
                fc2_weights[:, (n // 8) * 256 + (n % 8) * 16] = 0xAB
            fc2_scales = np.ones((8, 128, 2), dtype=np.float16)
            for slot, values in ((2, fc1_weights), (3, fc1_scales), (5, fc2_weights), (6, fc2_scales)):
                bank = banks[model.graph.node[0].input[slot]]
                external = {entry.key: entry.value for entry in bank.external_data}
                self.assertEqual(values.nbytes, int(external["length"]))
                with (Path(directory) / external["location"]).open("r+b") as stream:
                    stream.seek(int(external["offset"]))
                    stream.write(values.tobytes())
            x = np.zeros((1, 128), dtype=np.float32)
            x[0, 0] = 16
            router = np.full((1, 8), -100, dtype=np.float32)
            router[0, 7] = 10
            reference = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(64 + 8 * 13832),
                },
            )
            expected = self._run_fp16(reference, x, router)
            del reference
            outputs, telemetry = self._run_decode_with_telemetry(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(64 + 2 * 13832),
                },
                [(x, router)],
            )
            self.assertTrue(np.all(np.isfinite(expected)))
            self.assertTrue(np.all(np.isfinite(outputs[0])))
            np.testing.assert_array_equal(expected, np.zeros_like(expected))
            np.testing.assert_allclose(outputs[0], expected, rtol=0, atol=1e-3)
            self.assertEqual(telemetry[0]["cold_routes"], 1)
            self.assertEqual(telemetry[0]["cpu_doorbell_wakeups"], 1)

    def test_v4_int2_bf16_projection_rounding_before_bias(self):
        self._check_v4_int2_bf16_projection_rounding_before_bias()

    def test_v4_int2_bf16_projection_rounding_before_bias_tensorcore_dispatch(self):
        capability = _cuda_device_compute_capability()
        if capability is None or capability < (8, 0):
            self.skipTest("INT2 BF16 Tensor Core dispatch requires SM80+")
        self._check_v4_int2_bf16_projection_rounding_before_bias(verify_tensorcore=True)

    def _check_v4_int2_bf16_projection_rounding_before_bias(self, verify_tensorcore: bool = False):
        with tempfile.TemporaryDirectory() as directory:
            model_path = self._build_int_qmoe_model(
                directory, [(2, 64, 1)], hidden_size=128, inter_size=128, top_k=8, with_bias=True, swiglu_limit=1.0
            )
            model = onnx.load(model_path, load_external_data=False)
            banks = {value.name: value for value in model.graph.initializer}
            fc1_weights = np.full((8, 128 * 128 // 2), 0xAA, dtype=np.uint8)
            fc1_weights[:, 0] = 0xAB
            fc1_weights[:, 128] = 0xAB
            fc1_weights[:, 16] = 0xAB
            fc2_weights = np.full((8, 128 * 128 // 4), 0xAA, dtype=np.uint8)
            for n in range(128):
                fc2_weights[:, (n // 8) * 256 + (n % 8) * 16] = 0xAB
            fc1_bias = np.zeros((8, 256), dtype=np.float32)
            fc1_bias[:, 0] = -1
            values_by_slot = {
                2: fc1_weights,
                3: _float32_to_bfloat16_u16(np.ones((8, 256, 2), dtype=np.float32)),
                4: _float32_to_bfloat16_u16(fc1_bias),
                5: fc2_weights,
                6: _float32_to_bfloat16_u16(np.ones((8, 128, 2), dtype=np.float32)),
                7: _float32_to_bfloat16_u16(np.full((8, 128), 0.125, dtype=np.float32)),
            }
            for slot, values in values_by_slot.items():
                external = {entry.key: entry.value for entry in banks[model.graph.node[0].input[slot]].external_data}
                self.assertEqual(values.nbytes, int(external["length"]))
                with (Path(directory) / external["location"]).open("r+b") as stream:
                    stream.seek(int(external["offset"]))
                    stream.write(values.tobytes())

            # Five decode tokens produce forty routes, above the default Tensor Core threshold of 33.
            x = np.zeros((5, 128), dtype=np.float32)
            x[:, 0] = 1
            router = np.zeros((5, 8), dtype=np.float32)
            cases = []
            expectations = []
            # The midpoint rounds to 1 before bias cancellation; the second case retains a nonzero signal.
            for delta in (2**-8, 2**-6):
                x[:, 64] = delta
                gate = _round_to_act_dtype(np.array([1 + delta], dtype=np.float32)) - 1
                activated = _round_to_act_dtype(gate / (1 + np.exp(-gate)))
                cases.append((x.copy(), router))
                expectations.append(_round_to_act_dtype(activated + 0.125).item())
            # Finite BF16 inputs overflow only the gate sum; the up projection is clamped to -1.
            x[:, (0, 64)] = -(2.0**127)
            cases.append((x, router))
            expectations.append(0.125)
            slot_bytes = 13832 + (256 + 128) * 2
            for disable_tc, cpu_offload in ((1, False), (0, False), (0, True)):
                if verify_tensorcore and (disable_tc or cpu_offload):
                    continue
                with self.subTest(disable_tc=disable_tc, cpu_offload=cpu_offload):
                    config = {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "static",
                        "ep.cuda.disable_moe_gemv_tc": str(disable_tc),
                    }
                    if cpu_offload:
                        config["session.moe.offload.device_expert_bytes"] = str(64 + 2 * slot_bytes)
                        outputs, telemetry = self._run_decode_with_telemetry(model_path, config, cases, use_fp16=False)
                        self.assertGreater(sum(step["cold_routes"] for step in telemetry), 0)
                        self.assertGreater(sum(step["cpu_doorbell_wakeups"] for step in telemetry), 0)
                    else:
                        session = self._make_session(
                            model_path,
                            config,
                            profile_file_prefix=str(Path(directory) / "int2_tc") if verify_tensorcore else None,
                        )
                        try:
                            outputs = [self._run_bf16(session, inputs, routing) for inputs, routing in cases]
                        finally:
                            if verify_tensorcore:
                                profile_path = session.end_profiling()
                        del session
                    for actual, expected in zip(outputs, expectations, strict=True):
                        np.testing.assert_array_equal(actual, np.full_like(actual, expected))
                    if verify_tensorcore:
                        events = json.loads(Path(profile_path).read_text())
                        names = [event["name"] for event in events if event.get("cat") == "Kernel"]
                        if not names:
                            self.skipTest("CUDA kernel profiling unavailable; requires a CUPTI-enabled build")
                        for swiglu, encoded in (("true", 1), ("false", 0)):
                            launches = [
                                name
                                for name in names
                                if re.search(rf"MoeGemvTcInt2Kernel<\s*(?:{swiglu}|\(bool\)\s*{encoded}),", name)
                            ]
                            self.assertEqual(len(launches), len(cases), names)

    def test_v4_int2_bf16_tensorcore_scale_limits_and_dispatch(self):
        # The dispatch logging flag is cached on first use; isolate it from other tests.
        script = """
import sys
sys.path.insert(0, sys.argv[1])
from test_qmoe_offload_v4 import TestQMoEOffloadV4
TestQMoEOffloadV4.setUpClass()
TestQMoEOffloadV4()._check_int2_bf16_tensorcore_scale_limits_and_dispatch()
"""
        environment = dict(os.environ, ORT_QMOE_SHARED_RESIDENT_LOG="1")
        environment.pop("ORT_MOE_GEMV_TC_MIN_ROWS", None)
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve().parent)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _check_int2_bf16_tensorcore_scale_limits_and_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            tc_dir = os.path.join(directory, "int2_bf16_tc_scale_ovf")
            os.makedirs(tc_dir)
            model_path = self._build_int_qmoe_model(
                tc_dir, [(2, 64, 1)], num_experts=8, hidden_size=512, inter_size=512, top_k=8, use_fp16=False
            )
            model = onnx.load(model_path, load_external_data=False)
            banks = {value.name: value for value in model.graph.initializer}
            # Code 2 (0b10 -> centered 0) is 0xAA. For column 0 (gate) set K=0 code to 0 (-2 centered -> 0xA8);
            # for column 1 (up) set K=0 code to 3 (+1 centered -> 0xAB).
            fc1_weights = np.full((8, 512 * 512 // 2), 0xAA, dtype=np.uint8)
            fc1_weights[:, 0] = 0xA8
            fc1_weights[:, 16] = 0xAB
            fc1_scales_f32 = np.ones((8, 1024, 8), dtype=np.float32)
            fc2_weights = np.full((8, 512 * 512 // 4), 0xAA, dtype=np.uint8)
            for n in range(512):
                fc2_weights[:, (n // 8) * 1024 + (n % 8) * 16] = 0xAB
            fc2_scales = _float32_to_bfloat16_u16(np.ones((8, 512, 8), dtype=np.float32))
            for slot, values in ((2, fc1_weights), (5, fc2_weights), (6, fc2_scales)):
                bank = banks[model.graph.node[0].input[slot]]
                external = {entry.key: entry.value for entry in bank.external_data}
                self.assertEqual(values.nbytes, int(external["length"]))
                with (Path(tc_dir) / external["location"]).open("r+b") as stream:
                    stream.seek(int(external["offset"]))
                    stream.write(values.tobytes())

            slot_h512_i512 = 2 * 512 * 128 + 512 * 128 + 2 * 512 * 8 * 2 + 512 * 8 * 2 + 8
            standard = {"session.moe.offload.mode": "disabled", "ep.cuda.disable_moe_gemv_tc": "0"}
            resident = {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "static",
                "session.moe.offload.device_expert_bytes": str(64 + 8 * slot_h512_i512),
                "ep.cuda.disable_moe_gemv_tc": "0",
            }
            hybrid = {
                **resident,
                "session.moe.offload.popularity_file": "expert-profile.bin",
                "session.moe.offload.device_expert_bytes": str(64 + 4 * slot_h512_i512),
            }
            cases = (
                ("simt", {**standard, "ep.cuda.disable_moe_gemv_tc": "1"}, 5, None, 33),
                ("tc_default", standard, 5, 40, 33),
                ("resident_default", resident, 5, 40, 33),
                ("hybrid_default", hybrid, 5, 40, 33),
                ("tc_override", {**standard, "ep.cuda.moe_gemv_tc_min_rows": "24"}, 3, 24, 24),
                ("resident_override", {**resident, "ep.cuda.moe_gemv_tc_min_rows": "24"}, 3, 24, 24),
                ("hybrid_override", {**hybrid, "ep.cuda.moe_gemv_tc_min_rows": "24"}, 3, 24, 24),
                ("tc_below_threshold", {**standard, "ep.cuda.moe_gemv_tc_min_rows": "41"}, 5, None, 41),
                ("resident_below_threshold", {**resident, "ep.cuda.moe_gemv_tc_min_rows": "41"}, 5, None, 41),
            )
            for scale_exp, act_exp in ((127, -126), (-133, 126)):
                fc1_scales_f32[:, :2, 0] = np.ldexp(1.0, scale_exp)
                fc1_scales = _float32_to_bfloat16_u16(fc1_scales_f32)
                external = {entry.key: entry.value for entry in banks[model.graph.node[0].input[3]].external_data}
                with (Path(tc_dir) / external["location"]).open("r+b") as stream:
                    stream.seek(int(external["offset"]))
                    stream.write(fc1_scales.tobytes())

                # All experts are identical, so normalized routing preserves this independently calculated value.
                gate = -2.0 * np.ldexp(1.0, scale_exp + act_exp)
                up = np.ldexp(1.0, scale_exp + act_exp)
                expected = float(_round_to_act_dtype(np.array([gate / (1.0 + np.exp(-gate)) * up]))[0])
                self.assertNotEqual(expected, 0.0)
                for name, config, rows, tc_routes, min_rows in cases:
                    # Hybrid tiny-scale output mixes CPU and TC results; test the lower-range limit on GPU only.
                    if scale_exp == -133 and name.startswith("hybrid"):
                        continue
                    with self.subTest(scale_exp=scale_exp, case=name):
                        x = np.zeros((rows, 512), dtype=np.float32)
                        x[:, 0] = np.ldexp(1.0, act_exp)
                        router = np.tile(np.linspace(4.0, -4.0, 8, dtype=np.float32), (rows, 1))
                        with _capture_native_stderr() as captured:
                            session = self._make_session(model_path, config)
                            try:
                                actual = self._run_bf16(session, x, router)
                            finally:
                                del session
                                gc.collect()
                        dispatches = re.findall(
                            r"QMoE INT2 TensorCore swiglu=(\d+) routes=(\d+) min_rows=(\d+)", captured[0]
                        )
                        if tc_routes is None:
                            self.assertEqual(dispatches, [], f"{scale_exp=} {name=}\n{captured[0]}")
                        else:
                            self.assertEqual(
                                dispatches,
                                [("1", str(tc_routes), str(min_rows)), ("0", str(tc_routes), str(min_rows))],
                                f"{scale_exp=} {name=}\n{captured[0]}",
                            )
                        # The half-scaled +1 weight at scale 2^-133 is lost before MMA, unlike SIMT.
                        expected_value = 0.0 if scale_exp == -133 and tc_routes is not None else expected
                        np.testing.assert_allclose(
                            actual,
                            np.full((rows, 512), expected_value, dtype=np.float32),
                            rtol=0,
                            atol=0,
                            err_msg=f"{scale_exp=} {name=}",
                        )

    def test_v4_int2_bounce_alignment_and_fc3_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            # 2. INT2 H=64, I=448, block_size=64 background promotion bounce buffer alignment with unregistered banks,
            # explicit pinned_bytes, and ORT_QMOE_STEP_PROMOTION_BUDGET=1.
            promo_dir = os.path.join(directory, "int2_h64_i448_promo")
            os.makedirs(promo_dir)
            promo_model = self._build_int_qmoe_model(
                promo_dir, [(2, 64, 1)], num_experts=8, hidden_size=64, inter_size=448, top_k=2
            )
            # H=64, I=448, INT2 b64: w1=14336, w2=7168, s1=1792, s2=896, g1=4, g2=4 -> raw=24,200 B;
            # 256-aligned bank sum = 14336 + 7168 + 1792 + 1024 + 256 + 256 = 24,832 B (> 24,576 B).
            slot_h64_i448 = 24200
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "ORT_QMOE_SHARED_PIN_EXPERTS": "0",
                        "ORT_QMOE_STEP_PROMOTION_BUDGET": "1",
                        "ORT_QMOE_SHARED_RESIDENT_LOG": "1",
                    },
                ),
                _capture_native_stderr() as promo_log,
            ):
                ref_p = self._make_session(promo_model, {"session.moe.offload.mode": "disabled"})
                sess_p = self._make_session(
                    promo_model,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.popularity_file": "expert-profile.bin",
                        "session.moe.offload.device_expert_bytes": str(64 + 4 * slot_h64_i448),
                        "session.moe.offload.pinned_bytes": str(2 << 20),
                    },
                )
                rng = np.random.default_rng(33789)
                for step in range(6):
                    x_p = rng.standard_normal((4, 64), dtype=np.float32) * 0.5
                    r_p = np.full((4, 8), -20.0, dtype=np.float32)
                    r_p[:, 5] = 6.0
                    r_p[:, 7] = 5.0
                    self._assert_int_offload_parity(
                        self._run_bf16(ref_p, x_p, r_p),
                        self._run_bf16(sess_p, x_p, r_p),
                        f"int2_h64_i448_promo(step={step})",
                    )
                del sess_p, ref_p
                gc.collect()
            self.assertGreater(_resident_promotions_per_layer(promo_log[0]).get(0, 0), 0, promo_log[0])

            # 3. INT2 node with swiglu_fusion=0 and fc3_expert_weight_bits=4 must be rejected in mode='required'
            # and skipped by session placement in mode='auto'.
            fc3_dir = os.path.join(directory, "int2_fc3_mismatch")
            os.makedirs(fc3_dir)
            fc3_model_path = self._build_int_qmoe_model(
                fc3_dir, [(2, 64, 1)], num_experts=8, hidden_size=128, inter_size=128, top_k=2, swiglu_fusion=0
            )
            fc3_proto = onnx.load_model(fc3_model_path, load_external_data=False)
            fc3_proto.graph.node[0].attribute.append(helper.make_attribute("fc3_expert_weight_bits", 4))
            onnx.save_model(fc3_proto, fc3_model_path)
            with self.assertRaisesRegex(Exception, "quant_type='int' offload requires|rejected node"):
                self._make_session(fc3_model_path, {"session.moe.offload.mode": "required"})
            s_fc3_ref = self._make_session(fc3_model_path, {"session.moe.offload.mode": "disabled"})
            s_fc3_auto = self._make_session(fc3_model_path, {"session.moe.offload.mode": "auto"})
            x_fc3 = np.random.default_rng(33790).standard_normal((4, 128), dtype=np.float32) * 0.5
            r_fc3 = np.random.default_rng(33791).standard_normal((4, 8), dtype=np.float32)
            np.testing.assert_array_equal(
                self._run_bf16(s_fc3_ref, x_fc3, r_fc3),
                self._run_bf16(s_fc3_auto, x_fc3, r_fc3),
            )
            del s_fc3_auto, s_fc3_ref
            gc.collect()

    def test_v4_int4_eight_token_hybrid_cuda_graph_replay(self):
        self._skip_unless_int4_bf16_runner()
        cudart = _load_cudart()
        if cudart is None:
            self.skipTest("CUDA runtime library is required to update captured GPU inputs in place")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1", "ORT_QMOE_SHARED_CPU_TIMING": "1"}),
            _capture_native_stderr() as captured,
        ):
            model = self._build_int_qmoe_model(tmpdir, [(4, 32, 1)])
            graph_model = _wrap_qmoe_model_with_cast_io(model, os.path.join(tmpdir, "graph.onnx"), 256, 8)
            ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            sess = self._make_session(
                graph_model,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                    "session.moe.offload.cpu_execution": "enabled",
                    "session.moe.offload.device_expert_bytes": str(64 + 2 * 55304),
                    "session.moe.offload.host_expert_bytes": str(2 * 55808),
                },
                cuda_options={"enable_cuda_graph": "1", "min_num_runs_before_cuda_graph_capture": "2"},
            )
            device_id = self._device_id(self._selected_plugin_device())
            x_ptr = _cuda_malloc(cudart, 8 * 256 * 4, device_id)
            r_ptr = _cuda_malloc(cudart, 8 * 8 * 4, device_id)
            try:
                binding = sess.io_binding()
                binding.bind_input("X_f32", "cuda", device_id, np.float32, [8, 256], x_ptr.value)
                binding.bind_input("R_f32", "cuda", device_id, np.float32, [8, 8], r_ptr.value)
                binding.bind_output("Y_f32", "cuda", device_id)
                rng = np.random.default_rng(33624)
                for step in range(8):
                    x = _round_to_act_dtype(rng.standard_normal((8, 256), dtype=np.float32) * 0.5)
                    router = np.full((8, 8), -10.0, dtype=np.float32)
                    router[:, 0] = 10.0  # Resident hit on every token.
                    router[:, 4 + step % 4] = 10.0  # Cold routes cycle through more than the host cache can hold.
                    _cuda_upload(cudart, x_ptr, x)
                    _cuda_upload(cudart, r_ptr, router)
                    expected = self._run_bf16(ref, x, router)
                    sess.run_with_iobinding(binding)
                    actual = binding.copy_outputs_to_cpu()[0]
                    self._assert_int_offload_parity(expected, actual, f"s8_graph(step={step})")
                    # A missing CPU contribution must be observable even though every token has a resident hit.
                    hit_only = np.full_like(router, -10.0)
                    hit_only[:, 0] = 10.0
                    half_hit = 0.5 * self._run_bf16(ref, x, hit_only)
                    self.assertGreater(float(np.linalg.norm(expected - half_hit) / np.linalg.norm(expected)), 0.1)
            finally:
                cudart.cudaFree(x_ptr)
                cudart.cudaFree(r_ptr)
            del binding, sess, ref
            gc.collect()
        self.assertEqual(_resident_slots_per_layer(captured[0])[0], 2, captured[0])
        calls, total_routes, cold_routes = _cpu_path_route_rows(captured[0])[0]
        self.assertGreater(calls, 0, captured[0])
        self.assertEqual(2 * cold_routes, total_routes, captured[0])
        misses = re.findall(r"ssd_misses=(\d+)", captured[0])
        self.assertTrue(misses and any(int(count) > 0 for count in misses), captured[0])

    def test_v4_int4_omitted_or_zero_swiglu_fusion_is_interleaved_in_cutlass_prefill(self):
        self._skip_unless_int4_bf16_runner()
        rows = 512
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(
                os.environ,
                {
                    "ORT_QMOE_SHARED_RESIDENT_LOG": "1",
                    "ORT_QMOE_INT4_PREFILL_GEMM": "1",
                    "ORT_QMOE_MAX_SYNC_TILE_TOKENS": "512",
                },
            ),
        ):
            # The offload banks are always interleaved, so an omitted or zero swiglu_fusion has to behave like 1.
            ref_dir = os.path.join(tmpdir, "interleaved")
            os.makedirs(ref_dir)
            ref = self._make_session(
                self._build_int_qmoe_model(ref_dir, [(4, 32, 1)], swiglu_fusion=1),
                {"session.moe.offload.mode": "disabled"},
            )
            rng = np.random.default_rng(33626)
            x = rng.standard_normal((rows, 256), dtype=np.float32) * 0.5
            router = np.full((rows, 8), -10.0, dtype=np.float32)
            router[:, 4:6] = 10.0  # Both experts are cold, so they stream through the CUTLASS INT4 runner.
            expected = self._run_bf16(ref, x, router)
            for swiglu_fusion in (None, 0):
                with self.subTest(swiglu_fusion=swiglu_fusion):
                    model_dir = os.path.join(tmpdir, f"swiglu_fusion_{swiglu_fusion}")
                    os.makedirs(model_dir)
                    model = self._build_int_qmoe_model(model_dir, [(4, 32, 1)], swiglu_fusion=swiglu_fusion)
                    with _capture_native_stderr() as captured:
                        sess = self._make_session(
                            model,
                            {
                                "session.moe.offload.mode": "required",
                                "session.moe.offload.placement": "static",
                                "session.moe.offload.popularity_file": "expert-profile.bin",
                                "session.moe.offload.cpu_execution": "disabled",
                                "session.moe.offload.device_expert_bytes": str(64 + 3 * 55304),
                                "session.moe.offload.host_expert_bytes": str(2 * 55808),
                                "session.moe.offload.prefill_wave_experts": "2",
                            },
                        )
                        actual = self._run_bf16(sess, x, router)
                        del sess
                        gc.collect()
                    self._assert_int_offload_parity(expected, actual, f"interleaved({swiglu_fusion=})")
                    # Parity alone would pass if the runner were skipped, so require the CUTLASS dispatch itself.
                    dispatched = re.findall(r"INT4 CUTLASS layer=0 routes=(\d+) capacity=(\d+)", captured[0])
                    self.assertTrue(dispatched, captured[0])
                    self.assertEqual(sum(int(routes) for routes, _ in dispatched), 2 * rows, captured[0])
            del ref
            gc.collect()

    def test_v4_int4_bounded_host_cache_warmup_runs_compute_cold_misses(self):
        self._skip_unless_int4_bf16_runner()
        warmup = ort.RunOptions()
        warmup.add_run_config_entry("ep.cuda.qmoe_in_warmup", "1")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1"}),
        ):
            model = self._build_int_qmoe_model(tmpdir, [(4, 32, 1)])
            ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            rng = np.random.default_rng(33627)
            # 4 rows take the asynchronous decode path and 16 rows the synchronous CPU path; both evaluate cold experts
            # through leases on the bounded host cache.
            for rows in (4, 16):
                with self.subTest(rows=rows):
                    with _capture_native_stderr() as captured:
                        sess = self._make_session(
                            model,
                            {
                                "session.moe.offload.mode": "required",
                                "session.moe.offload.placement": "static",
                                "session.moe.offload.popularity_file": "expert-profile.bin",
                                "session.moe.offload.cpu_execution": "enabled",
                                # Experts 0 and 3 stay resident and two host cache slots are fewer than the cold set.
                                "session.moe.offload.device_expert_bytes": str(64 + 2 * 55304),
                                "session.moe.offload.host_expert_bytes": str(2 * 55808),
                            },
                        )
                        for step, run_options in enumerate([warmup, warmup, None, warmup, None]):
                            x = rng.standard_normal((rows, 256), dtype=np.float32) * 0.5
                            router = np.full((rows, 8), -10.0, dtype=np.float32)
                            router[:, 0] = 10.0
                            router[:, 4 + step % 4] = 10.0  # Rotates through experts 4..7, so most runs miss the cache.
                            self._assert_int_offload_parity(
                                self._run_bf16(ref, x, router),
                                _run_qmoe_bf16(sess, x, router, run_options),
                                f"warmup_cold_miss({rows=},{step=},warmup={run_options is not None})",
                            )
                        del sess
                        gc.collect()
                    misses = re.findall(r"ssd_misses=(\d+)", captured[0])
                    self.assertTrue(misses and any(int(count) > 0 for count in misses), captured[0])
            del ref
            gc.collect()

    def test_v4_int4_many_experts_large_bounded_host_cache_is_initialized(self):
        # The backend must keep the finite host cache that admission granted, whatever the amount of system RAM.
        # A cache of at least 64 MiB for at least 64 experts used to be dropped on low-RAM Windows hosts.
        self._skip_unless_int4_bf16_runner()
        hidden, inter, num_experts = 1536, 640, 64
        device_slot_bytes = (hidden * inter * 27) // 16 + 8
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1"}),
        ):
            model = self._build_int_qmoe_model(
                tmpdir, [(4, 32, 1)], num_experts=num_experts, hidden_size=hidden, inter_size=inter
            )
            ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            rng = np.random.default_rng(33916)
            with _capture_native_stderr() as captured:
                sess = self._make_session(
                    model,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "static",
                        "session.moe.offload.popularity_file": "expert-profile.bin",
                        "session.moe.offload.cpu_execution": "enabled",
                        # Experts 0 and 3 stay resident. The host cache holds fewer than the 62 cold experts.
                        "session.moe.offload.device_expert_bytes": str(num_experts * 8 + 2 * device_slot_bytes + 4096),
                        "session.moe.offload.host_expert_bytes": str(80 << 20),
                    },
                )
                for step in range(5):
                    x = rng.standard_normal((4, hidden), dtype=np.float32) * 0.5
                    router = np.full((4, num_experts), -10.0, dtype=np.float32)
                    router[:, 0] = 10.0
                    router[:, 4 + step % 4] = 10.0
                    self._assert_int_offload_parity(
                        self._run_bf16(ref, x, router),
                        self._run_bf16(sess, x, router),
                        f"large_bounded_host_cache({step=})",
                    )
                del sess
                gc.collect()
            accesses = [
                int(hits) + int(misses)
                for hits, misses in re.findall(r"ssd_hits=(\d+) ssd_pf_hits=\d+ ssd_misses=(\d+)", captured[0])
            ]
            self.assertTrue(accesses and any(count > 0 for count in accesses), captured[0])
            del ref
            gc.collect()

    def test_v4_int4_decode_sized_warmup_with_many_experts_then_live_runs_match(self):
        # Decode-sized warm-up runs take the asynchronous path and, with at least 64 experts, arm the post-warm-up
        # working-set trim. The trim must not change results of the live runs that follow.
        self._skip_unless_int4_bf16_runner()
        warmup = ort.RunOptions()
        warmup.add_run_config_entry("ep.cuda.qmoe_in_warmup", "1")
        num_experts = 64
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._build_int_qmoe_model(tmpdir, [(4, 32, 1)], num_experts=num_experts)
            ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            sess = self._make_session(
                model,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                    "session.moe.offload.cpu_execution": "enabled",
                    "session.moe.offload.device_expert_bytes": str(num_experts * 8 + 2 * 55304),
                    "session.moe.offload.host_expert_bytes": str(2 * 55808),
                },
            )
            rng = np.random.default_rng(33935)
            for step, run_options in enumerate([warmup, warmup, None, None, warmup, None]):
                x = rng.standard_normal((4, 256), dtype=np.float32) * 0.5
                router = np.full((4, num_experts), -10.0, dtype=np.float32)
                router[:, 0] = 10.0
                router[:, 4 + step % 4] = 10.0
                self._assert_int_offload_parity(
                    self._run_bf16(ref, x, router),
                    _run_qmoe_bf16(sess, x, router, run_options),
                    f"decode_sized_warmup({step=},warmup={run_options is not None})",
                )
            del sess, ref
            gc.collect()

    def test_v4_int4_default_stream_tile_and_wave_fit_the_working_budget(self):
        self._skip_unless_int4_bf16_runner()
        rows = 2048
        offload_config = {
            "session.moe.offload.mode": "required",
            "session.moe.offload.placement": "static",
            "session.moe.offload.popularity_file": "expert-profile.bin",
            "session.moe.offload.cpu_execution": "enabled",
            "session.moe.offload.device_expert_bytes": str(64 + 2 * 55304),
            "session.moe.offload.host_expert_bytes": str(2 * 55808),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._build_int_qmoe_model(tmpdir, [(4, 32, 1)])
            ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            rng = np.random.default_rng(33628)
            x = rng.standard_normal((rows, 256), dtype=np.float32) * 0.5
            router = np.full((rows, 8), -10.0, dtype=np.float32)
            router[:, 4:6] = 10.0  # Both experts are cold and receive every row, so the whole call streams.
            expected = self._run_bf16(ref, x, router)

            def run_offload(working_bytes: int, extra_env: dict[str, str] | None = None) -> tuple[np.ndarray, str]:
                env = {"ORT_QMOE_SHARED_RESIDENT_LOG": "1", **(extra_env or {})}
                with mock.patch.dict(os.environ, env), _capture_native_stderr() as captured:
                    sess = self._make_session(
                        model,
                        {**offload_config, "session.moe.offload.device_working_bytes": str(working_bytes)},
                    )
                    try:
                        actual = self._run_bf16(sess, x, router)
                    finally:
                        del sess
                        gc.collect()
                return actual, captured[0]

            with self.subTest(case="default tile fits"):
                # The default 2,048-row tile and 8-expert wave need about 8 MiB, so a larger budget keeps them.
                actual, log_text = run_offload(16 << 20)
                self._assert_int_offload_parity(expected, actual, "stream_budget(fits)")
                self.assertNotIn("stream budget", log_text)
                self.assertEqual(_stream_tiles_per_layer(log_text), {0: 1}, log_text)

            with self.subTest(case="default tile shrinks"):
                # 3 MiB holds neither a 2,048-row nor a 1,024-row tile, but holds a 512-row tile with the 8-expert wave.
                actual, log_text = run_offload(3 << 20)
                self._assert_int_offload_parity(expected, actual, "stream_budget(shrunk)")
                self.assertRegex(log_text, r"stream budget layer=0 tile=2048->512 wave=8->8\b")
                self.assertEqual(_stream_tiles_per_layer(log_text), {0: rows // 512}, log_text)

            with (
                self.subTest(case="explicit tile is honored"),
                self.assertRaisesRegex(Exception, "exceeds configured GPU or pinned limits"),
            ):
                run_offload(3 << 20, {"ORT_QMOE_SHARED_STREAM_ROWS": "2048"})
            del ref
            gc.collect()

    @staticmethod
    def _hybrid_stream_inputs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """T=264 hybrid streaming prefill inputs: one hot resident expert plus rare and streamed cold experts."""
        x_long = rng.standard_normal((264, 256), dtype=np.float32) * 0.5
        r_long = np.full((264, 8), -20.0, dtype=np.float32)
        for row in range(264):
            r_long[row, 0] = 6.0  # hot/resident expert
            if row < 4:
                r_long[row, 6] = 5.0  # rare cold expert 6 (4 routes < 8 -> CPU evaluated)
            elif row < 7:
                r_long[row, 7] = 5.0  # rare cold expert 7 (3 routes < 8 -> CPU evaluated)
            else:
                # cold experts 2, 4, 5 each get ~85 routes (>= 8 -> streamed across 3 waves of wave_size=1)
                stream_exp = (2, 4, 5)[row % 3]
                r_long[row, stream_exp] = 5.0
        return x_long, r_long

    def _assert_cuda_graph_replay_parity(
        self, cudart, rng: np.random.Generator, label: str, ref_model, graph_model, device_expert_bytes: int
    ):
        """Replays a CUDA graph offload session in place, switching between resident hits and cold misses, against a
        non-offload reference. Both sessions are locals, so they are released before the caller removes its temp dir."""
        sess_ref = self._make_session(ref_model, {"session.moe.offload.mode": "disabled"})
        sess_graph = self._make_session(
            graph_model,
            {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "adaptive",
                "session.moe.offload.device_expert_bytes": str(device_expert_bytes),
                "session.moe.offload.popularity_file": "expert-profile.bin",
            },
            cuda_options={"enable_cuda_graph": "1"},
        )
        dev_id = self._device_id(self._selected_plugin_device())
        x_ptr = ctypes.c_void_p()
        r_ptr = ctypes.c_void_p()
        self.assertEqual(cudart.cudaMalloc(ctypes.byref(x_ptr), ctypes.c_size_t(2 * 256 * 4)), 0)
        self.assertEqual(cudart.cudaMalloc(ctypes.byref(r_ptr), ctypes.c_size_t(2 * 8 * 4)), 0)
        try:
            binding = sess_graph.io_binding()
            binding.bind_input("X_f32", "cuda", dev_id, np.float32, [2, 256], x_ptr.value)
            binding.bind_input("R_f32", "cuda", dev_id, np.float32, [2, 8], r_ptr.value)
            binding.bind_output("Y_f32", "cuda", dev_id)

            route_patterns = (
                (0, 3),  # resident hits on layer 0
                (5, 7),  # cold misses on layer 0
                (0, 7),  # mixed resident hit + cold miss
                (2, 5),  # cold misses
            )
            for replay_idx, (e_a, e_b) in enumerate(route_patterns):
                x_g = _bfloat16_u16_to_float32(
                    _float32_to_bfloat16_u16(rng.standard_normal((2, 256), dtype=np.float32) * 0.5)
                )
                r_g = np.full((2, 8), -20.0, dtype=np.float32)
                r_g[:, e_a] = 6.0
                r_g[:, e_b] = 5.0
                r_g = _bfloat16_u16_to_float32(_float32_to_bfloat16_u16(r_g))
                x_contig = np.ascontiguousarray(x_g, dtype=np.float32)
                r_contig = np.ascontiguousarray(r_g, dtype=np.float32)
                self.assertEqual(
                    cudart.cudaMemcpy(x_ptr, x_contig.ctypes.data_as(ctypes.c_void_p), x_contig.nbytes, 1),
                    0,
                )
                self.assertEqual(
                    cudart.cudaMemcpy(r_ptr, r_contig.ctypes.data_as(ctypes.c_void_p), r_contig.nbytes, 1),
                    0,
                )
                sess_graph.run_with_iobinding(binding)
                self._assert_int_offload_parity(
                    self._run_bf16(sess_ref, x_g, r_g),
                    binding.copy_outputs_to_cpu()[0],
                    f"{label}_cuda_graph_replay(step={replay_idx})",
                )
        finally:
            cudart.cudaFree(x_ptr)
            cudart.cudaFree(r_ptr)

    def _build_decode_optimization_model(
        self,
        model_dir: str,
        num_layers: int = 2,
        use_fp16: bool = True,
        num_experts: int = 16,
        resident_slots: int = 10,
        top_k: int = 8,
        normalize_routing_weights: int = 1,
    ):
        model_path = self._build_int_qmoe_model(
            model_dir,
            layer_specs=[(4, 32, 1)] * num_layers,
            num_experts=num_experts,
            hidden_size=256,
            inter_size=128,
            top_k=top_k,
            use_fp16=use_fp16,
            normalize_routing_weights=normalize_routing_weights,
        )
        pairs = [(layer, expert) for expert in range(num_experts) for layer in range(num_layers)]
        Path(model_dir, "expert-profile.bin").write_bytes(
            struct.pack("<6I", 0x50525453, 1, num_layers, num_experts, num_layers * num_experts, len(pairs))
            + b"".join(struct.pack("<2H", *pair) for pair in pairs)
        )
        config = {
            "session.moe.offload.mode": "required",
            "session.moe.offload.placement": "static",
            "session.moe.offload.device_expert_bytes": str(num_layers * (2 * num_experts * 4 + resident_slots * 55304)),
            "session.moe.offload.popularity_file": "expert-profile.bin",
            "session.moe.offload.cpu_threads": "4",
        }
        return model_path, config

    def _run_decode_with_telemetry(
        self, model_path: str, config: dict, cases: list, use_fp16: bool = True, cuda_graph: bool = False
    ):
        with tempfile.TemporaryFile(mode="w+b") as capture:
            saved_stderr = os.dup(2)
            try:
                os.dup2(capture.fileno(), 2)
                with mock.patch.dict(os.environ, {"ORT_QMOE_OFFLOAD_TELEMETRY": "1"}):
                    options = {"enable_cuda_graph": "1"} if cuda_graph else None
                    session = self._make_session(model_path, config, options)
                    self.assertEqual(session.get_providers()[0], "CUDAExecutionProvider")
                    if cuda_graph:
                        self.assertTrue(use_fp16)
                        cudart = _load_cudart()
                        if cudart is None:
                            self.skipTest("CUDA runtime library is required to update captured GPU inputs in place")
                        device_id = self._device_id(self._selected_plugin_device())
                        # CUDA OrtValue allocation uses the legacy provider interface, absent in plugin-only builds.
                        with contextlib.ExitStack() as buffers:
                            x_ptr = _cuda_malloc(cudart, cases[0][0].size * np.dtype(np.float16).itemsize, device_id)
                            buffers.callback(cudart.cudaFree, x_ptr)
                            r_ptr = _cuda_malloc(cudart, cases[0][1].size * np.dtype(np.float16).itemsize, device_id)
                            buffers.callback(cudart.cudaFree, r_ptr)
                            binding = session.io_binding()
                            binding.bind_input("X", "cuda", device_id, np.float16, cases[0][0].shape, x_ptr.value)
                            binding.bind_input(
                                "router_probs", "cuda", device_id, np.float16, cases[0][1].shape, r_ptr.value
                            )
                            binding.bind_output("Y", "cuda", device_id)
                            outputs = []
                            for x, router in cases:
                                _cuda_upload(cudart, x_ptr, x.astype(np.float16))
                                _cuda_upload(cudart, r_ptr, router.astype(np.float16))
                                session.run_with_iobinding(binding)
                                outputs.append(binding.copy_outputs_to_cpu()[0].astype(np.float32))
                            del binding
                    else:
                        run = self._run_fp16 if use_fp16 else self._run_bf16
                        outputs = [run(session, x, router) for x, router in cases]
                    del session
            finally:
                if sys.exc_info()[0] is not None:
                    capture.seek(0)
                    os.write(saved_stderr, capture.read())
                os.dup2(saved_stderr, 2)
                os.close(saved_stderr)
            capture.seek(0)
            native_log = capture.read().decode("utf-8")
        prefix = "QMoEOffload telemetry "
        telemetry = [json.loads(line.split(prefix, 1)[1]) for line in native_log.splitlines() if prefix in line]
        self.assertTrue(telemetry, native_log)
        return outputs, telemetry

    def test_v4_zero_cold_doorbell_bitwise_parity_and_transitions(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_path, config = self._build_decode_optimization_model(model_dir)
            rng = np.random.default_rng(33575)
            cases = []
            for rows, cold in ((1, False), (4, False), (1, True), (16, True), (2, False), (1, True), (1, False)):
                x = rng.standard_normal((rows, 256), dtype=np.float32) * 0.25
                router = np.tile(np.linspace(3.0, -3.0, 16, dtype=np.float32), (rows, 1))
                if cold:
                    router[:, 12:14] = 4.0
                cases.append((x, router))
            baseline, baseline_stats = self._run_decode_with_telemetry(model_path, config, cases)
            optimized, optimized_stats = self._run_decode_with_telemetry(
                model_path, {**config, "session.moe.offload.skip_zero_cold_doorbell": "1"}, cases
            )
            for original, actual in zip(baseline, optimized, strict=True):
                np.testing.assert_array_equal(original.view(np.uint32), actual.view(np.uint32))
            self.assertGreater(sum(item["zero_cold_short_circuits"] for item in optimized_stats), 0)
            self.assertGreater(sum(item["cpu_doorbell_wakeups"] for item in optimized_stats), 0)
            self.assertGreater(sum(item["cold_routes"] for item in optimized_stats), 0)
            self.assertEqual(sum(item["zero_cold_short_circuits"] for item in baseline_stats), 0)
            for item in optimized_stats:
                self.assertEqual(item["done_seq"], item["launched_seq"])
                self.assertEqual(item["done_seq"], item["request_seq"])
                self.assertEqual(item["done_seq"], item["last_serviced_seq"])
            with self.subTest("adaptive observations retain CPU service"):
                _, adaptive_stats = self._run_decode_with_telemetry(
                    model_path,
                    {
                        **config,
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.skip_zero_cold_doorbell": "1",
                    },
                    [cases[0], cases[2], cases[0]],
                )
                self.assertEqual(sum(item["zero_cold_short_circuits"] for item in adaptive_stats), 0)
                self.assertGreater(sum(item["cpu_doorbell_wakeups"] for item in adaptive_stats), 0)

    def test_v4_zero_cold_doorbell_cuda_graph_replay(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_path, config = self._build_decode_optimization_model(model_dir, num_layers=4)
            rng = np.random.default_rng(33577)
            cases = []
            for step in range(128):
                x = rng.standard_normal((1, 256), dtype=np.float32) * 0.25
                router = np.linspace(3.0, -3.0, 16, dtype=np.float32).reshape(1, 16)
                if step % 3 == 1:
                    router[:, 12:14] = 4.0
                cases.append((x, router))
            baseline, _ = self._run_decode_with_telemetry(model_path, config, cases)
            optimized, stats = self._run_decode_with_telemetry(
                model_path, {**config, "session.moe.offload.skip_zero_cold_doorbell": "1"}, cases, cuda_graph=True
            )
            for original, actual in zip(baseline, optimized, strict=True):
                np.testing.assert_array_equal(original.view(np.uint32), actual.view(np.uint32))
            for item in stats:
                self.assertGreater(item["zero_cold_short_circuits"], 0)
                self.assertGreater(item["cpu_doorbell_wakeups"], 0)
                self.assertEqual(item["done_seq"], item["request_seq"])
                self.assertEqual(item["done_seq"], item["launched_seq"])

    def test_v4_zero_cold_doorbell_retains_cross_layer_promotion_service(self):
        script = """
import sys
import tempfile
import numpy as np
sys.path.insert(0, sys.argv[1])
from test_qmoe_offload_v4 import TestQMoEOffloadV4
TestQMoEOffloadV4.setUpClass()
case = TestQMoEOffloadV4()
with tempfile.TemporaryDirectory() as directory:
    model, config = case._build_decode_optimization_model(directory, num_layers=1)
    x = np.full((1, 256), 0.125, dtype=np.float32)
    router = np.linspace(3.0, -3.0, 16, dtype=np.float32).reshape(1, 16)
    baseline, _ = case._run_decode_with_telemetry(model, config, [(x, router)])
    actual, stats = case._run_decode_with_telemetry(
        model, {**config, "session.moe.offload.skip_zero_cold_doorbell": "1"}, [(x, router)]
    )
    np.testing.assert_array_equal(baseline[0].view(np.uint32), actual[0].view(np.uint32))
    assert stats[0]["zero_cold_short_circuits"] == 0, stats
    assert stats[0]["cpu_doorbell_wakeups"] == 1, stats
"""
        environment = dict(os.environ, ORT_QMOE_CROSS_LAYER_PROMO="1")
        environment.pop("ORT_QMOE_SHARED_EXPERT_BRIDGE_PTR", None)
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve().parent)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_v4_tail_substitution_oracle_unique_routes_and_normalized_weights(self):
        for use_fp16 in (True, False):
            if not use_fp16 and not self._int4_bf16_runner_supported:
                continue
            with self.subTest(use_fp16=use_fp16), tempfile.TemporaryDirectory() as model_dir:
                model_path, config = self._build_decode_optimization_model(model_dir, use_fp16=use_fp16)
                rng = np.random.default_rng(33576)
                x = rng.standard_normal((4, 256), dtype=np.float32) * 0.25
                router = np.tile(
                    np.array([3, 2.8, 2.6, 2.4, 2.2, 2, 0.99, 0.98, -1, -2, 1.1, 1, -5, -5, -5, -5], dtype=np.float32),
                    (4, 1),
                )
                baseline, _ = self._run_decode_with_telemetry(model_path, config, [(x, router)], use_fp16)
                disabled, disabled_stats = self._run_decode_with_telemetry(
                    model_path,
                    {**config, "session.moe.offload.tail_substitution_threshold": "0.0"},
                    [(x, router)],
                    use_fp16,
                )
                np.testing.assert_array_equal(baseline[0].view(np.uint32), disabled[0].view(np.uint32))
                self.assertEqual(sum(item["substituted_routes_count"] for item in disabled_stats), 0)
                substituted, stats = self._run_decode_with_telemetry(
                    model_path,
                    {
                        **config,
                        "session.moe.offload.skip_zero_cold_doorbell": "1",
                        "session.moe.offload.tail_substitution_threshold": "0.01",
                        "session.moe.offload.tail_substitution_min_rank": "6",
                    },
                    [(x, router)],
                    use_fp16,
                )
                rounded_router = (
                    router.astype(np.float16).astype(np.float32)
                    if use_fp16
                    else _bfloat16_u16_to_float32(_float32_to_bfloat16_u16(router))
                )
                repl_logits = rounded_router[:, 6:8]
                repl_exp = np.exp(repl_logits - np.max(repl_logits, axis=1, keepdims=True))
                repl_props = repl_exp / np.sum(repl_exp, axis=1, keepdims=True)
                orig_tail_logits = rounded_router[:, 10:12]
                orig_tail_sum = np.sum(np.exp(orig_tail_logits), axis=1, keepdims=True)
                repl_tail_sum = np.sum(np.exp(repl_logits), axis=1, keepdims=True)
                oracle_router = rounded_router.copy()
                oracle_router[:, 6:8] = repl_logits + np.log(orig_tail_sum / repl_tail_sum)
                oracle_router[:, 10:12] = -20.0
                oracle, _ = self._run_decode_with_telemetry(model_path, config, [(x, oracle_router)], use_fp16)
                self._assert_int_offload_parity(oracle[0], substituted[0], f"tail_substitution_oracle_fp16_{use_fp16}")
                self.assertGreater(float(np.max(np.abs(substituted[0] - disabled[0]))), 5.0)
                for disabled_item, item in zip(disabled_stats, stats, strict=True):
                    disabled_weights = np.asarray(disabled_item["last_weights"], dtype=np.float32).reshape(4, 8)
                    indices = np.asarray(item["last_indices"]).reshape(4, 8)
                    weights = np.asarray(item["last_weights"], dtype=np.float32).reshape(4, 8)
                    np.testing.assert_array_equal(indices, np.tile(np.arange(8), (4, 1)))
                    self.assertTrue(all(len(set(row)) == 8 for row in indices))
                    np.testing.assert_array_equal(
                        weights[:, :6].view(np.uint32),
                        disabled_weights[:, :6].view(np.uint32),
                    )
                    w_sub = np.sum(disabled_weights[:, 6:8], axis=1, keepdims=True)
                    expected_tail_weights = w_sub * repl_props
                    np.testing.assert_allclose(weights[:, 6:8], expected_tail_weights, rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0, atol=2e-7)
                    self.assertEqual(item["last_cold_count"], 0)
                    self.assertEqual(item["substituted_routes_count"], 8)
                    self.assertEqual(item["zero_cold_conversions_count"], 4)
                    self.assertEqual(item["zero_cold_short_circuits"], 1)
                    self.assertEqual(item["cpu_doorbell_wakeups"], 0)

    def test_v4_tail_substitution_preserves_pruning_scales_without_topk_normalization(self):
        for use_fp16 in (True, False):
            if not use_fp16 and not self._int4_bf16_runner_supported:
                continue
            with self.subTest(use_fp16=use_fp16), tempfile.TemporaryDirectory() as model_dir:
                model_path, config = self._build_decode_optimization_model(
                    model_dir,
                    num_layers=1,
                    use_fp16=use_fp16,
                    num_experts=5,
                    resident_slots=2,
                    top_k=4,
                    normalize_routing_weights=0,
                )
                Path(model_dir, "expert-profile.bin").write_bytes(
                    struct.pack("<6I", 0x50525453, 1, 1, 5, 5, 5)
                    + b"".join(struct.pack("<2H", 0, expert) for expert in (0, 2, 1, 3, 4))
                )
                config.update(
                    {
                        "session.moe.offload.skip_tail_cold_only": "0",
                        "session.moe.offload.min_expert_weight": "0.21",
                        "session.moe.offload.tail_substitution_min_rank": "1",
                    }
                )
                x = np.full((1, 256), 0.125, dtype=np.float32)
                # Pruning frees resident expert 2 to replace the nearby cold expert 1.
                router = np.log(np.array([[0.5, 0.211, 0.209, 0.04, 0.04]], dtype=np.float32))
                rounded_router = _round_to_act_dtype(router, use_fp16)
                probabilities = np.exp(rounded_router - np.max(rounded_router, axis=1, keepdims=True))
                probabilities /= np.sum(probabilities, axis=1, keepdims=True)
                for renorm_after_prune in (0, 1):
                    with (
                        self.subTest(renorm_after_prune=renorm_after_prune),
                        mock.patch.dict(
                            os.environ,
                            {"ORT_QMOE_DOORBELL": "1", "ORT_QMOE_RENORM_AFTER_PRUNE": str(renorm_after_prune)},
                        ),
                    ):
                        _, baseline_stats = self._run_decode_with_telemetry(
                            model_path,
                            {**config, "session.moe.offload.tail_substitution_threshold": "0"},
                            [(x, router)],
                            use_fp16,
                        )
                        _, stats = self._run_decode_with_telemetry(
                            model_path,
                            {**config, "session.moe.offload.tail_substitution_threshold": "0.004"},
                            [(x, router)],
                            use_fp16,
                        )
                        self.assertEqual(len(baseline_stats), 1)
                        self.assertEqual(len(stats), 1)
                        np.testing.assert_array_equal(baseline_stats[0]["last_indices"], [0, 1, -2, -2])
                        np.testing.assert_array_equal(stats[0]["last_indices"], [0, 2, -2, -2])
                        baseline_weights = np.asarray(baseline_stats[0]["last_weights"], dtype=np.float32)
                        weights = np.asarray(stats[0]["last_weights"], dtype=np.float32)
                        expected = np.zeros(4, dtype=np.float32)
                        expected[:2] = probabilities[0, :2]
                        if renorm_after_prune:
                            expected /= np.sum(expected)
                        np.testing.assert_allclose(baseline_weights, expected, rtol=1e-6, atol=1e-7)
                        np.testing.assert_array_equal(
                            weights[[0, 2, 3]].view(np.uint32), baseline_weights[[0, 2, 3]].view(np.uint32)
                        )
                        expected[1] *= np.exp(rounded_router[0, 2] - rounded_router[0, 1])
                        np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=1e-7)
                        self.assertEqual(stats[0]["substituted_routes_count"], 1)
                        self.assertEqual(stats[0]["last_cold_count"], 0)

    def test_v4_tail_substitution_covers_speculative_verify_tiles(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_path, config = self._build_decode_optimization_model(model_dir, use_fp16=True)
            rows = 8
            rng = np.random.default_rng(33577)
            x = rng.standard_normal((rows, 256), dtype=np.float32) * 0.25
            router = np.tile(
                np.array([3, 2.8, 2.6, 2.4, 2.2, 2, 0.99, 0.98, -1, -2, 1.1, 1, -5, -5, -5, -5], dtype=np.float32),
                (rows, 1),
            )
            _, disabled_stats = self._run_decode_with_telemetry(model_path, config, [(x, router)], True)
            _, stats = self._run_decode_with_telemetry(
                model_path,
                {
                    **config,
                    "session.moe.offload.skip_zero_cold_doorbell": "1",
                    "session.moe.offload.tail_substitution_threshold": "0.01",
                    "session.moe.offload.tail_substitution_min_rank": "6",
                },
                [(x, router)],
                True,
            )
            for disabled_item, item in zip(disabled_stats, stats, strict=True):
                indices = np.asarray(item["last_indices"]).reshape(rows, 8)
                np.testing.assert_array_equal(indices, np.tile(np.arange(8), (rows, 1)))
                weights = np.asarray(item["last_weights"], dtype=np.float32).reshape(rows, 8)
                np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0, atol=2e-7)
                self.assertEqual(disabled_item["substituted_routes_count"], 0)
                self.assertEqual(item["substituted_routes_count"], 2 * rows)
                self.assertEqual(item["zero_cold_conversions_count"], rows)

    def test_v4_tail_substitution_keeps_cold_routes_above_the_weight_cap(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_path, config = self._build_decode_optimization_model(model_dir, use_fp16=True)
            rows = 2
            x = np.random.default_rng(33944).standard_normal((rows, 256), dtype=np.float32) * 0.25
            # Eight equally weighted routes (0.125 each, above the 0.05 cap): experts 0, 1 and the cold experts 10..15.
            # The unselected resident experts are almost as likely, so both gap checks pass for a 0.001 threshold.
            router = np.full((rows, 16), 0.995, dtype=np.float32)
            router[:, [0, 1, 10, 11, 12, 13, 14, 15]] = 1.0
            _, stats = self._run_decode_with_telemetry(
                model_path,
                {
                    **config,
                    "session.moe.offload.tail_substitution_threshold": "0.001",
                    "session.moe.offload.tail_substitution_min_rank": "6",
                },
                [(x, router)],
                True,
            )
            for item in stats:
                indices = np.asarray(item["last_indices"]).reshape(rows, 8)
                for row in indices:
                    self.assertEqual(sorted(row.tolist()), [0, 1, 10, 11, 12, 13, 14, 15])
                self.assertEqual(item["substituted_routes_count"], 0)

    def test_v4_tail_substitution_stable_normalization_after_replacing_every_route(self):
        for use_fp16 in (True, False):
            if not use_fp16 and not self._int4_bf16_runner_supported:
                continue
            with self.subTest(use_fp16=use_fp16), tempfile.TemporaryDirectory() as model_dir:
                model_path, config = self._build_decode_optimization_model(
                    model_dir, num_layers=1, use_fp16=use_fp16, num_experts=128, resident_slots=32
                )
                x = np.full((1, 256), 0.125, dtype=np.float32)
                router = np.zeros((1, 128), dtype=np.float32)
                router[:, :32] = -1000
                actual, stats = self._run_decode_with_telemetry(
                    model_path,
                    {
                        **config,
                        "session.moe.offload.skip_zero_cold_doorbell": "1",
                        "session.moe.offload.tail_substitution_threshold": "0.02",
                        "session.moe.offload.tail_substitution_min_rank": "0",
                    },
                    [(x, router)],
                    use_fp16,
                )
                np.testing.assert_array_equal(stats[0]["last_indices"], np.arange(8))
                np.testing.assert_allclose(stats[0]["last_weights"], np.full(8, 0.125), rtol=0, atol=1e-7)
                self.assertEqual(stats[0]["substituted_routes_count"], 8)
                self.assertEqual(stats[0]["last_cold_count"], 0)
                oracle_router = np.full((1, 128), -1000, dtype=np.float32)
                oracle_router[:, :8] = 0
                oracle, _ = self._run_decode_with_telemetry(model_path, config, [(x, oracle_router)], use_fp16)
                np.testing.assert_array_equal(actual[0].view(np.uint32), oracle[0].view(np.uint32))

    def test_v4_tail_substitution_gap_and_rank_guards(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_path, config = self._build_decode_optimization_model(model_dir, num_layers=1)
            x = np.full((1, 256), 0.125, dtype=np.float32)
            router = np.array(
                [[3, 2.8, 2.6, 2.4, 2.2, 2, 0.99, 0.98, -1, -2, 1.1, 1, -5, -5, -5, -5]],
                dtype=np.float32,
            )
            baseline, _ = self._run_decode_with_telemetry(model_path, config, [(x, router)])
            rejected, stats = self._run_decode_with_telemetry(
                model_path,
                {**config, "session.moe.offload.tail_substitution_threshold": "0.000001"},
                [(x, router)],
            )
            np.testing.assert_array_equal(baseline[0].view(np.uint32), rejected[0].view(np.uint32))
            self.assertEqual(stats[0]["substituted_routes_count"], 0)
            self.assertEqual(stats[0]["last_cold_count"], 2)
            self.assertEqual(stats[0]["cpu_doorbell_wakeups"], 1)
            _, rank_stats = self._run_decode_with_telemetry(
                model_path,
                {
                    **config,
                    "session.moe.offload.tail_substitution_threshold": "0.01",
                    "session.moe.offload.tail_substitution_min_rank": "7",
                },
                [(x, router)],
            )
            self.assertEqual(rank_stats[0]["last_indices"][6], 10)
            self.assertEqual(rank_stats[0]["last_indices"][7], 6)
            self.assertEqual(rank_stats[0]["last_cold_count"], 1)
        with tempfile.TemporaryDirectory() as model_dir:
            model_path, config = self._build_decode_optimization_model(
                model_dir, num_layers=1, num_experts=64, resident_slots=32, top_k=40
            )
            x = np.full((1, 256), 0.125, dtype=np.float32)
            router = np.linspace(4.0, -4.0, 64, dtype=np.float32).reshape(1, 64)
            baseline, _ = self._run_decode_with_telemetry(model_path, config, [(x, router)])
            actual, k40_stats = self._run_decode_with_telemetry(
                model_path,
                {
                    **config,
                    "session.moe.offload.tail_substitution_threshold": "0.02",
                    "session.moe.offload.tail_substitution_min_rank": "6",
                },
                [(x, router)],
            )
            np.testing.assert_array_equal(baseline[0].view(np.uint32), actual[0].view(np.uint32))
            self.assertEqual(k40_stats[0]["substituted_routes_count"], 0)

    def test_v4_decode_optimization_policy_validation_and_env_precedence(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_path, config = self._build_decode_optimization_model(model_dir, num_layers=1)
            prefix = "session.moe.offload."
            for key, value in (
                ("skip_zero_cold_doorbell", "2"),
                ("tail_substitution_threshold", "-0.1"),
                ("tail_substitution_threshold", "nan"),
                ("tail_substitution_threshold", "inf"),
                ("tail_substitution_threshold", "0.01junk"),
                ("tail_substitution_min_rank", "-1"),
            ):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(Exception, "Invalid " + prefix + key):
                    self._make_session(model_path, {**config, prefix + key: value})
            with self.assertRaisesRegex(Exception, "min_rank < k"):
                self._make_session(
                    model_path,
                    {
                        **config,
                        prefix + "tail_substitution_threshold": "0.01",
                        prefix + "tail_substitution_min_rank": "8",
                    },
                )
            # In mode=auto, a candidate QMoE node with k <= tail_substitution_min_rank that is rejected from offload
            # because its expert initializer is shared with a non-offload consumer must not fail session creation.
            shared_proto = onnx.load_model(model_path, load_external_data=False)
            for attr in shared_proto.graph.node[0].attribute:
                if attr.name == "k":
                    attr.i = 4
            shared_proto.graph.node.append(
                helper.make_node(
                    "Identity",
                    ["model.layers.0.mlp.fc1_experts_weights"],
                    ["shared_w1_tap"],
                    name="NonOffloadInitializerConsumer",
                )
            )
            shared_path = os.path.join(model_dir, "model_shared_init_k4.onnx")
            onnx.save_model(shared_proto, shared_path)
            with self.assertRaisesRegex(Exception, "shared with a non-offload consumer"):
                self._make_session(
                    shared_path,
                    {
                        **config,
                        prefix + "tail_substitution_threshold": "0.01",
                        prefix + "tail_substitution_min_rank": "6",
                    },
                )
            auto_shared_session = self._make_session(
                shared_path,
                {
                    **config,
                    "session.moe.offload.mode": "auto",
                    prefix + "tail_substitution_threshold": "0.01",
                    prefix + "tail_substitution_min_rank": "6",
                },
            )
            del auto_shared_session
            x = np.full((1, 256), 0.125, dtype=np.float32)
            router = np.linspace(3.0, -3.0, 16, dtype=np.float32).reshape(1, 16)
            with mock.patch.dict(os.environ, {"ORT_QMOE_SKIP_ZERO_COLD_DOORBELL": "1"}):
                _, enabled = self._run_decode_with_telemetry(model_path, config, [(x, router)])
                _, explicit_zero = self._run_decode_with_telemetry(
                    model_path, {**config, prefix + "skip_zero_cold_doorbell": "0"}, [(x, router)]
                )
            self.assertEqual(enabled[0]["zero_cold_short_circuits"], 1)
            self.assertEqual(explicit_zero[0]["zero_cold_short_circuits"], 0)
            with mock.patch.dict(
                os.environ,
                {
                    "ORT_QMOE_TAIL_SUBSTITUTION_THRESHOLD": "invalid",
                    "ORT_QMOE_TAIL_SUBSTITUTION_MIN_RANK": "invalid",
                },
            ):
                with self.assertRaisesRegex(Exception, "tail_substitution_threshold"):
                    self._make_session(model_path, config)
                session = self._make_session(
                    model_path,
                    {**config, prefix + "tail_substitution_threshold": "0", prefix + "tail_substitution_min_rank": "0"},
                )
                del session

    def test_v4_min_expert_weight_env_override_and_callback_doorbell_equivalence(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_path = self._build_int_qmoe_model(
                model_dir,
                layer_specs=[(2, 64, 1)],
                num_experts=8,
                hidden_size=256,
                inter_size=128,
                top_k=4,
                use_fp16=False,
            )
            slot_bytes = 27656  # H=256, I=128, INT2 b64
            fixed_bytes = 2 * 8 * 4
            base_cfg = {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "static",
                "session.moe.offload.popularity_file": "expert-profile.bin",
                "session.moe.offload.device_expert_bytes": str(fixed_bytes + 2 * slot_bytes),
                "session.moe.offload.cpu_threads": "2",
            }
            rng = np.random.default_rng(33916)
            x = _round_to_act_dtype(rng.standard_normal((2, 256), dtype=np.float32) * 0.5)
            # Normalized top-4 softmax probabilities for [4.0, 2.5, 1.0, 0.5]:
            # rank 0 (expert 0, resident) ~ 0.775, rank 1 (expert 3, resident) ~ 0.173,
            # rank 2 (expert 4, cold) ~ 0.038, rank 3 (expert 5, cold) ~ 0.023.
            router = np.full((2, 8), -20.0, dtype=np.float32)
            router[:, 0] = 4.0
            router[:, 3] = 2.5
            router[:, 4] = 1.0
            router[:, 5] = 0.5

            # 1. Explicit session.moe.offload.min_expert_weight="0" must disable pruning even when
            # ORT_QMOE_MIN_EXPERT_WEIGHT="0.10" is set in the process environment.
            unpruned_out, unpruned_stats = self._run_decode_with_telemetry(
                model_path, {**base_cfg, "session.moe.offload.min_expert_weight": "0"}, [(x, router)], use_fp16=False
            )
            self.assertEqual(unpruned_stats[0]["cold_routes"], 4)
            with mock.patch.dict(os.environ, {"ORT_QMOE_MIN_EXPERT_WEIGHT": "0.10"}):
                env_pruned_out, env_pruned_stats = self._run_decode_with_telemetry(
                    model_path, base_cfg, [(x, router)], use_fp16=False
                )
                opt_out_out, opt_out_stats = self._run_decode_with_telemetry(
                    model_path,
                    {**base_cfg, "session.moe.offload.min_expert_weight": "0"},
                    [(x, router)],
                    use_fp16=False,
                )
            self.assertEqual(env_pruned_stats[0]["cold_routes"], 0)
            self.assertEqual(opt_out_stats[0]["cold_routes"], 4)
            np.testing.assert_array_equal(unpruned_out[0], opt_out_out[0])
            self.assertFalse(np.allclose(unpruned_out[0], env_pruned_out[0], atol=1e-3))

            # 2. Callback decode (ORT_QMOE_DOORBELL=0) and Doorbell decode (ORT_QMOE_DOORBELL=1) must produce
            # identical cold-only pruned outputs and identical CPU cold route counts, including when resident
            # tail experts (rank 1 = expert 3, weight ~ 0.173 < 0.25) are retained while cold tail experts
            # (ranks 2, 3 = experts 4, 5, weights < 0.25) are pruned.
            prune_cfg = {
                **base_cfg,
                "session.moe.offload.min_expert_weight": "0.25",
                "session.moe.offload.skip_tail_cold_only": "1",
            }
            with (
                mock.patch.dict(os.environ, {"ORT_QMOE_DOORBELL": "1", "ORT_QMOE_SHARED_CPU_TIMING": "1"}),
                _capture_native_stderr() as db_log,
            ):
                sess_db = self._make_session(model_path, prune_cfg)
                out_db = self._run_bf16(sess_db, x, router)
                del sess_db
                gc.collect()
            with (
                mock.patch.dict(os.environ, {"ORT_QMOE_DOORBELL": "0", "ORT_QMOE_SHARED_CPU_TIMING": "1"}),
                _capture_native_stderr() as cb_log,
            ):
                sess_cb = self._make_session(model_path, prune_cfg)
                out_cb = self._run_bf16(sess_cb, x, router)
                del sess_cb
                gc.collect()
            np.testing.assert_allclose(out_db, out_cb, rtol=0, atol=0)
            self.assertEqual(_cpu_path_route_rows(db_log[0])[0][2], 0, db_log[0])
            self.assertEqual(_cpu_path_route_rows(cb_log[0])[0][2], 0, cb_log[0])
            self.assertFalse(np.allclose(out_db, unpruned_out[0], atol=1e-3))

            # 3. All-cold decode (device_expert_bytes="0", 0 resident slots) must also prune cold tail experts.
            all_cold_cfg = {
                **prune_cfg,
                "session.moe.offload.device_expert_bytes": "0",
                "session.moe.offload.min_expert_weight": "0.10",
            }
            all_cold_skip_all_cfg = {
                **all_cold_cfg,
                "session.moe.offload.skip_tail_cold_only": "0",
            }
            with (
                mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_CPU_TIMING": "1"}),
                _capture_native_stderr() as ac_log,
            ):
                sess_ac = self._make_session(model_path, all_cold_cfg)
                out_ac = self._run_bf16(sess_ac, x, router)
                del sess_ac
                gc.collect()
            sess_ac_all = self._make_session(model_path, all_cold_skip_all_cfg)
            out_ac_all = self._run_bf16(sess_ac_all, x, router)
            del sess_ac_all
            gc.collect()
            np.testing.assert_allclose(out_ac, out_ac_all, rtol=0, atol=0)
            # Only ranks 0 and 1 (2 routes/token * 2 tokens = 4 routes) survive the 0.10 threshold in all-cold mode.
            self.assertEqual(_cpu_path_route_rows(ac_log[0])[0][2], 4, ac_log[0])

    def test_v4_int4_and_int3_bf16_int4_prepacked_offload_parity(self):
        # Quick builds (ORT_QUICK_BUILD) do not compile the native BF16 INT4 QMoE runner that every session here needs.
        self._skip_unless_int4_bf16_runner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Prepacked INT4 (bits=4, block_size=32, weights_prepacked=1)
            dir_int4 = os.path.join(tmpdir, "int4_b32")
            os.makedirs(dir_int4, exist_ok=True)
            m_int4 = self._build_int_qmoe_model(
                dir_int4, layer_specs=[(4, 32, 1), (4, 32, 1)], num_experts=8, hidden_size=256, inter_size=128, top_k=2
            )
            # H=256, I=128, INT4 b32: 32768 + 16384 + 4096 + 2048 + 8 = 55,304 B gpu slot; 55,808 B host stride
            int4_slot_bytes = 55304
            int4_host_stride = 55808
            fixed_bytes = 2 * 8 * 4  # 64 B per layer
            budget_int4_adaptive = 2 * fixed_bytes + 12 * int4_slot_bytes  # 6 slots/layer (4 resident + 2 spare)

            sess_int4_ref = self._make_session(m_int4, {"session.moe.offload.mode": "disabled"})
            sess_int4_static = self._make_session(
                m_int4,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(2 * fixed_bytes + 6 * int4_slot_bytes),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )
            sess_int4_adaptive = self._make_session(
                m_int4,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_int4_adaptive),
                    "session.moe.offload.host_expert_bytes": str(2 * 2 * int4_host_stride),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                    "session.moe.offload.prefill_wave_experts": "1",
                },
            )
            sess_int4_stream = self._make_session(
                m_int4,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.cpu_execution": "disabled",
                    "session.moe.offload.device_expert_bytes": str(2 * fixed_bytes + 6 * int4_slot_bytes),
                    "session.moe.offload.prefill_wave_experts": "1",
                },
            )

            rng = np.random.default_rng(202610)
            for num_rows in (1, 2, 4, 8, 16, 64, 256):
                x = rng.standard_normal((num_rows, 256), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 8), dtype=np.float32)
                y_ref = self._run_bf16(sess_int4_ref, x, r)
                for name, s in (
                    ("int4_static", sess_int4_static),
                    ("int4_adaptive_bounded_host", sess_int4_adaptive),
                    ("int4_gpu_stream_wave1", sess_int4_stream),
                ):
                    y_cand = self._run_bf16(s, x, r)
                    self._assert_int_offload_parity(y_ref, y_cand, f"{name}(T={num_rows})")

            # CPU-enabled hybrid streaming prefill beyond the 257-token threshold (T=264) with bounded host cache,
            # multiple 1-expert waves, streamed cold experts (>= 8 routes), and rare CPU cold experts (< 8 routes).
            x_long, r_long = self._hybrid_stream_inputs(rng)
            y_ref_long = self._run_bf16(sess_int4_ref, x_long, r_long)
            y_adapt_long = self._run_bf16(sess_int4_adaptive, x_long, r_long)
            self._assert_int_offload_parity(y_ref_long, y_adapt_long, "int4_hybrid_stream_bounded_host(T=264)")
            del sess_int4_stream, sess_int4_adaptive, sess_int4_static, sess_int4_ref

            # Explicit ORT_QMOE_STREAM_MIN_TOKENS override must be preserved on integer banks, and a 2-token doorbell
            # decode followed by repeated 8-token doorbell runs must still advance promotions. The callback branch
            # after a doorbell run is covered by test_v4_int4_callback_service_runs_after_a_doorbell_run.
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "ORT_QMOE_STREAM_MIN_TOKENS": "12",
                        "ORT_QMOE_SHARED_RESIDENT_LOG": "1",
                    },
                ),
                _capture_native_stderr() as int4_log_cap,
            ):
                sess_int4_promo = self._make_session(
                    m_int4,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.device_expert_bytes": str(budget_int4_adaptive),
                        "session.moe.offload.popularity_file": "expert-profile.bin",
                    },
                )
                sess_int4_ref2 = self._make_session(m_int4, {"session.moe.offload.mode": "disabled"})
                # 16-token run >= explicit ORT_QMOE_STREAM_MIN_TOKENS=12 must execute hybrid streaming
                x_s16 = rng.standard_normal((16, 256), dtype=np.float32) * 0.5
                r_s16 = np.full((16, 8), -20.0, dtype=np.float32)
                r_s16[:, 5] = 6.0
                r_s16[:, 7] = 5.0
                self._assert_int_offload_parity(
                    self._run_bf16(sess_int4_ref2, x_s16, r_s16),
                    self._run_bf16(sess_int4_promo, x_s16, r_s16),
                    "int4_explicit_stream_min_tokens(T=16)",
                )
                del sess_int4_promo
                gc.collect()
            self.assertEqual(sorted(_stream_tiles_per_layer(int4_log_cap[0])), [0, 1])
            for layer_id, tile_cnt in _stream_tiles_per_layer(int4_log_cap[0]).items():
                self.assertGreater(
                    tile_cnt, 0, f"layer {layer_id} must stream at T=16 when ORT_QMOE_STREAM_MIN_TOKENS=12"
                )

            with (
                mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1"}),
                _capture_native_stderr() as seq_promo_cap,
            ):
                sess_seq_promo = self._make_session(
                    m_int4,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.device_expert_bytes": str(budget_int4_adaptive),
                        "session.moe.offload.popularity_file": "expert-profile.bin",
                    },
                )
                # 2-token doorbell run creates persistent decode_state_ (routing only to resident experts 0 and 1
                # for layer 0 and 1 and 4 for layer 1 so no cold promotion is queued yet).
                x_d2 = rng.standard_normal((2, 256), dtype=np.float32) * 0.5
                r_d2 = np.full((2, 8), -20.0, dtype=np.float32)
                r_d2[:, 0] = 6.0
                r_d2[:, 1] = 5.0
                self._assert_int_offload_parity(
                    self._run_bf16(sess_int4_ref2, x_d2, r_d2),
                    self._run_bf16(sess_seq_promo, x_d2, r_d2),
                    "int4_doorbell_init(T=2)",
                )
                # Repeated 8-token uncaptured decode runs take the doorbell path (16 routes) and route to cold experts
                # 5 and 7; they must advance the promotion sequence guard and promote the hot cold experts.
                for step_idx in range(5):
                    x_d8 = rng.standard_normal((8, 256), dtype=np.float32) * 0.5
                    r_d8 = np.full((8, 8), -20.0, dtype=np.float32)
                    r_d8[:, 5] = 6.0
                    r_d8[:, 2 if step_idx % 2 == 0 else 6] = 5.0
                    self._assert_int_offload_parity(
                        self._run_bf16(sess_int4_ref2, x_d8, r_d8),
                        self._run_bf16(sess_seq_promo, x_d8, r_d8),
                        f"int4_doorbell_after_doorbell(T=8,step={step_idx})",
                    )
                del sess_seq_promo, sess_int4_ref2
                gc.collect()
            promo_by_layer = _resident_promotions_per_layer(seq_promo_cap[0])
            self.assertEqual(sorted(promo_by_layer), [0, 1])
            for layer_id, promo_cnt in promo_by_layer.items():
                self.assertGreater(
                    promo_cnt,
                    0,
                    f"layer {layer_id}: repeated 8-token doorbell runs after 2-token doorbell run must promote hot cold experts",
                )
            self.assertEqual(
                _async_service_runs_per_layer(seq_promo_cap[0]),
                {0: (6, 0), 1: (6, 0)},
                "the 2-token and 8-token runs must all be served by the doorbell",
            )

            # 2. Mixed INT4 (layer 0, b32) + INT3 (layer 1, b64) with FC1/FC2 bias
            dir_int4_int3 = os.path.join(tmpdir, "int4_b32_int3_b64")
            os.makedirs(dir_int4_int3, exist_ok=True)
            m_int4_int3 = self._build_int_qmoe_model(
                dir_int4_int3,
                layer_specs=[(4, 32, 1), (3, 64, 0)],
                num_experts=8,
                hidden_size=256,
                inter_size=128,
                top_k=2,
                seed=2028,
                with_bias=True,
            )
            # H=256, I=128, INT3 b64: 24576 + 12288 + 2048 + 1024 + 8 = 39,944 B gpu slot
            int3_b64_slot = 39944
            fc1_bias_slot = 2 * 128 * 2  # 512 B
            fc2_bias_layer = 8 * 256 * 2  # 4096 B
            bias_fixed = fixed_bytes + 2 * fc2_bias_layer + 8 * fc1_bias_slot
            budget_mix = 2 * bias_fixed + 4 * (int4_slot_bytes + fc1_bias_slot) + 4 * (int3_b64_slot + fc1_bias_slot)
            sess_mix_ref = self._make_session(m_int4_int3, {"session.moe.offload.mode": "disabled"})
            sess_mix_off = self._make_session(
                m_int4_int3,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_mix),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )
            for num_rows in (1, 2, 4, 16):
                x = rng.standard_normal((num_rows, 256), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 8), dtype=np.float32)
                y_ref = self._run_bf16(sess_mix_ref, x, r)
                y_off = self._run_bf16(sess_mix_off, x, r)
                self._assert_int_offload_parity(y_ref, y_off, f"mixed_int4_int3(T={num_rows})")
            del sess_mix_off, sess_mix_ref

            # 3. Unused 1-element global-scale inputs (slots 15 and 16) on INT4 must be ignored and replaced by E unit
            # scales rather than read out of bounds during host-cache construction or expert upload.
            dir_unused_g = os.path.join(tmpdir, "int4_unused_global_scale")
            os.makedirs(dir_unused_g, exist_ok=True)
            m_unused_g_path = self._build_int_qmoe_model(
                dir_unused_g,
                layer_specs=[(4, 32, 1)],
                num_experts=8,
                hidden_size=256,
                inter_size=128,
                top_k=2,
            )
            m_unused_g_proto = onnx.load_model(m_unused_g_path, load_external_data=False)
            while len(m_unused_g_proto.graph.node[0].input) < 15:
                m_unused_g_proto.graph.node[0].input.append("")
            m_unused_g_proto.graph.node[0].input.extend(["unused_fc1_g", "unused_fc2_g"])
            for g_name in ("unused_fc1_g", "unused_fc2_g"):
                m_unused_g_proto.graph.initializer.append(helper.make_tensor(g_name, TensorProto.FLOAT, [1], [123.0]))
            onnx.save_model(m_unused_g_proto, m_unused_g_path)
            s_unused_g_ref = self._make_session(m_unused_g_path, {"session.moe.offload.mode": "disabled"})
            s_unused_g_off = self._make_session(
                m_unused_g_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(fixed_bytes + 4 * int4_slot_bytes),
                    "session.moe.offload.host_expert_bytes": str(2 * int4_host_stride),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )
            x_ug = rng.standard_normal((4, 256), dtype=np.float32) * 0.5
            r_ug = rng.standard_normal((4, 8), dtype=np.float32)
            self._assert_int_offload_parity(
                self._run_bf16(s_unused_g_ref, x_ug, r_ug),
                self._run_bf16(s_unused_g_off, x_ug, r_ug),
                "int4_unused_1elem_global_scale",
            )
            del s_unused_g_off, s_unused_g_ref

            # 4. CUDA Graph capture and replay for prepacked INT4, updating captured GPU buffers in place and switching
            # routes between resident hits and cold misses across replays.
            cudart = _load_cudart()
            if cudart is not None:
                m_int4_cg = _wrap_qmoe_model_with_cast_io(m_int4, os.path.join(dir_int4, "model_cg.onnx"), 256, 8)
                self._assert_cuda_graph_replay_parity(cudart, rng, "int4", m_int4, m_int4_cg, budget_int4_adaptive)

    def test_v4_int4_callback_service_runs_after_a_doorbell_run(self):
        self._skip_unless_int4_bf16_runner()
        int4_slot_bytes = 55304  # H=256, I=128, INT4 b32
        fixed_bytes = 2 * 8 * 4
        callback_runs = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._build_int_qmoe_model(
                tmpdir, layer_specs=[(4, 32, 1), (4, 32, 1)], num_experts=8, hidden_size=256, inter_size=128, top_k=2
            )
            ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            rng = np.random.default_rng(33624)
            with (
                mock.patch.dict(os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1"}),
                _capture_native_stderr() as captured,
            ):
                sess = self._make_session(
                    model,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.device_expert_bytes": str(2 * fixed_bytes + 12 * int4_slot_bytes),
                        "session.moe.offload.popularity_file": "expert-profile.bin",
                        # 129 tokens * top_k 2 = 258 routes exceed the 256-row doorbell limit but stay in the decode path.
                        "session.moe.offload.async_max_tokens": "129",
                    },
                )
                try:
                    # A resident-only two-token run is served by the doorbell and creates the persistent decode state.
                    x_d2 = rng.standard_normal((2, 256), dtype=np.float32) * 0.5
                    r_d2 = np.full((2, 8), -20.0, dtype=np.float32)
                    r_d2[:, 0] = 6.0
                    r_d2[:, 1] = 5.0
                    self._assert_int_offload_parity(
                        self._run_bf16(ref, x_d2, r_d2), self._run_bf16(sess, x_d2, r_d2), "int4_doorbell_init(T=2)"
                    )
                    for step_idx in range(callback_runs):
                        x_cb = rng.standard_normal((129, 256), dtype=np.float32) * 0.5
                        r_cb = np.full((129, 8), -20.0, dtype=np.float32)
                        r_cb[:, 5] = 6.0
                        r_cb[:, 2 if step_idx % 2 == 0 else 6] = 5.0
                        self._assert_int_offload_parity(
                            self._run_bf16(ref, x_cb, r_cb),
                            self._run_bf16(sess, x_cb, r_cb),
                            f"int4_callback_after_doorbell(T=129,step={step_idx})",
                        )
                finally:
                    del sess, ref
                    gc.collect()
            log_text = captured[0]
            self.assertEqual(
                _async_service_runs_per_layer(log_text),
                dict.fromkeys(range(2), (1, callback_runs)),
                "the 2-token run must use the doorbell and every 129-token run the callback",
            )
            promotions = _resident_promotions_per_layer(log_text)
            self.assertEqual(sorted(promotions), [0, 1])
            for layer_id, promo_cnt in promotions.items():
                self.assertGreater(promo_cnt, 0, f"layer {layer_id}: callback runs after a doorbell run must promote")

    def test_v4_int4_bounded_cache_pending_loads_do_not_expose_a_previous_group_lease(self):
        self._skip_unless_int4_bf16_runner()
        int4_slot_bytes = 55304  # H=256, I=128, INT4 b32
        int4_host_stride = 55808
        fixed_bytes = 2 * 8 * 4
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._build_int_qmoe_model(
                tmpdir, layer_specs=[(4, 32, 1), (4, 32, 1)], num_experts=8, hidden_size=256, inter_size=128, top_k=2
            )
            ref = self._make_session(model, {"session.moe.offload.mode": "disabled"})
            rng = np.random.default_rng(33625)
            # Twelve tokens all route to expert 3 (two 8-row calls) and cycle through the other experts, so the CPU
            # path sees calls [0, 1, 2, 3, 3, 4, 5, ...]. A 3-slot cache and 3-call groups make each group acquire its
            # experts while the previous group's leases were released and prefetch refills those slots.
            x = rng.standard_normal((12, 256), dtype=np.float32) * 0.5
            r = np.full((12, 8), -20.0, dtype=np.float32)
            r[:, 3] = 6.0
            for row in range(12):
                r[row, (0, 1, 2, 4, 5)[row % 5]] = 5.0
            env = {"ORT_QMOE_SHARED_RESIDENT_LOG": "1", "ORT_QMOE_SHARED_GROUPED_CALLS": "3"}
            with mock.patch.dict(os.environ, env), _capture_native_stderr() as captured:
                sess = self._make_session(
                    model,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.device_expert_bytes": str(2 * fixed_bytes + 4 * int4_slot_bytes),
                        "session.moe.offload.host_expert_bytes": str(2 * 3 * int4_host_stride),
                        "session.moe.offload.popularity_file": "expert-profile.bin",
                        "session.moe.offload.async_max_tokens": "12",
                    },
                )
                try:
                    # The run throws if a pending load still exposes a lease from the previous group.
                    for step_idx in range(3):
                        self._assert_int_offload_parity(
                            self._run_bf16(ref, x, r),
                            self._run_bf16(sess, x, r),
                            f"int4_bounded_cache_groups(T=12,step={step_idx})",
                        )
                finally:
                    del sess, ref
                    gc.collect()
            log_text = captured[0]
            ssd_misses = [int(m) for m in re.findall(r"ssd_misses=(\d+)", log_text)]
            self.assertTrue(
                any(ssd_misses), f"the bounded host cache must serve cold experts through loads: {log_text}"
            )

    def test_v4_int4_and_int3_offload_modes_and_numerical_parity(self):
        # The BF16 INT4 sessions that quick builds (ORT_QUICK_BUILD) cannot create live in the prepacked INT4 test.
        with tempfile.TemporaryDirectory() as tmpdir:
            fixed_bytes = 2 * 8 * 4  # 64 B per layer
            # The seed keeps the T=264 INT3 hybrid check ~30% below its L2 bound; that check is a heavy-tailed max
            # over per-token errors, so arbitrary seeds land anywhere between 0.7x and 1.4x of the bound.
            rng = np.random.default_rng(202613)
            x_long, r_long = self._hybrid_stream_inputs(rng)

            # 2. Raw INT3 (bits=3, weights_prepacked=0) with block_size=32 and block_size=64 in a mixed 2-layer model,
            # plus an FP16 block_size=128 INT3 model.
            dir_int3_mix = os.path.join(tmpdir, "int3_b32_b64")
            dir_int3_fp16_b128 = os.path.join(tmpdir, "int3_fp16_b128")
            os.makedirs(dir_int3_mix, exist_ok=True)
            os.makedirs(dir_int3_fp16_b128, exist_ok=True)

            m_int3 = self._build_int_qmoe_model(
                dir_int3_mix,
                layer_specs=[(3, 32, 0), (3, 64, 0)],
                num_experts=8,
                hidden_size=256,
                inter_size=128,
                top_k=2,
                seed=2027,
            )
            # H=256, I=128, INT3 b32: 24576 + 12288 + 4096 + 2048 + 8 = 43,016 B gpu slot; 43,520 B host stride
            # H=256, I=128, INT3 b64: 24576 + 12288 + 2048 + 1024 + 8 = 39,944 B gpu slot; 40,448 B host stride
            int3_b32_slot = 43016
            int3_b64_slot = 39944
            int3_b32_host = 43520
            int3_b64_host = 40448
            budget_int3 = 2 * fixed_bytes + 4 * int3_b32_slot + 4 * int3_b64_slot

            sess_int3_ref = self._make_session(m_int3, {"session.moe.offload.mode": "disabled"})
            sess_int3_off = self._make_session(
                m_int3,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_int3),
                    "session.moe.offload.host_expert_bytes": str(2 * (int3_b32_host + int3_b64_host)),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                    "session.moe.offload.prefill_wave_experts": "1",
                },
            )
            sess_int3_stream = self._make_session(
                m_int3,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.cpu_execution": "disabled",
                    "session.moe.offload.device_expert_bytes": str(budget_int3),
                    "session.moe.offload.prefill_wave_experts": "2",
                },
            )
            for num_rows in (1, 2, 4, 8, 16, 64):
                x = rng.standard_normal((num_rows, 256), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 8), dtype=np.float32)
                y_ref = self._run_bf16(sess_int3_ref, x, r)
                for name, s in (("int3_adaptive", sess_int3_off), ("int3_gpu_stream", sess_int3_stream)):
                    y_cand = self._run_bf16(s, x, r)
                    self._assert_int_offload_parity(y_ref, y_cand, f"{name}(T={num_rows})")
            y_int3_ref_long = self._run_bf16(sess_int3_ref, x_long, r_long)
            y_int3_off_long = self._run_bf16(sess_int3_off, x_long, r_long)
            self._assert_int_offload_parity(y_int3_ref_long, y_int3_off_long, "int3_hybrid_stream_bounded_host(T=264)")
            del sess_int3_stream, sess_int3_off, sess_int3_ref

            # FP16 + block_size=128 INT3 model
            m_int3_fp16_b128 = self._build_int_qmoe_model(
                dir_int3_fp16_b128,
                layer_specs=[(3, 128, 0)],
                num_experts=8,
                hidden_size=256,
                inter_size=128,
                top_k=2,
                seed=2029,
                use_fp16=True,
            )
            int3_b128_slot = 24576 + 12288 + 1024 + 512 + 8  # 38,408 B
            s_fp16_ref = self._make_session(m_int3_fp16_b128, {"session.moe.offload.mode": "disabled"})
            s_fp16_off = self._make_session(
                m_int3_fp16_b128,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(fixed_bytes + 3 * int3_b128_slot),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )
            for num_rows in (1, 4, 16):
                x = rng.standard_normal((num_rows, 256), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 8), dtype=np.float32)
                y_ref_fp16 = self._run_fp16(s_fp16_ref, x, r)
                y_off_fp16 = self._run_fp16(s_fp16_off, x, r)
                self._assert_int_offload_parity(y_ref_fp16, y_off_fp16, f"int3_fp16_b128(T={num_rows})")
            del s_fp16_off, s_fp16_ref

            # 3. INT3 scalar fallback shape (H=128, I=128, block_size=32) + controlled cold-route check
            dir_int3_scalar = os.path.join(tmpdir, "int3_h128_scalar")
            dir_int3_tc = os.path.join(tmpdir, "int3_h512_tc")
            os.makedirs(dir_int3_scalar, exist_ok=True)
            os.makedirs(dir_int3_tc, exist_ok=True)
            m_int3_scalar = self._build_int_qmoe_model(
                dir_int3_scalar,
                layer_specs=[(3, 32, 0)],
                num_experts=8,
                hidden_size=128,
                inter_size=128,
                top_k=2,
                seed=2030,
                scale_range=(0.06, 0.16),
            )
            int3_h128_slot = 2 * 128 * 48 + 128 * 48 + 2 * 128 * 4 * 2 + 128 * 4 * 2 + 8
            s_scalar_ref = self._make_session(m_int3_scalar, {"session.moe.offload.mode": "disabled"})
            # Static placement with 2 resident slots: popularity profile places experts 0 and 3 on GPU;
            # experts 1, 2, 4, 5, 6, 7 are cold on CPU.
            s_scalar_off = self._make_session(
                m_int3_scalar,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(fixed_bytes + 2 * int3_h128_slot),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )
            for num_rows in (1, 4, 16):
                x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 8), dtype=np.float32)
                y_ref_sc = self._run_bf16(s_scalar_ref, x, r)
                y_off_sc = self._run_bf16(s_scalar_off, x, r)
                self._assert_int_offload_parity(y_ref_sc, y_off_sc, f"int3_scalar(T={num_rows})")

            # Controlled cold-route test: route each token to 1 resident expert (0) and 1 cold expert (7).
            # Verify that the offload output matches the reference and that dropping the cold expert route (7)
            # decisively fails the parity threshold.
            x_cold = rng.standard_normal((4, 128), dtype=np.float32) * 0.5
            r_hit_and_cold = np.full((4, 8), -20.0, dtype=np.float32)
            r_hit_and_cold[:, 0] = 0.0  # resident expert 0 (weight ~ 0.5 after softmax+normalize)
            r_hit_and_cold[:, 7] = 0.0  # cold expert 7 (weight ~ 0.5 after softmax+normalize)
            r_hit_only = np.full((4, 8), -20.0, dtype=np.float32)
            r_hit_only[:, 0] = 0.0
            r_hit_only[:, 3] = -10.0  # second resident expert 3 gets negligible weight (~4.5e-5)
            y_ref_hc = self._run_bf16(s_scalar_ref, x_cold, r_hit_and_cold)
            y_off_hc = self._run_bf16(s_scalar_off, x_cold, r_hit_and_cold)
            y_missing_cold = 0.5 * self._run_bf16(s_scalar_ref, x_cold, r_hit_only)
            self._assert_int_offload_parity(y_ref_hc, y_off_hc, "int3_scalar_hit_and_cold")
            missing_cold_rel_l2 = float(
                np.max(
                    np.linalg.norm((y_ref_hc - y_missing_cold).astype(np.float64), axis=-1)
                    / np.linalg.norm(y_ref_hc.astype(np.float64), axis=-1)
                )
            )
            self.assertGreater(
                missing_cold_rel_l2,
                0.25,
                f"Dropping the cold INT3 route must cause a large relative error, got {missing_cold_rel_l2:.4f}",
            )
            del s_scalar_off, s_scalar_ref

            # 4. Unsupported quant_type='int' attribute combinations (raw INT4 weights_prepacked=0,
            # omitted/default weights_prepacked, explicit weights_prepacked=-1) must be rejected in
            # mode='required' and ignored (executed via standard GPU QMoE) in mode='auto'.
            for prepacked_decl, label in ((0, "raw_0"), (-1, "explicit_neg1"), (None, "omitted_default")):
                dir_unsup = os.path.join(tmpdir, f"int4_{label}")
                os.makedirs(dir_unsup, exist_ok=True)
                m_unsup = self._build_int_qmoe_model(
                    dir_unsup,
                    layer_specs=[(4, 32, prepacked_decl)],
                    num_experts=8,
                    hidden_size=256,
                    inter_size=128,
                    top_k=2,
                )
                with self.assertRaisesRegex(Exception, "quant_type='int' offload requires|rejected node"):
                    self._make_session(m_unsup, {"session.moe.offload.mode": "required"})

                if not self._int4_bf16_runner_supported:
                    continue  # The BF16 INT4 sessions below need the runner that quick builds do not compile.
                s_unsup_base = self._make_session(m_unsup, {"session.moe.offload.mode": "disabled"})
                s_unsup_auto = self._make_session(m_unsup, {"session.moe.offload.mode": "auto"})
                for num_rows in (1, 4, 16):
                    x_u = rng.standard_normal((num_rows, 256), dtype=np.float32) * 0.5
                    r_u = rng.standard_normal((num_rows, 8), dtype=np.float32)
                    y_u_base = self._run_bf16(s_unsup_base, x_u, r_u)
                    y_u_auto = self._run_bf16(s_unsup_auto, x_u, r_u)
                    np.testing.assert_array_equal(y_u_base, y_u_auto)
                del s_unsup_auto, s_unsup_base

            dir_int3_pre = os.path.join(tmpdir, "int3_prepacked_1")
            os.makedirs(dir_int3_pre, exist_ok=True)
            m_int3_pre = self._build_int_qmoe_model(
                dir_int3_pre,
                layer_specs=[(3, 32, 1)],
                num_experts=8,
                hidden_size=256,
                inter_size=128,
                top_k=2,
            )
            with self.assertRaisesRegex(
                Exception,
                "quant_type='int' offload requires|rejected node|QMoE INT3 requires integer quantization and explicit weights_prepacked=0",
            ):
                self._make_session(m_int3_pre, {"session.moe.offload.mode": "required"})

            # 5. Empty expert banks (E=0) must be rejected cleanly in mode='required' without division by zero.
            dir_e0 = os.path.join(tmpdir, "int4_e0")
            os.makedirs(dir_e0, exist_ok=True)
            m_e0 = self._build_int_qmoe_model(
                dir_e0,
                layer_specs=[(4, 32, 1)],
                num_experts=0,
                hidden_size=256,
                inter_size=128,
                top_k=1,
            )
            with self.assertRaisesRegex(Exception, "positive dimensions|rejected node|ShapeInferenceError|invalid"):
                self._make_session(m_e0, {"session.moe.offload.mode": "required"})

            # 4. Prepacked INT2 (bits=2, weights_prepacked=1) with block_size=32 and block_size=64, plus a
            # 3-layer mixed INT4 (b32) + INT3 (b64) + INT2 (b64) model matching qmoe_2b375_x02_y14_z32_dp23_b64.
            dir_int2_mix = os.path.join(tmpdir, "int2_b32_b64")
            dir_int4_int3_int2 = os.path.join(tmpdir, "int4_int3_int2_mix")
            os.makedirs(dir_int2_mix, exist_ok=True)
            os.makedirs(dir_int4_int3_int2, exist_ok=True)

            m_int2 = self._build_int_qmoe_model(
                dir_int2_mix,
                layer_specs=[(2, 32, 1), (2, 64, 1)],
                num_experts=8,
                hidden_size=512,
                inter_size=128,
                top_k=4,
                seed=2029,
                swiglu_limit=7.0,
            )
            # H=512, I=128, INT2 b32: 32768 + 16384 + 8192 + 4096 + 8 = 61,448 B gpu slot; 61,952 B host stride
            # H=512, I=128, INT2 b64: 32768 + 16384 + 4096 + 2048 + 8 = 55,304 B gpu slot; 55,808 B host stride
            int2_b32_slot = 61448
            int2_b64_slot = 55304
            int2_b32_host = 61952
            int2_b64_host = 55808
            budget_int2 = 2 * fixed_bytes + 4 * int2_b32_slot + 4 * int2_b64_slot

            # Use the same fused SwiGLU rounding boundary for the GPU and CPU reference paths.
            sess_int2_ref = self._make_session(
                m_int2, {"session.moe.offload.mode": "required", "session.moe.offload.placement": "static"}
            )
            sess_int2_off = self._make_session(
                m_int2,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_int2),
                    "session.moe.offload.host_expert_bytes": str(2 * (int2_b32_host + int2_b64_host)),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )
            sess_int2_stream = self._make_session(
                m_int2,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.cpu_execution": "disabled",
                    "session.moe.offload.device_expert_bytes": str(budget_int2),
                },
            )
            for num_rows in (1, 2, 4, 8, 16, 64, 256):
                x = rng.standard_normal((num_rows, 512), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 8), dtype=np.float32)
                y_ref = self._run_bf16(sess_int2_ref, x, r)
                for name, s in (("int2_adaptive", sess_int2_off), ("int2_gpu_stream", sess_int2_stream)):
                    y_cand = self._run_bf16(s, x, r)
                    self._assert_int_offload_parity(y_ref, y_cand, f"{name}(T={num_rows})", min_ref_peak=0.01)
            del sess_int2_stream, sess_int2_off, sess_int2_ref

            # The mixed model contains a BF16 INT4 layer, which needs the runner that quick builds do not compile.
            if self._int4_bf16_runner_supported:
                m_tri_mix = self._build_int_qmoe_model(
                    dir_int4_int3_int2,
                    layer_specs=[(4, 32, 1), (3, 64, 0), (2, 64, 1)],
                    num_experts=8,
                    hidden_size=512,
                    inter_size=128,
                    top_k=4,
                    seed=2030,
                    with_bias=True,
                    swiglu_limit=7.0,
                    # Exact scales isolate mixed-bank routing and bias from compounded dequantization rounding.
                    scale_range=(0.015625, 0.015625),
                )
                fc1_b_slot = 2 * 128 * 2  # 512 B
                fc2_b_layer = 8 * 512 * 2  # 8192 B
                tri_fixed = fixed_bytes + 2 * fc2_b_layer + 8 * fc1_b_slot
                int4_h512_slot = 110600
                int3_h512_slot = 79880
                budget_tri = (
                    3 * tri_fixed
                    + 4 * (int4_h512_slot + fc1_b_slot)
                    + 4 * (int3_h512_slot + fc1_b_slot)
                    + 4 * (int2_b64_slot + fc1_b_slot)
                )
                sess_tri_ref = self._make_session(
                    m_tri_mix, {"session.moe.offload.mode": "required", "session.moe.offload.placement": "static"}
                )
                sess_tri_off = self._make_session(
                    m_tri_mix,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.device_expert_bytes": str(budget_tri),
                        "session.moe.offload.popularity_file": "expert-profile.bin",
                    },
                )
                for num_rows in (1, 2, 4, 8, 16, 64):
                    x = rng.standard_normal((num_rows, 512), dtype=np.float32) * 0.25
                    r = rng.standard_normal((num_rows, 8), dtype=np.float32)
                    y_ref = self._run_bf16(sess_tri_ref, x, r)
                    y_off = self._run_bf16(sess_tri_off, x, r)
                    self._assert_int_offload_parity(
                        y_ref, y_off, f"mixed_int4_int3_int2(T={num_rows})", min_ref_peak=0.01
                    )
                del sess_tri_off, sess_tri_ref
            gc.collect()

            # 6. Prepacked INT4 with partial 64-element reduction tiles (H=96, I=128) must be rejected.
            dir_h96 = os.path.join(tmpdir, "int4_h96")
            os.makedirs(dir_h96, exist_ok=True)
            m_h96 = self._build_int_qmoe_model(
                dir_h96,
                layer_specs=[(4, 32, 1)],
                num_experts=8,
                hidden_size=96,
                inter_size=128,
                top_k=2,
            )
            with self.assertRaisesRegex(Exception, "divisible by 64|rejected node|invalid"):
                self._make_session(m_h96, {"session.moe.offload.mode": "required"})

            # 7. Malformed optional biases (e.g. FC1 bias [8, 1] or FC2 bias [8, 1]) must be rejected before upload.
            for bad_bias_suffix in ("fc1_bias", "fc2_bias"):
                dir_bad_b = os.path.join(tmpdir, f"int4_bad_{bad_bias_suffix}")
                os.makedirs(dir_bad_b, exist_ok=True)
                m_bad_b_path = self._build_int_qmoe_model(
                    dir_bad_b,
                    layer_specs=[(4, 32, 1)],
                    num_experts=8,
                    hidden_size=256,
                    inter_size=128,
                    top_k=2,
                    with_bias=True,
                )
                m_bad_b_proto = onnx.load_model(m_bad_b_path, load_external_data=False)
                for init in m_bad_b_proto.graph.initializer:
                    if init.name.endswith(bad_bias_suffix):
                        init.ClearField("external_data")
                        init.data_location = TensorProto.DEFAULT
                        del init.dims[:]
                        init.dims.extend([8, 1])
                        init.raw_data = np.zeros((8, 1), dtype=np.uint16).tobytes()
                onnx.save_model(m_bad_b_proto, m_bad_b_path)
                with self.assertRaisesRegex(Exception, "optional bias slot|fc1_bias|fc2_bias|rejected node"):
                    self._make_session(m_bad_b_path, {"session.moe.offload.mode": "required"})

            # 8. On pre-SM80 devices (simulated via ORT_QMOE_TEST_DEVICE_SM=75), mode='auto' must skip offload admission
            # before relocating expert initializers and execute on the standard GPU QMoE path, while mode='required'
            # rejects with a clear SM80 diagnostic.
            with mock.patch.dict(os.environ, {"ORT_QMOE_TEST_DEVICE_SM": "75"}):
                with self.assertRaisesRegex(Exception, "SM80|sm=75|rejected node"):
                    self._make_session(m_int3_fp16_b128, {"session.moe.offload.mode": "required"})
                s_sm75_ref = self._make_session(m_int3_fp16_b128, {"session.moe.offload.mode": "disabled"})
                s_sm75_auto = self._make_session(m_int3_fp16_b128, {"session.moe.offload.mode": "auto"})
                x_75 = rng.standard_normal((4, 256), dtype=np.float32) * 0.5
                r_75 = rng.standard_normal((4, 8), dtype=np.float32)
                np.testing.assert_array_equal(
                    self._run_fp16(s_sm75_ref, x_75, r_75),
                    self._run_fp16(s_sm75_auto, x_75, r_75),
                )
                del s_sm75_auto, s_sm75_ref

            # 9. Prepacked FP16 INT4 parity (resident hits, CPU cold routes, CPU-disabled streaming, and tight
            # working/pinned budget wave auto-sizing) + CUDA Graph capture/replay for INT3.
            dir_int4_fp16 = os.path.join(tmpdir, "int4_fp16_b32")
            os.makedirs(dir_int4_fp16, exist_ok=True)
            m_int4_fp16 = self._build_int_qmoe_model(
                dir_int4_fp16,
                layer_specs=[(4, 32, 1), (4, 32, 1)],
                num_experts=8,
                hidden_size=512,
                inter_size=512,
                top_k=2,
                seed=2031,
                with_bias=True,
                use_fp16=True,
                scale_range=(0.015, 0.045),
            )
            int4_fp16_slot = 442376 + 2048
            fp16_bias_fixed = fixed_bytes + 2 * (8 * 512 * 2) + 8 * 2048
            s_int4_fp16_ref = self._make_session(m_int4_fp16, {"session.moe.offload.mode": "disabled"})
            s_int4_fp16_hybrid = self._make_session(
                m_int4_fp16,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(2 * fp16_bias_fixed + 12 * int4_fp16_slot),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                    # Tight working/pinned budgets (4 MiB / 1 MiB) force wave auto-sizing below the default 8-expert
                    # wave (which would need > 7 MiB of GPU landing buffers alone).
                    "session.moe.offload.device_working_bytes": str(4 * 1024 * 1024),
                    "session.moe.offload.pinned_bytes": str(1 * 1024 * 1024),
                },
            )
            s_int4_fp16_stream = self._make_session(
                m_int4_fp16,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.cpu_execution": "disabled",
                    "session.moe.offload.device_expert_bytes": str(2 * fp16_bias_fixed + 6 * int4_fp16_slot),
                    "session.moe.offload.device_working_bytes": str(4 * 1024 * 1024),
                    "session.moe.offload.pinned_bytes": str(1 * 1024 * 1024),
                },
            )
            for num_rows in (1, 4, 16):
                x_f16 = rng.standard_normal((num_rows, 512), dtype=np.float32) * 0.5
                r_f16 = rng.standard_normal((num_rows, 8), dtype=np.float32)
                y_f16_ref = self._run_fp16(s_int4_fp16_ref, x_f16, r_f16)
                self._assert_int_offload_parity(
                    y_f16_ref,
                    self._run_fp16(s_int4_fp16_hybrid, x_f16, r_f16),
                    f"int4_fp16_hybrid(T={num_rows})",
                )
                self._assert_int_offload_parity(
                    y_f16_ref,
                    self._run_fp16(s_int4_fp16_stream, x_f16, r_f16),
                    f"int4_fp16_stream(T={num_rows})",
                )
            del s_int4_fp16_stream, s_int4_fp16_hybrid

            # CUDA Graph capture and replay for raw INT3, updating captured GPU buffers in place and switching routes
            # between resident hits and cold misses across replays.
            cudart = _load_cudart()
            if cudart is not None:
                m_int3_cg = _wrap_qmoe_model_with_cast_io(m_int3, os.path.join(dir_int3_mix, "model_cg.onnx"), 256, 8)
                self._assert_cuda_graph_replay_parity(
                    cudart, rng, "int3", m_int3, m_int3_cg, 2 * fixed_bytes + 6 * int3_b32_slot + 6 * int3_b64_slot
                )
            del s_int4_fp16_ref

            # 10. FP16 INT3 FC2 bias cancellation regression: dot product 4095.0 + fc2_bias -4096.0 must equal -1.0
            # (not 0.0 from premature FP16 rounding before adding bias) on both resident GPU and cold CPU routes.
            dir_int3_cancel = os.path.join(tmpdir, "int3_fp16_bias_cancel")
            os.makedirs(dir_int3_cancel, exist_ok=True)
            m_cancel_path = self._build_int_qmoe_model(
                dir_int3_cancel,
                layer_specs=[(3, 32, 0)],
                num_experts=2,
                hidden_size=128,
                inter_size=128,
                top_k=1,
                with_bias=True,
                use_fp16=True,
            )
            m_cancel = onnx.load_model(m_cancel_path, load_external_data=False)
            # Packed 3-bit code 4 (0b100 -> centered 0) in every weight byte: (0x92, 0x24, 0x49) repeating
            zero_int3_row = np.tile(np.array([0x92, 0x24, 0x49], dtype=np.uint8), 16)
            fc1_w_c = np.tile(zero_int3_row, (2, 256, 1))
            # Set FC2 weight codes to 5 (+1 centered) for the first 63 features and 0 (-4 centered) for remaining:
            # Simpler: set FC1 weights to 0 (code 4) and use fc1_bias to produce exact intermediate activation!
            # For j=0..63: gate_bias=+100 (so silu(gate) = 100), up_bias = 65.0 - 1.0 = 64.0 -> inter[0..62] = 65.0
            # Wait: even simpler! With gate_bias=0 for j>=1 (up_bias=-1 so inter[1..127]=0), and for j=0:
            # wait: in INT3 FC2, each weight code is in {-4..3}. If inter[0..127] = 1.0 for features 0..62 (63 features)
            # and inter[63] = 0.75, and FC2 weight code = +1 (code 5) with scale = 64.0:
            # 63 * 64.0 + 0.75 * 64.0 = 4032.0 + 48.0 = 4080.0 (or with scale=1.0: 63 * 64.0 + 63.0 = 4095.0!).
            # Specifically: inter[0] = 4032.0, inter[1] = 63.0, inter[2..127] = 0.0, FC2 weight = +1 (code 5) on
            # features 0 and 1, FC2 scale = 1.0 -> exact dot product = 4032.0 + 63.0 = 4095.0!
            fc2_w_c = np.tile(zero_int3_row, (2, 128, 1))
            # For features 0 and 1 in 3-bit packing: code0=5 (101), code1=5 (101), code2=4 (100) -> byte0 = 5 | (5<<3) | ((4&3)<<6) = 5 + 40 + 0 = 45 (0x2D)
            fc2_w_c[:, :, 0] = 0x2D
            fc1_s_c = np.ones((2, 256, 4), dtype=np.float16).view(np.uint16)
            fc2_s_c = np.ones((2, 128, 4), dtype=np.float16).view(np.uint16)
            fc1_b_c = np.zeros((2, 256), dtype=np.float16)
            # Set up_bias = -1.0 everywhere so inter[j] = 0 by default (since beta=1.0 in QMoE SwiGLU? Wait: default beta=0 in QMoE!).
            # In QMoE node, activation_alpha and activation_beta default to 1.0 and 0.0!
            # When gate_bias = 32.0 (silu(32.0) == 32.0 in float32) and up_bias = 4032.0 / 32.0 = 126.0 for j=0,
            # and gate_bias = 32.0, up_bias = 63.0 / 32.0 = 1.96875 for j=1:
            # Then inter[0] = 32.0 * 126.0 = 4032.0, inter[1] = 32.0 * 1.96875 = 63.0, and inter[2..127] = 0.0!
            fc1_b_c[:, 0] = np.float16(32.0)
            fc1_b_c[:, 1] = np.float16(126.0)
            fc1_b_c[:, 2] = np.float16(32.0)
            fc1_b_c[:, 3] = np.float16(1.96875)
            fc2_b_c = np.full((2, 128), -4096.0, dtype=np.float16)
            for init in m_cancel.graph.initializer:
                init.ClearField("external_data")
                init.data_location = TensorProto.DEFAULT
                if init.name.endswith("fc1_experts_weights"):
                    init.raw_data = fc1_w_c.tobytes()
                elif init.name.endswith("fc2_experts_weights"):
                    init.raw_data = fc2_w_c.tobytes()
                elif init.name.endswith("fc1_scales"):
                    init.raw_data = fc1_s_c.tobytes()
                elif init.name.endswith("fc2_scales"):
                    init.raw_data = fc2_s_c.tobytes()
                elif init.name.endswith("fc1_bias"):
                    init.raw_data = fc1_b_c.view(np.uint16).tobytes()
                elif init.name.endswith("fc2_bias"):
                    init.raw_data = fc2_b_c.view(np.uint16).tobytes()
            onnx.save_model(m_cancel, m_cancel_path)
            cancel_slot = 2 * 128 * 48 + 128 * 48 + 2 * 128 * 4 * 2 + 128 * 4 * 2 + 8 + 2 * 128 * 2
            cancel_fixed = 2 * 2 * 4 + 2 * (2 * 128 * 2) + 2 * (2 * 128 * 2)
            s_cancel_ref = self._make_session(m_cancel_path, {"session.moe.offload.mode": "disabled"})
            s_cancel_off = self._make_session(
                m_cancel_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(cancel_fixed + 1 * cancel_slot),
                },
            )
            x_zero = np.zeros((2, 128), dtype=np.float32)
            # Row 0 routes to resident expert 0; row 1 routes to cold CPU expert 1
            r_cancel = np.array([[10.0, -10.0], [-10.0, 10.0]], dtype=np.float32)
            y_cancel_ref = self._run_fp16(s_cancel_ref, x_zero, r_cancel)
            y_cancel_off = self._run_fp16(s_cancel_off, x_zero, r_cancel)
            np.testing.assert_allclose(y_cancel_ref, -1.0, rtol=0, atol=0)
            np.testing.assert_allclose(y_cancel_off, -1.0, rtol=0, atol=0)
            del s_cancel_off, s_cancel_ref

            # 11. FP16 INT4 cold-route SwiGLU projection overflow regression: gate_bias=65504 + gate_proj=32 = 65536
            # multiplied by up=2^-16 must evaluate to finite 1.0 on both GPU resident and CPU cold routes (without
            # rounding the intermediate gate projection to FP16 +inf before SwiGLU).
            dir_int4_ovf = os.path.join(tmpdir, "int4_fp16_swiglu_ovf")
            os.makedirs(dir_int4_ovf, exist_ok=True)
            m_ovf_path = self._build_int_qmoe_model(
                dir_int4_ovf,
                layer_specs=[(4, 32, 1)],
                num_experts=2,
                hidden_size=512,
                inter_size=512,
                top_k=1,
                with_bias=True,
                use_fp16=True,
            )
            m_ovf = onnx.load_model(m_ovf_path, load_external_data=False)
            fc1_w_ovf = np.full((2, 512, 512), 0x99, dtype=np.uint8)
            fc2_w_ovf = np.full((2, 512, 256), 0x99, dtype=np.uint8)
            fc1_s_ovf = np.zeros((2, 1024, 16), dtype=np.float16)
            fc1_s_ovf[:, 0::2, :] = np.float16(1.0)
            fc2_s_ovf = np.full((2, 512, 16), 1.0 / 512.0, dtype=np.float16)
            fc1_b_ovf = np.zeros((2, 1024), dtype=np.float16)
            fc1_b_ovf[:, 0::2] = np.float16(65504.0)
            fc1_b_ovf[:, 1::2] = np.float16(np.ldexp(1.0, -16))
            fc2_b_ovf = np.zeros((2, 512), dtype=np.float16)
            for init in m_ovf.graph.initializer:
                init.ClearField("external_data")
                init.data_location = TensorProto.DEFAULT
                if init.name.endswith("fc1_experts_weights"):
                    init.raw_data = fc1_w_ovf.tobytes()
                elif init.name.endswith("fc2_experts_weights"):
                    init.raw_data = fc2_w_ovf.tobytes()
                elif init.name.endswith("fc1_scales"):
                    init.raw_data = fc1_s_ovf.view(np.uint16).tobytes()
                elif init.name.endswith("fc2_scales"):
                    init.raw_data = fc2_s_ovf.view(np.uint16).tobytes()
                elif init.name.endswith("fc1_bias"):
                    init.raw_data = fc1_b_ovf.view(np.uint16).tobytes()
                elif init.name.endswith("fc2_bias"):
                    init.raw_data = fc2_b_ovf.view(np.uint16).tobytes()
            onnx.save_model(m_ovf, m_ovf_path)
            ovf_fixed = 2 * 2 * 4 + 2 * (2 * 512 * 2) + 2 * 2048
            s_ovf_ref = self._make_session(m_ovf_path, {"session.moe.offload.mode": "disabled"})
            s_ovf_off = self._make_session(
                m_ovf_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(ovf_fixed + 1 * int4_fp16_slot),
                },
            )
            x_ovf = np.zeros((2, 512), dtype=np.float32)
            x_ovf[:, :32] = 1.0
            r_ovf = np.array([[10.0, -10.0], [-10.0, 10.0]], dtype=np.float32)
            y_ovf_ref = self._run_fp16(s_ovf_ref, x_ovf, r_ovf)
            y_ovf_off = self._run_fp16(s_ovf_off, x_ovf, r_ovf)
            self.assertTrue(np.all(np.isfinite(y_ovf_ref)))
            self.assertTrue(np.all(np.isfinite(y_ovf_off)))
            np.testing.assert_allclose(y_ovf_ref, 1.0, rtol=0, atol=0)
            np.testing.assert_allclose(y_ovf_off, 1.0, rtol=0, atol=0)
            del s_ovf_off, s_ovf_ref
            gc.collect()

    def test_v4_int4_and_int3_async_resident_routes_use_their_own_fc2_bias(self):
        # The async resident path computes one row per route (token * top_k + slot), so each row has to read the FC2
        # bias of its own expert. Looking the bias up with the route's input token instead gives every later route of
        # a token the bias of an earlier one. Only experts with different FC2 biases show that, which the small random
        # biases of the other tests do not.
        num_experts = 4
        top_k = 2
        # Static placement with the profile that _build_int_qmoe_model writes keeps experts 0 and 3 resident.
        resident_experts = (0, 3)
        gate_logits = (1.0, -1.0, 0.5, -0.5, 0.75, -0.75, 0.25, -0.25)
        row_counts = (1, 2, 4, 8)
        # The first configuration runs the fast INT3 kernels and the second one the scalar kernels.
        configs = ((False, 256, 128, 64), (True, 128, 128, 32))
        rng = np.random.default_rng(2031)
        with tempfile.TemporaryDirectory() as tmpdir:
            for use_fp16, hidden_size, inter_size, block_size in configs:
                label = f"{'fp16' if use_fp16 else 'bf16'}_h{hidden_size}_i{inter_size}_b{block_size}"
                model_dir = os.path.join(tmpdir, label)
                os.makedirs(model_dir, exist_ok=True)
                model_path = self._build_int_qmoe_model(
                    model_dir,
                    layer_specs=[(3, block_size, 0)],
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    inter_size=inter_size,
                    top_k=top_k,
                    seed=2031,
                    with_bias=True,
                    use_fp16=use_fp16,
                )
                # A zero FC1 bias and a zero input make the FC1 output and the SwiGLU intermediate exactly zero, so
                # each route contributes only the FC2 bias of its expert: 0.5, 1.0, 1.5 or 2.0 plus a per-column offset.
                fc2_bias = (np.arange(num_experts, dtype=np.float32)[:, None] + 1.0) * 0.5 + (
                    np.arange(hidden_size, dtype=np.float32)[None, :] % 8
                ) * 0.125
                bias_values = {
                    "fc1_bias": np.zeros((num_experts, 2 * inter_size), dtype=np.float32),
                    "fc2_bias": fc2_bias,
                }
                model = onnx.load_model(model_path, load_external_data=False)
                patched = 0
                for init in model.graph.initializer:
                    values = bias_values.get(init.name.rsplit(".", 1)[-1])
                    if values is not None:
                        init.ClearField("external_data")
                        init.data_location = TensorProto.DEFAULT
                        init.raw_data = _encode_act_dtype_u16(values, use_fp16).tobytes()
                        patched += 1
                self.assertEqual(patched, len(bias_values), label)
                onnx.save_model(model, model_path)

                fixed_bytes = (
                    2 * num_experts * 4 + 2 * (num_experts * hidden_size * 2) + num_experts * (2 * inter_size * 2)
                )
                slot_bytes = (
                    2 * inter_size * (hidden_size * 3 // 8)
                    + hidden_size * (inter_size * 3 // 8)
                    + 2 * inter_size * (hidden_size // block_size) * 2
                    + hidden_size * (inter_size // block_size) * 2
                    + 8
                    + 2 * inter_size * 2
                )
                run = self._run_fp16 if use_fp16 else self._run_bf16
                s_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})
                with (
                    mock.patch.dict(
                        os.environ, {"ORT_QMOE_SHARED_RESIDENT_LOG": "1", "ORT_QMOE_SHARED_CPU_TIMING": "1"}
                    ),
                    _capture_native_stderr() as captured,
                ):
                    s_off = self._make_session(
                        model_path,
                        {
                            "session.moe.offload.mode": "required",
                            "session.moe.offload.placement": "static",
                            "session.moe.offload.device_expert_bytes": str(
                                fixed_bytes + len(resident_experts) * slot_bytes
                            ),
                            "session.moe.offload.popularity_file": "expert-profile.bin",
                        },
                    )
                    for num_rows in row_counts:
                        first, second = resident_experts
                        logits = np.array(gate_logits[:num_rows], dtype=np.float32)
                        router = np.full((num_rows, num_experts), -20.0, dtype=np.float32)
                        router[:, first] = logits
                        router[:, second] = 0.0
                        # The selected pair is {first, second} for every row, so normalizing the top-k weights is a
                        # softmax over the two logits. Which of them comes first in a row depends on the logit's sign.
                        weight_first = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
                        expected = (
                            weight_first[:, None] * fc2_bias[first] + (1.0 - weight_first)[:, None] * fc2_bias[second]
                        )
                        x_zero = np.zeros((num_rows, hidden_size), dtype=np.float32)
                        y_ref = run(s_ref, x_zero, router)
                        y_off = run(s_off, x_zero, router)
                        np.testing.assert_allclose(
                            y_ref, expected, rtol=2e-2, atol=5e-2, err_msg=f"{label}: disabled(T={num_rows})"
                        )
                        np.testing.assert_allclose(
                            y_off, expected, rtol=2e-2, atol=5e-2, err_msg=f"{label}: offload(T={num_rows})"
                        )

                        x = rng.standard_normal((num_rows, hidden_size), dtype=np.float32) * 0.5
                        router_random = np.full((num_rows, num_experts), -20.0, dtype=np.float32)
                        router_random[:, list(resident_experts)] = rng.uniform(-1.0, 1.0, size=(num_rows, 2))
                        self._assert_int_offload_parity(
                            run(s_ref, x, router_random),
                            run(s_off, x, router_random),
                            f"{label}: random(T={num_rows})",
                        )
                    if hidden_size == 256:
                        # Grouped INT3 prefill (total_routes >= 32) passes compact resident or wave slot indices in
                        # params.experts while keeping the full [E, H] FC2 bias bank on device; verify both resident
                        # (experts 0, 3 in slots 0, 1) and streamed (cold experts 1, 2 in wave slots 0, 1) waves resolve
                        # FC2 bias by original expert ID.
                        for grouped_rows, target_pair, source in (
                            (20, resident_experts, "resident"),
                            (40, (1, 2), "streamed"),
                        ):
                            first, second = target_pair
                            logits = np.resize(np.array(gate_logits, dtype=np.float32), grouped_rows)
                            router = np.full((grouped_rows, num_experts), -20.0, dtype=np.float32)
                            router[:, first] = logits
                            router[:, second] = 0.0
                            weight_first = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
                            expected = (
                                weight_first[:, None] * fc2_bias[first]
                                + (1.0 - weight_first)[:, None] * fc2_bias[second]
                            )
                            x_zero = np.zeros((grouped_rows, hidden_size), dtype=np.float32)
                            y_ref = run(s_ref, x_zero, router)
                            y_off = run(s_off, x_zero, router)
                            np.testing.assert_allclose(
                                y_ref,
                                expected,
                                rtol=2e-2,
                                atol=5e-2,
                                err_msg=f"{label}: grouped_{source}_disabled(T={grouped_rows})",
                            )
                            np.testing.assert_allclose(
                                y_off,
                                expected,
                                rtol=2e-2,
                                atol=5e-2,
                                err_msg=f"{label}: grouped_{source}_offload(T={grouped_rows})",
                            )
                            x_grp = rng.standard_normal((grouped_rows, hidden_size), dtype=np.float32) * 0.5
                            router_grp = np.full((grouped_rows, num_experts), -20.0, dtype=np.float32)
                            router_grp[:, list(target_pair)] = rng.uniform(-1.0, 1.0, size=(grouped_rows, 2))
                            self._assert_int_offload_parity(
                                run(s_ref, x_grp, router_grp),
                                run(s_off, x_grp, router_grp),
                                f"{label}: grouped_{source}_random(T={grouped_rows})",
                            )
                    del s_off
                    gc.collect()
                del s_ref

                # Every route has to run on a resident expert, so that the GPU kernels and not the CPU path produce the
                # rows. The CPU path counts the routes it classified as cold, and the runs of at most 4 rows are timed.
                self.assertEqual(_resident_slots_per_layer(captured[0]), {0: len(resident_experts)}, label)
                _calls, routes, cold_routes = _cpu_path_route_rows(captured[0])[0]
                self.assertGreater(routes, 0, label)
                self.assertEqual(cold_routes, 0, f"{label}: a route ran on a cold expert")
                if hidden_size == 256:
                    self.assertIn("QMoEOffload INT3 prefill layer=0 source=resident routes=40 grouped=1", captured[0])
                    self.assertIn("QMoEOffload INT3 prefill layer=0 source=streamed routes=80 grouped=1", captured[0])

    def test_v4_int4_and_int3_overridable_expert_initializers_are_not_relocated_to_cpu(self):
        # An initializer that is also a graph input can be overridden per run, so it is not constant, and the offload
        # backend only takes constant banks. Placement has to check that before it moves any bank to the CPU:
        # 'required' must reject the node, and 'auto' must leave the banks on the GPU and honor an override.
        num_experts, hidden_size, inter_size, top_k = 8, 256, 128, 2
        rng = np.random.default_rng(2032)
        x = (rng.standard_normal((4, hidden_size), dtype=np.float32) * 0.5).astype(np.float16)
        router = rng.standard_normal((4, num_experts), dtype=np.float32).astype(np.float16)
        overrides = {
            # A required slot (FC1 weights), a required scale slot, and an optional bias slot.
            "fc1_experts_weights": rng.integers(0, 256, size=(num_experts, hidden_size, inter_size), dtype=np.uint8),
            "fc1_scales": rng.uniform(0.04, 0.12, size=(num_experts, 2 * inter_size, hidden_size // 32)).astype(
                np.float16
            ),
            "fc1_bias": rng.uniform(-0.5, 0.5, size=(num_experts, 2 * inter_size)).astype(np.float16),
        }

        def run(sess, extra_inputs):
            feeds = {"X": x, "router_probs": router, **extra_inputs}
            return sess.run(["Y"], feeds)[0].astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            for suffix, override in overrides.items():
                model_dir = os.path.join(tmpdir, suffix)
                os.makedirs(model_dir, exist_ok=True)
                model_path = self._build_int_qmoe_model(
                    model_dir,
                    layer_specs=[(4, 32, 1)],
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    inter_size=inter_size,
                    top_k=top_k,
                    seed=2032,
                    with_bias=True,
                    use_fp16=True,
                )
                model = onnx.load_model(model_path, load_external_data=False)
                self.assertGreaterEqual(model.ir_version, 4, "initializers can only be overridden from IR version 4")
                name = f"model.layers.0.mlp.{suffix}"
                initializer = next(init for init in model.graph.initializer if init.name == name)
                self.assertEqual(list(initializer.dims), list(override.shape), name)
                model.graph.input.append(
                    helper.make_tensor_value_info(name, initializer.data_type, list(initializer.dims))
                )
                onnx.save_model(model, model_path)

                with self.assertRaisesRegex(Exception, "must be a constant initializer"):
                    self._make_session(model_path, {"session.moe.offload.mode": "required"})

                s_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})
                s_auto = self._make_session(model_path, {"session.moe.offload.mode": "auto"})
                y_default = run(s_ref, {})
                np.testing.assert_array_equal(run(s_auto, {}), y_default, f"{suffix}: default initializer")
                y_override = run(s_ref, {name: override})
                np.testing.assert_array_equal(run(s_auto, {name: override}), y_override, f"{suffix}: override")
                self.assertFalse(
                    np.allclose(y_default, y_override, rtol=1e-2, atol=1e-2),
                    f"{suffix}: the override must change the output for this test to observe it",
                )
                del s_auto, s_ref
                gc.collect()


def _background_thread_lifecycle_child(model_path: str) -> int:
    """Child-process half of the EP-library unregistration test.

    Unregistering the CUDA EP library removes it for the rest of the process, so it cannot run in the test process.
    """
    TestQMoEOffloadV4.setUpClass()
    slot_bytes = 13832
    sess = TestQMoEOffloadV4._make_session(
        model_path,
        {
            "session.moe.offload.mode": "required",
            "session.moe.offload.placement": "adaptive",
            "session.moe.offload.device_expert_bytes": str(2 * (2 * 16 * 4) + 16 * slot_bytes),
            "session.moe.offload.popularity_file": "expert-profile.bin",
        },
    )
    rng = np.random.default_rng(1702)
    for _ in range(4):
        x = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
        r = rng.standard_normal((2, 16), dtype=np.float32)
        _run_qmoe_bf16(sess, x, r)
    running = _qmoe_background_thread_names()
    if not running:
        print("expected the QMoE doorbell and spin threads to be running after the decode steps")
        return 1
    del sess
    gc.collect()

    ort.unregister_execution_provider_library("CUDAExecutionProvider")
    remaining = _qmoe_background_thread_names()
    deadline = time.monotonic() + 30
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = _qmoe_background_thread_names()
    if remaining:
        print(f"{len(remaining)} QMoE background threads are still running after unregistration: {remaining}")
        return 1
    print(f"{len(running)} QMoE background threads were joined by the EP library unregistration")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--background-thread-child":
        sys.exit(_background_thread_lifecycle_child(sys.argv[2]))
    # A native crash leaves no Python traceback. This prints every thread's Python stack, which names the test and
    # the call that was running. It writes to a duplicate of the original stderr, because the tests that capture
    # fd 2 would otherwise swallow the traceback.
    faulthandler.enable(file=os.dup(2))
    unittest.main(verbosity=2)
