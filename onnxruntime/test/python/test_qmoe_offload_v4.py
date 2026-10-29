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
    arr = np.ascontiguousarray(arr, dtype=np.float32)
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

    @classmethod
    def setUpClass(cls):
        for env_key in ("ORT_CUDA_PLUGIN_PATH", "QWEN_PROVIDER"):
            plugin_path = os.environ.get(env_key)
            if plugin_path and os.path.exists(plugin_path):
                with contextlib.suppress(Exception):
                    ort.unregister_execution_provider_library("CUDAExecutionProvider")
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

    def setUp(self):
        if not self._nvfp4_supported and not self._testMethodName.startswith(("test_v4_int4", "test_v4_int3")):
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
    ):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
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

            # Control: plain runs must record routes. CPU-disabled streaming keeps placement fixed, so promotions
            # remain zero even though all of these ordinary cold routes are observed.
            control = run_session(warmup_runs=0, plain_runs=6)
            assert_streamed(control, "control")
            self.assertEqual(_resident_slots_per_layer(control), {0: 4, 1: 4})
            self.assertEqual(_resident_routes_per_layer(control), dict.fromkeys(range(2), 6 * rows * top_k))
            for layer, promotions in _resident_promotions_per_layer(control).items():
                self.assertEqual(promotions, 0, f"layer {layer}: CPU-disabled streaming keeps placement fixed")

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
    ) -> str:
        """Build a multi-layer INT4/INT3 QMoE model where each layer spec is (weight_bits, block_size, prepacked)."""
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
                    "normalize_routing_weights": 1,
                    "k": min(top_k, max(1, num_experts)),
                    "expert_weight_bits": bits,
                    "block_size": blk_size,
                    "swiglu_fusion": 1,
                    "quant_type": "int",
                }
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

    def test_v4_int4_streaming_reduces_scale_staging_wave_to_fit_pinned_budget(self):
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

    def test_v4_int4_eight_token_hybrid_cuda_graph_replay(self):
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

    def _build_decode_optimization_model(
        self,
        model_dir: str,
        num_layers: int = 2,
        use_fp16: bool = True,
        num_experts: int = 16,
        resident_slots: int = 10,
    ):
        model_path = self._build_int_qmoe_model(
            model_dir,
            layer_specs=[(4, 32, 1)] * num_layers,
            num_experts=num_experts,
            hidden_size=256,
            inter_size=128,
            top_k=8,
            use_fp16=use_fp16,
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
                        x_value = ort.OrtValue.ortvalue_from_numpy(cases[0][0].astype(np.float16), "cuda", 0)
                        r_value = ort.OrtValue.ortvalue_from_numpy(cases[0][1].astype(np.float16), "cuda", 0)
                        y_value = ort.OrtValue.ortvalue_from_shape_and_type(cases[0][0].shape, np.float16, "cuda", 0)
                        binding = session.io_binding()
                        binding.bind_ortvalue_input("X", x_value)
                        binding.bind_ortvalue_input("router_probs", r_value)
                        binding.bind_ortvalue_output("Y", y_value)
                        outputs = []
                        for x, router in cases:
                            x_value.update_inplace(x.astype(np.float16))
                            r_value.update_inplace(router.astype(np.float16))
                            session.run_with_iobinding(binding)
                            outputs.append(y_value.numpy().astype(np.float32))
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

    def test_v4_skip_zero_cold_doorbell_policy_validation_and_env_precedence(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_path, config = self._build_decode_optimization_model(model_dir, num_layers=1)
            key = "session.moe.offload.skip_zero_cold_doorbell"
            for invalid in ("2", "-1", "invalid"):
                with self.subTest(value=invalid), self.assertRaisesRegex(Exception, "Invalid " + key):
                    self._make_session(model_path, {**config, key: invalid})
            x = np.full((1, 256), 0.125, dtype=np.float32)
            router = np.linspace(3.0, -3.0, 16, dtype=np.float32).reshape(1, 16)
            with mock.patch.dict(os.environ, {"ORT_QMOE_SKIP_ZERO_COLD_DOORBELL": "1"}):
                _, enabled = self._run_decode_with_telemetry(model_path, config, [(x, router)])
                _, explicit_zero = self._run_decode_with_telemetry(model_path, {**config, key: "0"}, [(x, router)])
            self.assertEqual(enabled[0]["zero_cold_short_circuits"], 1)
            self.assertEqual(explicit_zero[0]["zero_cold_short_circuits"], 0)

    def test_v4_int4_and_int3_offload_modes_and_numerical_parity(self):
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
            budget_int4_adaptive = 2 * fixed_bytes + 8 * int4_slot_bytes  # 4 slots/layer (2 resident + 2 spare)

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
            del sess_int4_stream, sess_int4_adaptive, sess_int4_static, sess_int4_ref

            # 2. Raw INT3 (bits=3, weights_prepacked=0) with block_size=32 and block_size=64 in a mixed 2-layer model,
            # plus a mixed INT4 (layer 0, b32) + INT3 (layer 1, b64) model and an FP16 block_size=128 INT3 model.
            dir_int3_mix = os.path.join(tmpdir, "int3_b32_b64")
            dir_int4_int3 = os.path.join(tmpdir, "int4_b32_int3_b64")
            dir_int3_fp16_b128 = os.path.join(tmpdir, "int3_fp16_b128")
            os.makedirs(dir_int3_mix, exist_ok=True)
            os.makedirs(dir_int4_int3, exist_ok=True)
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
            del sess_int3_stream, sess_int3_off, sess_int3_ref

            # Mixed INT4 + INT3 with FC1/FC2 bias
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
            for num_rows in (1, 2, 4, 8, 16, 64):
                x = rng.standard_normal((num_rows, 256), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 8), dtype=np.float32)
                y_ref = self._run_bf16(sess_mix_ref, x, r)
                y_off = self._run_bf16(sess_mix_off, x, r)
                self._assert_int_offload_parity(y_ref, y_off, f"mixed_int4_int3(T={num_rows})")
            del sess_mix_off, sess_mix_ref

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

                s_unsup_base = self._make_session(m_unsup, {"session.moe.offload.mode": "disabled"})
                s_unsup_auto = self._make_session(m_unsup, {"session.moe.offload.mode": "auto"})
                for num_rows in (1, 4, 16):
                    x_u = rng.standard_normal((num_rows, 256), dtype=np.float32) * 0.5
                    r_u = rng.standard_normal((num_rows, 8), dtype=np.float32)
                    y_u_base = self._run_bf16(s_unsup_base, x_u, r_u)
                    y_u_auto = self._run_bf16(s_unsup_auto, x_u, r_u)
                    np.testing.assert_array_equal(y_u_base, y_u_auto)
                del s_unsup_auto, s_unsup_base

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
