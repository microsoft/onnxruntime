#!/usr/bin/env python3
"""
Test QMoE CPU kernel (SwiGLU only) with quantized MLAS path (4-bit and 8-bit),
and FP32 kernel (expert_weight_bits=0).

This script:
  - builds a minimal ONNX model with a single com.microsoft:QMoE node (activation_type="swiglu")
  - generates random inputs and per-expert weights
  - quantizes FC1/FC2 expert weights per output column with zero-points (8 for 4b, 128 for 8b)
  - packs 4-bit along K into K/2 bytes to match the new layout [E, N, K/pack]
  - runs the model on the CPU EP
  - computes a NumPy reference following the exact CPU kernel semantics (top-k normalize, SwiGLU interleaved, bias, accumulation)
  - compares ORT vs reference outputs and reports PASS/FAIL

Requirements: onnx, onnxruntime, numpy
"""

import math
import sys

import numpy as np

try:
    from onnx import TensorProto, helper

    import onnxruntime as ort
except Exception:
    print("ERROR: Please install onnx and onnxruntime into your Python env.")
    raise

try:
    import torch
except Exception:
    torch = None


def swiglu_interleaved_inplace(x, inter_size, alpha=1.702, beta=1.0):
    """Apply SwiGLU on interleaved data: [gate0, lin0, gate1, lin1, ...] in-place semantics for reference."""
    # Clamp behavior matches cpu kernel
    clamp_limit = 7.0
    out = np.empty((x.shape[0], inter_size), dtype=np.float32)
    # x has shape [M, 2*inter]
    for r in range(x.shape[0]):
        for i in range(inter_size):
            gate = x[r, 2 * i]
            linear = x[r, 2 * i + 1]
            gate = min(gate, clamp_limit)
            linear = max(min(linear, clamp_limit), -clamp_limit)
            sig = 1.0 / (1.0 + math.exp(-alpha * gate))
            swish = gate * sig
            out[r, i] = swish * (linear + beta)
    return out


def topk_normalized(weights_row, k):
    """Select top-k with deterministic tie-break (lower expert id first) and normalize selected weights."""
    num_experts = weights_row.shape[0]
    # sort by (-prob, expert_id)
    idx = np.lexsort((np.arange(num_experts), -weights_row))
    top_idx = idx[:k]
    top_vals = weights_row[top_idx]
    s = float(np.sum(top_vals))
    if s > 0:
        top_vals = top_vals / s
    return top_idx.astype(np.int32), top_vals.astype(np.float32)


def quantize_per_column(W, bit_width):
    """Quantize W of shape [N, K] column-wise with zp=128 (8-bit) or 8 (4-bit). Returns (qbytes, scales).

    For 8-bit: qbytes shape [N, K] uint8
    For 4-bit: qbytes shape [N, K//2] uint8 where each byte packs (low=even k, high=odd k)
    """
    N, K = W.shape
    assert (K % 2 == 0) if bit_width == 4 else True
    if bit_width == 8:
        zp = 128
        max_q = 127
        q = np.empty((N, K), dtype=np.uint8)
        scales = np.empty((N,), dtype=np.float32)
        for n in range(N):
            s = np.max(np.abs(W[n, :]))
            s = 1.0 if s == 0 else (s / max_q)
            scales[n] = s
            qn = np.clip(np.round(W[n, :] / s + zp), 0, 255).astype(np.uint8)
            q[n, :] = qn
        return q, scales
    elif bit_width == 4:
        zp = 8
        max_q = 7
        out = np.empty((N, K // 2), dtype=np.uint8)
        scales = np.empty((N,), dtype=np.float32)
        for n in range(N):
            s = np.max(np.abs(W[n, :]))
            s = 1.0 if s == 0 else (s / max_q)
            scales[n] = s
            # Vectorized bit packing - much faster than element-by-element loops
            # Quantize all values at once
            quantized = np.clip(np.round(W[n, :] / s + zp), 0, 15).astype(np.uint8)
            # Split into even and odd indices for packing
            q_even = quantized[::2]  # elements at indices 0, 2, 4, ...
            q_odd = quantized[1::2]  # elements at indices 1, 3, 5, ...
            # Pack two 4-bit values per byte: high nibble = odd, low nibble = even
            out[n, :] = (q_odd << 4) | q_even
        return out, scales
    else:
        raise ValueError("bit_width must be 4 or 8")


def dequantize_per_column(q, scales, bit_width):
    """Inverse of quantize_per_column to support NumPy reference GEMM.
    Returns float32 W of shape [N, K].
    """
    if bit_width == 8:
        zp = 128
        N, K = q.shape
        W = np.empty((N, K), dtype=np.float32)
        for n in range(N):
            W[n, :] = scales[n] * (q[n, :].astype(np.float32) - zp)
        return W
    else:
        zp = 8
        N, Khalf = q.shape
        K = Khalf * 2
        W = np.empty((N, K), dtype=np.float32)
        for n in range(N):
            # Vectorized unpacking - much faster than element-by-element loops
            packed_bytes = q[n, :]
            # Extract low and high nibbles using bitwise operations
            lo_nibbles = packed_bytes & 0x0F  # low 4 bits (even indices)
            hi_nibbles = (packed_bytes >> 4) & 0x0F  # high 4 bits (odd indices)
            # Interleave the values back to original order
            W[n, ::2] = scales[n] * (lo_nibbles.astype(np.float32) - zp)  # even indices
            W[n, 1::2] = scales[n] * (hi_nibbles.astype(np.float32) - zp)  # odd indices
        return W


def onnx_tensor_type_from_dtype(dtype):
    if dtype == np.float32:
        return TensorProto.FLOAT
    if dtype == np.float16:
        return TensorProto.FLOAT16
    raise ValueError(f"Unsupported dtype for ONNX tensor: {dtype}")


def cast_array(arr, dtype):
    return arr.astype(dtype)


# Note: This test only supports the 'swiglu' activation.


def build_qmoe_model_general(
    bit_width,
    activation_type,
    dtype,
    with_bias=True,
    normalize=True,
    input_rank=2,
    k=2,
    hidden=8,
    inter=6,
    num_rows=4,
    batch=1,
    seq=4,
    num_experts=3,
    alpha=1.702,
    beta=1.0,
):
    if activation_type != "swiglu":
        raise ValueError("This test now supports only 'swiglu' activation.")
    # Shapes
    if input_rank == 3:
        input_t = helper.make_tensor_value_info("X", onnx_tensor_type_from_dtype(dtype), [batch, seq, hidden])
        num_rows_eff = batch * seq
    else:
        input_t = helper.make_tensor_value_info("X", onnx_tensor_type_from_dtype(dtype), [num_rows, hidden])
        num_rows_eff = num_rows
    router_t = helper.make_tensor_value_info("R", onnx_tensor_type_from_dtype(dtype), [num_rows_eff, num_experts])

    is_swiglu = True
    N1 = 2 * inter
    K1 = hidden
    N2 = hidden
    K2 = inter

    rng = np.random.default_rng(2025)
    W1_fp = rng.standard_normal((num_experts, N1, K1), dtype=np.float32) * 0.1
    W2_fp = rng.standard_normal((num_experts, N2, K2), dtype=np.float32) * 0.1
    b1_fp = (
        (rng.standard_normal((num_experts, N1), dtype=np.float32) * 0.01)
        if with_bias
        else np.zeros((num_experts, N1), np.float32)
    )
    b2_fp = (
        (rng.standard_normal((num_experts, N2), dtype=np.float32) * 0.01)
        if with_bias
        else np.zeros((num_experts, N2), np.float32)
    )

    # Quantize weights or pass through for FP32
    if bit_width == 0:
        W1_q = W1_fp.copy()
        W2_q = W2_fp.copy()
        S1 = np.ones((num_experts, N1), dtype=np.float32)
        S2 = np.ones((num_experts, N2), dtype=np.float32)
        w_type = TensorProto.FLOAT
    else:
        W1_q_list, S1_list, W2_q_list, S2_list = [], [], [], []
        for e in range(num_experts):
            q1, s1 = quantize_per_column(W1_fp[e], bit_width)
            q2, s2 = quantize_per_column(W2_fp[e], bit_width)
            W1_q_list.append(q1)
            W2_q_list.append(q2)
            S1_list.append(s1)
            S2_list.append(s2)
        W1_q = np.stack(W1_q_list, axis=0).astype(np.uint8)
        W2_q = np.stack(W2_q_list, axis=0).astype(np.uint8)
        S1 = np.stack(S1_list, axis=0)
        S2 = np.stack(S2_list, axis=0)
        w_type = TensorProto.UINT8

    # Initializers (weights/scales/bias)
    def make_init(name_, arr, tp):
        return helper.make_tensor(name_, tp, arr.shape, arr.tobytes(), raw=True)

    fc1_w = make_init("fc1_w_gen", W1_q, w_type)
    fc1_s = make_init("fc1_s_gen", cast_array(S1, dtype), onnx_tensor_type_from_dtype(dtype))
    fc1_b = make_init("fc1_b_gen", cast_array(b1_fp, dtype), onnx_tensor_type_from_dtype(dtype))
    fc2_w = make_init("fc2_w_gen", W2_q, w_type)
    fc2_s = make_init("fc2_s_gen", cast_array(S2, dtype), onnx_tensor_type_from_dtype(dtype))
    fc2_b = make_init("fc2_b_gen", cast_array(b2_fp, dtype), onnx_tensor_type_from_dtype(dtype))

    qmoe = helper.make_node(
        "QMoE",
        inputs=["X", "R", fc1_w.name, fc1_s.name, fc1_b.name, fc2_w.name, fc2_s.name, fc2_b.name],
        outputs=["Y"],
        domain="com.microsoft",
        k=k,
        activation_type="swiglu",
        normalize_routing_weights=1 if normalize else 0,
        expert_weight_bits=bit_width,
        activation_alpha=float(alpha),
        activation_beta=float(beta),
    )

    y_t = helper.make_tensor_value_info(
        "Y", onnx_tensor_type_from_dtype(dtype), [batch, seq, hidden] if input_rank == 3 else [num_rows, hidden]
    )

    graph = helper.make_graph(
        nodes=[qmoe],
        name=f"QMoE_Spec_{activation_type}_{bit_width}b_{dtype!s}",
        inputs=[input_t, router_t],
        outputs=[y_t],
        initializer=[fc1_w, fc1_s, fc1_b, fc2_w, fc2_s, fc2_b],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
        producer_name="qmoe_spec_test",
    )

    # Inputs
    if input_rank == 3:
        X = cast_array(rng.standard_normal((batch, seq, hidden), dtype=np.float32), dtype)
        R = cast_array(rng.random((batch * seq, num_experts), dtype=np.float32), dtype)
    else:
        X = cast_array(rng.standard_normal((num_rows, hidden), dtype=np.float32), dtype)
        R = cast_array(rng.random((num_rows, num_experts), dtype=np.float32), dtype)

    ref = {
        "X": X,
        "R": R,
        "W1_fp": W1_fp,
        "B1": b1_fp,
        "W2_fp": W2_fp,
        "B2": b2_fp,
        "W1_q": W1_q,
        "W2_q": W2_q,
        "S1": S1,
        "S2": S2,
    }

    return (
        model,
        ref,
        {
            "is_swiglu": is_swiglu,
            "input_rank": input_rank,
            "num_rows_eff": num_rows_eff,
            "hidden": hidden,
            "inter": inter,
            "dtype": dtype,
        },
    )


def run_reference_general(ref, meta, bit_width, activation_type, k, normalize=True, alpha=1.702, beta=1.0):
    if activation_type != "swiglu":
        raise ValueError("This reference path supports only 'swiglu'.")
    # All math in fp32 for reference
    X = np.array(ref["X"]).astype(np.float32)
    R = np.array(ref["R"]).astype(np.float32)
    num_rows_eff = X.shape[0] if X.ndim == 2 else (X.shape[0] * X.shape[1])
    hidden = meta["hidden"]
    inter = meta["inter"]
    num_experts = R.shape[1]
    N1 = 2 * inter
    K1 = hidden
    N2 = hidden
    K2 = inter

    # weights
    if bit_width == 0:
        W1 = ref["W1_fp"].astype(np.float32)
        W2 = ref["W2_fp"].astype(np.float32)
    else:
        W1 = np.stack(
            [dequantize_per_column(ref["W1_q"][e], ref["S1"][e], bit_width) for e in range(num_experts)], axis=0
        )
        W2 = np.stack(
            [dequantize_per_column(ref["W2_q"][e], ref["S2"][e], bit_width) for e in range(num_experts)], axis=0
        )
    B1 = ref["B1"].astype(np.float32)
    B2 = ref["B2"].astype(np.float32)

    # flatten X if 3D
    if X.ndim == 3:
        Xf = X.reshape(-1, hidden)
    else:
        Xf = X
    Y = np.zeros((num_rows_eff, hidden), dtype=np.float32)

    # top-k selection
    def topk_and_scales(row):
        if normalize:
            idx, vals = topk_normalized(row, k)
            return idx, vals
        else:
            ord_ = np.argsort(-row)[:k]
            return ord_.astype(np.int32), row[ord_].astype(np.float32)

    for r in range(num_rows_eff):
        idx, scales = topk_and_scales(R[r])
        for j, e in enumerate(idx):
            t1 = Xf[r : r + 1, :] @ W1[e].T
            t1 = t1 + B1[e : e + 1, :]
            t1 = swiglu_interleaved_inplace(t1.astype(np.float32), K2, alpha, beta)
            t2 = t1 @ W2[e].T
            t2 = t2 + B2[e : e + 1, :]
            Y[r, :] += scales[j] * t2[0, :]
    return Y


def build_qmoe_model(
    bit_width, num_rows, hidden, inter, num_experts, k, with_bias=True, normalize=True, alpha=1.702, beta=1.0
):
    # Inputs
    input_t = helper.make_tensor_value_info("X", TensorProto.FLOAT, [num_rows, hidden])
    router_t = helper.make_tensor_value_info("R", TensorProto.FLOAT, [num_rows, num_experts])

    # Shapes according to new layout (pack_size = 2 if 4-bit else 1)
    pack = 2 if bit_width == 4 else 1
    N1 = 2 * inter  # swiglu doubles fc1 out
    K1 = hidden
    N2 = hidden
    K2 = inter

    def name(n):
        return f"{n}_{bit_width}b"

    # Random FP32 weights to quantize
    rng = np.random.default_rng(1234)
    W1_fp = rng.standard_normal((num_experts, N1, K1), dtype=np.float32) * 0.1
    W2_fp = rng.standard_normal((num_experts, N2, K2), dtype=np.float32) * 0.1
    b1_fp = (
        rng.standard_normal((num_experts, N1), dtype=np.float32) * 0.01
        if with_bias
        else np.zeros((num_experts, N1), np.float32)
    )
    b2_fp = (
        rng.standard_normal((num_experts, N2), dtype=np.float32) * 0.01
        if with_bias
        else np.zeros((num_experts, N2), np.float32)
    )

    # Quantize per expert per column
    W1_q_list, S1_list, W2_q_list, S2_list = [], [], [], []
    for e in range(num_experts):
        q1, s1 = quantize_per_column(W1_fp[e], bit_width)
        q2, s2 = quantize_per_column(W2_fp[e], bit_width)
        W1_q_list.append(q1)
        S1_list.append(s1)
        W2_q_list.append(q2)
        S2_list.append(s2)

    W1_q = np.stack(W1_q_list, axis=0)
    W2_q = np.stack(W2_q_list, axis=0)
    S1 = np.stack(S1_list, axis=0)
    S2 = np.stack(S2_list, axis=0)

    # Initializers (match kernel input order)
    def make_init(name_, arr, tp):
        return helper.make_tensor(name_, tp, arr.shape, arr.tobytes(), raw=True)

    fc1_w = make_init(name("fc1_w"), W1_q.astype(np.uint8), TensorProto.UINT8)
    fc1_s = make_init(name("fc1_s"), S1.astype(np.float32), TensorProto.FLOAT)
    fc1_b = make_init(name("fc1_b"), b1_fp.astype(np.float32), TensorProto.FLOAT)
    fc2_w = make_init(name("fc2_w"), W2_q.astype(np.uint8), TensorProto.UINT8)
    fc2_s = make_init(name("fc2_s"), S2.astype(np.float32), TensorProto.FLOAT)
    fc2_b = make_init(name("fc2_b"), b2_fp.astype(np.float32), TensorProto.FLOAT)

    # Node
    qmoe = helper.make_node(
        "QMoE",
        inputs=["X", "R", fc1_w.name, fc1_s.name, fc1_b.name, fc2_w.name, fc2_s.name, fc2_b.name],
        outputs=["Y"],
        domain="com.microsoft",
        k=k,
        activation_type="swiglu",
        normalize_routing_weights=1 if normalize else 0,
        expert_weight_bits=bit_width,
        activation_alpha=float(alpha),
        activation_beta=float(beta),
    )

    y_t = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [num_rows, hidden])

    graph = helper.make_graph(
        nodes=[qmoe],
        name=f"QMoE_CPU_SwiGLU_{bit_width}b",
        inputs=[input_t, router_t],
        outputs=[y_t],
        initializer=[fc1_w, fc1_s, fc1_b, fc2_w, fc2_s, fc2_b],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
        producer_name="qmoe_cpu_swiglu_test",
    )

    # Inputs (data)
    X = rng.standard_normal((num_rows, hidden), dtype=np.float32)
    # router probs non-negative (softmax logits scenario); normalization handled by op attribute
    R = rng.random((num_rows, num_experts), dtype=np.float32)

    # Pack reference dict
    ref = {
        "X": X,
        "R": R,
        "W1_q": W1_q,
        "S1": S1,
        "B1": b1_fp,
        "W2_q": W2_q,
        "S2": S2,
        "B2": b2_fp,
    }

    return model, ref


def run_reference(ref, bit_width, k, inter, normalize=True, alpha=1.702, beta=1.0):
    X = ref["X"]
    R = ref["R"]
    W1_q = ref["W1_q"]
    S1 = ref["S1"]
    B1 = ref["B1"]
    W2_q = ref["W2_q"]
    S2 = ref["S2"]
    B2 = ref["B2"]

    num_rows, hidden = X.shape
    num_experts = R.shape[1]
    N1 = B1.shape[1]
    K1 = hidden
    N2 = hidden
    K2 = inter

    # Precompute dequantized float weights per expert
    W1 = np.stack([dequantize_per_column(W1_q[e], S1[e], bit_width) for e in range(num_experts)], axis=0)
    W2 = np.stack([dequantize_per_column(W2_q[e], S2[e], bit_width) for e in range(num_experts)], axis=0)

    Y = np.zeros((num_rows, hidden), dtype=np.float32)

    for r in range(num_rows):
        # select top-k and normalize
        if normalize:
            idx, vals = topk_normalized(R[r], k)
            scales = vals
        else:
            order = np.argsort(-R[r])[:k]
            idx = order.astype(np.int32)
            scales = R[r][order].astype(np.float32)
        for j, e in enumerate(idx):
            # FC1: [1,K1] @ [N1,K1]^T -> [1,N1]
            t1 = X[r : r + 1, :] @ W1[e].T  # (1, N1)
            t1 = t1 + B1[e : e + 1, :]
            # SwiGLU
            t1_swiglu = swiglu_interleaved_inplace(t1.astype(np.float32), K2, alpha, beta)
            # FC2: [1,K2] @ [N2,K2]^T
            t2 = t1_swiglu @ W2[e].T  # (1, N2)
            t2 = t2 + B2[e : e + 1, :]
            Y[r, :] += scales[j] * t2[0, :]
    return Y


def run_reference_torch(ref, bit_width, k, inter, normalize=True, alpha=1.702, beta=1.0):
    if torch is None:
        raise RuntimeError("PyTorch is not available in this environment.")

    X = torch.from_numpy(ref["X"]).to(torch.float32)
    R = torch.from_numpy(ref["R"]).to(torch.float32)
    W1_q = ref["W1_q"]
    S1 = ref["S1"]
    B1 = torch.from_numpy(ref["B1"]).to(torch.float32)
    W2_q = ref["W2_q"]
    S2 = ref["S2"]
    B2 = torch.from_numpy(ref["B2"]).to(torch.float32)

    num_rows, hidden = X.shape
    num_experts = R.shape[1]

    # Dequantize to float32 torch tensors
    W1 = []
    W2 = []
    for e in range(num_experts):
        W1e = torch.from_numpy(dequantize_per_column(W1_q[e], S1[e], bit_width)).to(torch.float32)
        W2e = torch.from_numpy(dequantize_per_column(W2_q[e], S2[e], bit_width)).to(torch.float32)
        W1.append(W1e)
        W2.append(W2e)

    Y = torch.zeros((num_rows, hidden), dtype=torch.float32)

    for r in range(num_rows):
        # Use numpy helper to ensure same tie-break semantics
        idx, vals = (
            topk_normalized(R[r].cpu().numpy(), k)
            if normalize
            else (
                np.argsort(-R[r].cpu().numpy())[:k].astype(np.int32),
                R[r].cpu().numpy()[np.argsort(-R[r].cpu().numpy())[:k]].astype(np.float32),
            )
        )
        scales = torch.from_numpy(vals).to(torch.float32)
        for j, e in enumerate(idx):
            t1 = X[r : r + 1, :] @ W1[e].T  # (1, 2*inter)
            t1 = t1 + B1[e : e + 1, :]
            gate = t1[..., ::2]
            linear = t1[..., 1::2]
            gate = torch.minimum(gate, torch.tensor(7.0, dtype=torch.float32))
            linear = torch.clamp(linear, -7.0, 7.0)
            swish = gate * torch.sigmoid(torch.tensor(alpha, dtype=torch.float32) * gate)
            t1_swiglu = swish * (linear + torch.tensor(beta, dtype=torch.float32))  # (1, inter)
            t2 = t1_swiglu @ W2[e].T  # (1, hidden)
            t2 = t2 + B2[e : e + 1, :]
            Y[r, :] += scales[j] * t2[0, :]

    return Y.cpu().numpy()


# FP32 path helpers
def build_qmoe_model_fp32(
    num_rows, hidden, inter, num_experts, k, with_bias=True, normalize=True, alpha=1.702, beta=1.0
):
    input_t = helper.make_tensor_value_info("X", TensorProto.FLOAT, [num_rows, hidden])
    router_t = helper.make_tensor_value_info("R", TensorProto.FLOAT, [num_rows, num_experts])

    N1 = 2 * inter
    K1 = hidden
    N2 = hidden
    K2 = inter

    rng = np.random.default_rng(4321)
    W1_fp = rng.standard_normal((num_experts, N1, K1), dtype=np.float32) * 0.1
    W2_fp = rng.standard_normal((num_experts, N2, K2), dtype=np.float32) * 0.1
    b1_fp = (
        rng.standard_normal((num_experts, N1), dtype=np.float32) * 0.01
        if with_bias
        else np.zeros((num_experts, N1), np.float32)
    )
    b2_fp = (
        rng.standard_normal((num_experts, N2), dtype=np.float32) * 0.01
        if with_bias
        else np.zeros((num_experts, N2), np.float32)
    )

    # Scales are not used in FP32 path but included to satisfy shape checks: set to ones
    S1 = np.ones((num_experts, N1), dtype=np.float32)
    S2 = np.ones((num_experts, N2), dtype=np.float32)

    def make_init(name_, arr, tp):
        return helper.make_tensor(name_, tp, arr.shape, arr.tobytes(), raw=True)

    fc1_w = make_init("fc1_w_fp32", W1_fp.astype(np.float32), TensorProto.FLOAT)
    fc1_s = make_init("fc1_s_fp32", S1.astype(np.float32), TensorProto.FLOAT)
    fc1_b = make_init("fc1_b_fp32", b1_fp.astype(np.float32), TensorProto.FLOAT)
    fc2_w = make_init("fc2_w_fp32", W2_fp.astype(np.float32), TensorProto.FLOAT)
    fc2_s = make_init("fc2_s_fp32", S2.astype(np.float32), TensorProto.FLOAT)
    fc2_b = make_init("fc2_b_fp32", b2_fp.astype(np.float32), TensorProto.FLOAT)

    qmoe = helper.make_node(
        "QMoE",
        inputs=["X", "R", fc1_w.name, fc1_s.name, fc1_b.name, fc2_w.name, fc2_s.name, fc2_b.name],
        outputs=["Y"],
        domain="com.microsoft",
        k=k,
        activation_type="swiglu",
        normalize_routing_weights=1 if normalize else 0,
        expert_weight_bits=0,
        activation_alpha=float(alpha),
        activation_beta=float(beta),
    )

    y_t = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [num_rows, hidden])

    graph = helper.make_graph(
        nodes=[qmoe],
        name="QMoE_CPU_SwiGLU_FP32",
        inputs=[input_t, router_t],
        outputs=[y_t],
        initializer=[fc1_w, fc1_s, fc1_b, fc2_w, fc2_s, fc2_b],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
        producer_name="qmoe_cpu_swiglu_test_fp32",
    )

    X = rng.standard_normal((num_rows, hidden), dtype=np.float32)
    R = rng.random((num_rows, num_experts), dtype=np.float32)

    ref = {
        "X": X,
        "R": R,
        "W1_fp": W1_fp,
        "B1": b1_fp,
        "W2_fp": W2_fp,
        "B2": b2_fp,
    }

    return model, ref


def run_reference_fp32(ref, k, inter, normalize=True, alpha=1.702, beta=1.0):
    X = ref["X"]
    R = ref["R"]
    W1_fp = ref["W1_fp"]
    B1 = ref["B1"]
    W2_fp = ref["W2_fp"]
    B2 = ref["B2"]

    num_rows, hidden = X.shape
    num_experts = R.shape[1]

    Y = np.zeros((num_rows, hidden), dtype=np.float32)
    for r in range(num_rows):
        idx, vals = (
            topk_normalized(R[r], k)
            if normalize
            else (np.argsort(-R[r])[:k].astype(np.int32), R[r][np.argsort(-R[r])[:k]].astype(np.float32))
        )
        scales = vals
        for j, e in enumerate(idx):
            t1 = X[r : r + 1, :] @ W1_fp[e].T
            t1 = t1 + B1[e : e + 1, :]
            t1_swiglu = swiglu_interleaved_inplace(t1.astype(np.float32), inter, alpha, beta)
            t2 = t1_swiglu @ W2_fp[e].T
            t2 = t2 + B2[e : e + 1, :]
            Y[r, :] += scales[j] * t2[0, :]
    return Y


def run_reference_torch_fp32(ref, k, inter, normalize=True, alpha=1.702, beta=1.0):
    if torch is None:
        raise RuntimeError("PyTorch is not available in this environment.")
    X = torch.from_numpy(ref["X"]).to(torch.float32)
    R = torch.from_numpy(ref["R"]).to(torch.float32)
    W1_fp = torch.from_numpy(ref["W1_fp"]).to(torch.float32)
    B1 = torch.from_numpy(ref["B1"]).to(torch.float32)
    W2_fp = torch.from_numpy(ref["W2_fp"]).to(torch.float32)
    B2 = torch.from_numpy(ref["B2"]).to(torch.float32)

    num_rows, hidden = X.shape
    num_experts = R.shape[1]
    Y = torch.zeros((num_rows, hidden), dtype=torch.float32)
    for r in range(num_rows):
        idx, vals = (
            topk_normalized(R[r].cpu().numpy(), k)
            if normalize
            else (
                np.argsort(-R[r].cpu().numpy())[:k].astype(np.int32),
                R[r].cpu().numpy()[np.argsort(-R[r].cpu().numpy())[:k]].astype(np.float32),
            )
        )
        scales = torch.from_numpy(vals).to(torch.float32)
        for j, e in enumerate(idx):
            t1 = X[r : r + 1, :] @ W1_fp[e].T
            t1 = t1 + B1[e : e + 1, :]
            gate = t1[..., ::2]
            linear = t1[..., 1::2]
            gate = torch.minimum(gate, torch.tensor(7.0, dtype=torch.float32))
            linear = torch.clamp(linear, -7.0, 7.0)
            swish = gate * torch.sigmoid(torch.tensor(alpha, dtype=torch.float32) * gate)
            t1_swiglu = swish * (linear + torch.tensor(beta, dtype=torch.float32))
            t2 = t1_swiglu @ W2_fp[e].T
            t2 = t2 + B2[e : e + 1, :]
            Y[r, :] += scales[j] * t2[0, :]
    return Y.cpu().numpy()


def run_one(bit_width):
    # Small, even K for 4-bit
    num_rows = 4
    hidden = 2880
    inter = 2880
    num_experts = 4
    k = 2

    print("Building qmoe model")
    model, ref = build_qmoe_model(bit_width, num_rows, hidden, inter, num_experts, k, with_bias=True, normalize=True)

    print("starting session")
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    print("Running session")
    ort_out = sess.run(["Y"], {"X": ref["X"], "R": ref["R"]})[0]
    print("Session run complete")

    ref_np = run_reference(ref, bit_width, k, inter, normalize=True)
    max_abs_np = float(np.max(np.abs(ort_out - ref_np)))
    print(f"bit_width={bit_width} ORT-vs-NumPy max_diff={max_abs_np:.6f}")

    if torch is not None:
        ref_torch = run_reference_torch(ref, bit_width, k, inter, normalize=True)
        max_abs_torch = float(np.max(np.abs(ort_out - ref_torch)))
        print(f"bit_width={bit_width} ORT-vs-PyTorch max_diff={max_abs_torch:.6f}")
    else:
        print("PyTorch not available; skipping ORT-vs-PyTorch comparison.")

    return True


def run_fp32():
    num_rows = 4
    hidden = 2880
    inter = 2880
    num_experts = 4
    k = 2

    model, ref = build_qmoe_model_fp32(num_rows, hidden, inter, num_experts, k, with_bias=True, normalize=True)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["Y"], {"X": ref["X"], "R": ref["R"]})[0]

    ref_np = run_reference_fp32(ref, k, inter, normalize=True)
    max_abs_np = float(np.max(np.abs(ort_out - ref_np)))
    print(f"fp32 ORT-vs-NumPy max_diff={max_abs_np:.6f}")

    if torch is not None:
        ref_torch = run_reference_torch_fp32(ref, k, inter, normalize=True)
        max_abs_torch = float(np.max(np.abs(ort_out - ref_torch)))
        print(f"fp32 ORT-vs-PyTorch max_diff={max_abs_torch:.6f}")
    else:
        print("PyTorch not available; skipping FP32 ORT-vs-PyTorch comparison.")


def run_spec_coverage_tests():
    print("\n==== QMoE op spec coverage tests ====")
    dtypes = [np.float32, np.float16]
    tests = [
        # activation, bits, with_bias, normalize, input_rank, k
        ("swiglu", 8, True, True, 2, 2),
        ("swiglu", 4, True, True, 2, 2),
        ("swiglu", 0, True, True, 2, 2),
    ]
    for dt in dtypes:
        print(f"\n-- dtype={dt} --")
        for act, bits, with_bias, norm, rank, k in tests:
            try:
                model, ref, meta = build_qmoe_model_general(bits, act, dt, with_bias, norm, input_rank=rank, k=k)
                sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
                ort_out = sess.run(["Y"], {"X": ref["X"], "R": ref["R"]})[0]
                ref_out = run_reference_general(ref, meta, bits, act, k, normalize=norm)
                if rank == 3:
                    ort_flat = np.array(ort_out).reshape(-1, meta["hidden"]).astype(np.float32)
                else:
                    ort_flat = np.array(ort_out).astype(np.float32)
                max_diff = float(np.max(np.abs(ort_flat - ref_out)))
                tol = 0.2 if bits in (4, 8) else 0.02
                status = "PASS" if max_diff < tol else "FAIL"
                print(
                    f"[{status}] act={act} bits={bits} bias={with_bias} norm={norm} rank={rank} k={k} max_diff={max_diff:.4f} tol={tol}"
                )
            except Exception as e:
                print(f"[ERROR] act={act} bits={bits} bias={with_bias} norm={norm} rank={rank} k={k} -> {e}")


def main():
    run_one(8)
    run_one(4)
    run_fp32()
    run_spec_coverage_tests()
    sys.exit(0)


if __name__ == "__main__":
    main()
