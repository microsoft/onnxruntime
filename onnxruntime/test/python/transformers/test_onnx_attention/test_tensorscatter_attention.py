# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Tests for TensorScatter(opset 24) + Attention(opset 24) pattern.

Demonstrates a decode step where new KV entries are scattered into a
pre-allocated cache via TensorScatter, then Attention uses the updated
KV cache with nonpad_kv_seqlen to mask out padding positions.

Uses IO Binding for in-place KV cache updates, matching the real-world LLM
inference pattern where KV cache buffers are pre-allocated on the device
and reused across decode steps.

The graph looks like:

  key_cache (B, S, kv_hidden)  ──────────┐
  new_k (B, q_seq, kv_hidden)  ──────────┤
  write_indices (B,)  ───────────────────┤
                                          ├─ TensorScatter(axis=1) ─→ updated_key_cache ─┐
                                                                                          │
  value_cache (B, S, kv_hidden)  ────────┐                                                │
  new_v (B, q_seq, kv_hidden)  ──────────┤                                                │
  write_indices (B,)  ──────────────────┤                                                 │
                                          ├─ TensorScatter(axis=1) ─→ updated_value_cache ┤
                                                                                           │
  Q (B, q_seq, q_hidden) ──────────────┬─ Attention(opset 24)  ←──────────────────────────┘
  nonpad_kv_seqlen (B,)  ──────────────┘          │
                                                   ├─ output
                                                   ├─ present_key
                                                   └─ present_value

IO Binding enables in-place cache updates: the same OrtValue buffer is bound as
both TensorScatter input (key_cache/value_cache) and output
(updated_key_cache/updated_value_cache), avoiding unnecessary copies.

CUDA support:
  - GQA path (kv_num_heads != q_num_heads) uses flash attention for external KV cache (fp16/bf16)
  - MHA path (kv_num_heads == q_num_heads) uses flash attention for fp16/bf16,
    unfused attention_bias fallback for fp32
"""

import gc
import math
import os
import re
import sys
import threading
import unittest

import numpy
import torch
from onnx import TensorProto, helper
from parameterized import parameterized

from onnxruntime import InferenceSession, OrtValue, SessionOptions, get_available_providers

# #################################################################################################
#  Helper Functions
# #################################################################################################


class _CaptureStdout:
    """Capture output written to OS file descriptor 1 (C++ stdout).

    The attention kernel debug info is emitted by the native ONNX Runtime library directly to
    fd 1, which Python's contextlib.redirect_stdout (which only swaps sys.stdout) cannot
    intercept, so fd-level dup2 redirection is used instead. Mirrors CaptureStdout in test_gqa.py.
    """

    def __init__(self):
        self.fd = 1
        self.chunk_size = 1024
        self.output = b""

    def _capture(self):
        chunks = []
        while chunk := os.read(self._pipe_reader, self.chunk_size):
            chunks.append(chunk)
        self.output = b"".join(chunks)

    def __enter__(self):
        sys.stdout.flush()
        self._duped_fd = os.dup(self.fd)
        self._pipe_reader, pipe_writer = os.pipe()
        os.dup2(pipe_writer, self.fd)
        os.close(pipe_writer)
        self._capture_thread = threading.Thread(target=self._capture)
        self._capture_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        os.dup2(self._duped_fd, self.fd)
        self._capture_thread.join()
        os.close(self._pipe_reader)
        os.close(self._duped_fd)


def _parse_sdpa_kernel(captured_text):
    """Extract the SdpaKernel=... tier name emitted by AttentionKernelDebugInfo::Print."""
    match = re.search(r"SdpaKernel=(?P<kernel>[A-Z_]+)", captured_text)
    return match.group("kernel") if match is not None else None


_CUDNN_DECODE_SUPPORTED_CACHE = {}


def _observe_cudnn_decode_dispatch(head_size):
    """Build a minimal q_seq==1 external-KV decode graph, run it once requesting the cuDNN SDPA
    kernel with dispatch debug info enabled, and return the observed SdpaKernel string (or None).

    This asks the ORT build itself which tier it selects, so the test's notion of "supported"
    exactly matches the kernel's own cuDNN gate (cudnn_sdpa::is_stable and the cuDNN version ORT
    actually dlopened) with no Python-side version reimplementation.
    """
    batch_size, total_kv_seq_len, q_num_heads, kv_num_heads = 1, 2, 1, 1
    model = build_tensorscatter_attention_graph(
        batch_size=batch_size,
        total_kv_seq_len=total_kv_seq_len,
        q_seq_len=1,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        head_size=head_size,
        ort_type=TensorProto.FLOAT16,
        is_causal=0,
    )
    kv_hidden = kv_num_heads * head_size
    feeds = {
        "key_cache": numpy.zeros((batch_size, total_kv_seq_len, kv_hidden), numpy.float16),
        "value_cache": numpy.zeros((batch_size, total_kv_seq_len, kv_hidden), numpy.float16),
        "new_k": numpy.zeros((batch_size, 1, kv_hidden), numpy.float16),
        "new_v": numpy.zeros((batch_size, 1, kv_hidden), numpy.float16),
        "query": numpy.zeros((batch_size, 1, q_num_heads * head_size), numpy.float16),
        "write_indices": numpy.zeros((batch_size,), numpy.int64),
        "nonpad_kv_seqlen": numpy.ones((batch_size,), numpy.int64),
    }
    provider_options = {"sdpa_kernel": str(_SDPA_KERNEL_CUDNN_WITH_MATH_FALLBACK)}
    previous = os.environ.get("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO")
    os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "1"
    try:
        session = InferenceSession(model, SessionOptions(), providers=[("CUDAExecutionProvider", provider_options)])
        with _CaptureStdout() as captured:
            session.run(None, feeds)
    finally:
        if previous is None:
            os.environ.pop("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO", None)
        else:
            os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = previous
    return _parse_sdpa_kernel(captured.output.decode(errors="replace"))


def cudnn_decode_supported(head_size):
    """Return True iff this ORT build actually dispatches the cuDNN SDPA decode tier on this machine.

    Instead of re-implementing the kernel's cuDNN version rule in Python (which would drift from the
    C++ cudnn_sdpa::is_stable gate and from whichever cuDNN ORT dlopened), this OBSERVES the real
    dispatch: it runs a minimal q_seq==1 decode graph once with the cuDNN kernel requested and the
    attention-kernel debug info enabled, and reports True only when the observed tier is
    CUDNN_FLASH_ATTENTION. The result is cached per head_size. Any failure (no CUDA provider,
    unsupported cuDNN, etc.) yields False so dependent tests skip cleanly rather than false-failing.
    """
    if head_size in _CUDNN_DECODE_SUPPORTED_CACHE:
        return _CUDNN_DECODE_SUPPORTED_CACHE[head_size]
    supported = False
    if has_cuda_provider():
        try:
            supported = _observe_cudnn_decode_dispatch(head_size) == "CUDNN_FLASH_ATTENTION"
        except Exception:
            supported = False
    _CUDNN_DECODE_SUPPORTED_CACHE[head_size] = supported
    return supported


def require_cudnn_sdpa():
    """Return True when the environment demands the cuDNN SDPA decode tier be dispatched (CI gate).

    On a known-good GPU CI leg the operator sets ORT_TEST_REQUIRE_CUDNN_SDPA=1. When set, the decode
    dispatch assertions become NON-skippable: a MATH fallback / non-dispatch fails loudly instead of
    being hidden as an all-green skip by the observe-dispatch gating in cudnn_decode_supported().
    When unset, tests fall back to the normal cudnn_decode_supported() skip guard.
    """
    return os.environ.get("ORT_TEST_REQUIRE_CUDNN_SDPA") == "1"


def _run_capturing_sdpa_kernel(run_func):
    """Run run_func with attention-kernel debug info enabled and return (result, sdpa_kernel).

    Sets ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO before run_func creates its session (the option
    is initialized once per provider at session creation) and captures the native fd-1 output so
    the selected tier (SdpaKernel=...) can be asserted, following the test_gqa.py pattern.
    """
    previous = os.environ.get("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO")
    os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "1"
    try:
        with _CaptureStdout() as captured:
            result = run_func()
    finally:
        if previous is None:
            os.environ.pop("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO", None)
        else:
            os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = previous
    return result, _parse_sdpa_kernel(captured.output.decode(errors="replace"))


def has_cuda_provider():
    return "CUDAExecutionProvider" in get_available_providers()


def has_cuda_device(min_capability: int = 53):
    if not has_cuda_provider() or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= min_capability


def has_flash_attention():
    """Return True if the CUDA device meets the SM80+ requirement for Flash Attention."""
    return has_cuda_device(80)


def numpy_attention_ref(q, k, v, nonpad_kv_seqlen, is_causal=False, attn_bias=None):
    """
    NumPy reference implementation of scaled dot-product attention with padding mask.

    Args:
        q: Query [batch, q_seq, num_heads, head_size]
        k: Key [batch, kv_seq, kv_num_heads, head_size]
        v: Value [batch, kv_seq, kv_num_heads, head_size]
        nonpad_kv_seqlen: [batch] — number of valid KV positions per batch
        is_causal: whether to apply causal masking
        attn_bias: optional additive attention bias, broadcastable to [batch, num_heads, q_seq, kv_seq]

    Returns:
        output: [batch, q_seq, num_heads, head_size]
    """
    batch_size, q_seq, num_heads, head_size = q.shape
    _, kv_seq, kv_num_heads, _ = k.shape
    groups = num_heads // kv_num_heads

    # Repeat KV heads for GQA
    if groups > 1:
        k = numpy.repeat(k, groups, axis=2)
        v = numpy.repeat(v, groups, axis=2)

    scale = 1.0 / math.sqrt(head_size)

    # scores: [batch, num_heads, q_seq, kv_seq]
    q_t = numpy.transpose(q, (0, 2, 1, 3))
    k_t = numpy.transpose(k, (0, 2, 3, 1))
    scores = numpy.matmul(q_t, k_t) * scale

    # Apply nonpad_kv_seqlen mask: positions >= valid_len get -inf
    for b in range(batch_size):
        valid_len = int(nonpad_kv_seqlen[b])
        if valid_len < kv_seq:
            scores[b, :, :, valid_len:] = -numpy.inf

    # Apply additive attention bias (from attn_mask conversion)
    if attn_bias is not None:
        scores = scores + attn_bias

    # Apply causal mask
    if is_causal:
        # NOTE (Phase-3 caveat): this uses a CAPACITY-anchored bottom-right offset (kv_seq - q_seq).
        # It is exact for decode (q_seq == 1), where every batch's single query attends the whole
        # valid KV region regardless of nonpad[b]. If this reference is reused for prefill
        # (q_seq > 1) with heterogeneous nonpad lengths, it DIVERGES from the ONNX per-batch
        # frontier (nonpad[b] - q_seq) per onnx/onnx#8068 — switch to a per-batch offset there.
        for sq in range(q_seq):
            offset = kv_seq - q_seq
            for sk in range(kv_seq):
                if sk > sq + offset:
                    scores[:, :, sq, sk] = -numpy.inf

    # Softmax along last axis
    # Handle all-masked rows: if entire row is -inf, softmax gives nan; we want 0.
    # This happens when nonpad_kv_seqlen=0 for a batch (all KV positions masked).
    # Callers zero out those batches in both ORT and reference outputs for comparison.
    max_scores = numpy.max(scores, axis=-1, keepdims=True)
    # Clip -inf max to 0 to avoid nan in exp
    max_scores = numpy.where(numpy.isinf(max_scores) & (max_scores < 0), 0.0, max_scores)
    exp_scores = numpy.exp(scores - max_scores)
    sum_exp = numpy.sum(exp_scores, axis=-1, keepdims=True)
    sum_exp = numpy.where(sum_exp == 0.0, 1.0, sum_exp)
    attention = exp_scores / sum_exp

    # output: [batch, num_heads, q_seq, head_size]
    v_t = numpy.transpose(v, (0, 2, 1, 3))
    output = numpy.matmul(attention, v_t)

    # Transpose back: [batch, q_seq, num_heads, head_size]
    output = numpy.transpose(output, (0, 2, 1, 3))
    return output


def build_tensorscatter_attention_graph(
    batch_size,
    total_kv_seq_len,
    q_seq_len,
    q_num_heads,
    kv_num_heads,
    head_size,
    ort_type,
    is_causal=0,
    use_4d=False,
):
    """
    Build ONNX graph: TensorScatter(opset 24) → Attention(opset 24).

    TensorScatter uses write_indices [B] to scatter new KV entries into cache
    at per-batch positions. Attention uses updated cache with nonpad_kv_seqlen
    to mask padding.

    The graph exposes updated_key_cache and updated_value_cache as graph outputs
    to enable in-place buffer binding via IO Binding.

    Layout (use_4d selects the Attention op input rank):
      - use_4d=False → 3-D BSNH: Q/caches are [B, S, N*H]; TensorScatter axis=1.
        The Attention op sets transpose_output=True (attention_helper.h) and takes the
        3-D-only decode path.
      - use_4d=True  → 4-D BNSH: Q is [B, q_heads, S, head_size], caches are
        [B, kv_heads, S, head_size]; TensorScatter axis=2. The Attention op sets
        transpose_output=False and exercises the 4-D path in RunCudnnSdpaAttention
        (Q transpose, Q_K_V_BSNH_BNSH_BNSH cuDNN layout, output transpose, and the
        device-to-device present-cache copies). present_key/value stay BNSH either way.

    3-D inputs / outputs:
      0: key_cache        [B, total_kv_seq_len, kv_hidden]
      1: value_cache      [B, total_kv_seq_len, kv_hidden]
      2: new_k            [B, q_seq_len, kv_hidden]
      3: new_v            [B, q_seq_len, kv_hidden]
      4: write_indices    [B]   (int64 — per-batch write position)
      5: query            [B, q_seq_len, q_hidden]
      6: nonpad_kv_seqlen [B]   (int64 — valid KV length after scatter)
      out: output [B, q_seq_len, q_hidden], present_key/value [B, kv_num_heads, total_kv_seq_len, head_size]

    4-D inputs / outputs (BNSH): caches [B, kv_num_heads, total_kv_seq_len, head_size],
      new_k/new_v [B, kv_num_heads, q_seq_len, head_size], query [B, q_num_heads, q_seq_len, head_size],
      output [B, q_num_heads, q_seq_len, head_size], present_key/value [B, kv_num_heads, total_kv_seq_len, head_size].
    """
    kv_hidden = kv_num_heads * head_size
    q_hidden = q_num_heads * head_size

    if use_4d:
        # BNSH: sequence dimension is axis=2 (axis=-2 in the TensorScatter spec).
        scatter_axis = 2
        cache_shape = [batch_size, kv_num_heads, total_kv_seq_len, head_size]
        update_shape = [batch_size, kv_num_heads, q_seq_len, head_size]
        query_shape = [batch_size, q_num_heads, q_seq_len, head_size]
        output_shape = [batch_size, q_num_heads, q_seq_len, head_size]
    else:
        # BSNH: sequence dimension is axis=1.
        scatter_axis = 1
        cache_shape = [batch_size, total_kv_seq_len, kv_hidden]
        update_shape = [batch_size, q_seq_len, kv_hidden]
        query_shape = [batch_size, q_seq_len, q_hidden]
        output_shape = [batch_size, q_seq_len, q_hidden]

    # TensorScatter for key cache update (sequence dim is axis=1 for 3-D BSNH, axis=2 for 4-D BNSH)
    scatter_k_node = helper.make_node(
        "TensorScatter",
        inputs=["key_cache", "new_k", "write_indices"],
        outputs=["updated_key_cache"],
        name="TensorScatterKey",
        axis=scatter_axis,
    )

    # TensorScatter for value cache update
    scatter_v_node = helper.make_node(
        "TensorScatter",
        inputs=["value_cache", "new_v", "write_indices"],
        outputs=["updated_value_cache"],
        name="TensorScatterValue",
        axis=scatter_axis,
    )

    # Attention node with nonpad_kv_seqlen
    attention_node = helper.make_node(
        "Attention",
        inputs=[
            "query",
            "updated_key_cache",
            "updated_value_cache",
            "",  # attn_mask
            "",  # past_key
            "",  # past_value
            "nonpad_kv_seqlen",
        ],
        outputs=["output", "present_key", "present_value"],
        name="Attention_0",
        is_causal=is_causal,
        kv_num_heads=kv_num_heads,
        q_num_heads=q_num_heads,
        softcap=0.0,
        qk_matmul_output_mode=0,
        domain="",
    )

    # Graph inputs
    graph_inputs = [
        helper.make_tensor_value_info("key_cache", ort_type, cache_shape),
        helper.make_tensor_value_info("value_cache", ort_type, cache_shape),
        helper.make_tensor_value_info("new_k", ort_type, update_shape),
        helper.make_tensor_value_info("new_v", ort_type, update_shape),
        helper.make_tensor_value_info("write_indices", TensorProto.INT64, [batch_size]),
        helper.make_tensor_value_info("query", ort_type, query_shape),
        helper.make_tensor_value_info("nonpad_kv_seqlen", TensorProto.INT64, [batch_size]),
    ]

    # Graph outputs: Attention outputs + TensorScatter outputs for in-place binding.
    # present_key/value are BNSH regardless of the input layout.
    present_shape = [batch_size, kv_num_heads, total_kv_seq_len, head_size]
    graph_outputs = [
        helper.make_tensor_value_info("output", ort_type, output_shape),
        helper.make_tensor_value_info("present_key", ort_type, present_shape),
        helper.make_tensor_value_info("present_value", ort_type, present_shape),
        helper.make_tensor_value_info("updated_key_cache", ort_type, cache_shape),
        helper.make_tensor_value_info("updated_value_cache", ort_type, cache_shape),
    ]

    graph = helper.make_graph(
        [scatter_k_node, scatter_v_node, attention_node],
        "TensorScatterAttention_Graph",
        graph_inputs,
        graph_outputs,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])
    return model.SerializeToString()


def run_tensorscatter_attention(
    batch_size,
    total_kv_seq_len,
    q_seq_len,
    q_num_heads,
    kv_num_heads,
    head_size,
    nonpad_seqlens,
    scatter_positions,
    ep,
    torch_type,
    ort_type,
    is_causal=0,
    std=0.2,
    provider_options=None,
    use_4d=False,
):
    """
    Run TensorScatter + Attention test with IO Binding and compare against NumPy reference.

    Uses IO Binding to:
    1. Pre-allocate KV cache as OrtValues on the target device
    2. Bind the same OrtValue as both TensorScatter input and output (in-place update)
    3. Feed the updated cache to Attention
    4. Pre-allocate output buffers on the target device

    Args:
        scatter_positions: list of ints per batch — the write index for TensorScatter.
        nonpad_seqlens: list of ints per batch — valid KV length AFTER scatter.
        is_causal: 1 for causal attention, 0 for non-causal.
        provider_options: optional single-provider options dict (e.g. {"sdpa_kernel": "24"})
            applied to the InferenceSession for `ep`. Pass a plain dict, not a list — it is
            wrapped in a one-element list internally to match the single `providers=[ep]` entry.
        use_4d: when True, feed 4-D BNSH inputs (query [B, q_heads, S, head_size], caches
            [B, kv_heads, S, head_size]) so the Attention op takes the 4-D path
            (transpose_output=False) instead of the 3-D BSNH path. present_key/value stay BNSH
            in both layouts, so those assertions are unchanged.
    """
    torch.manual_seed(42)
    kv_hidden = kv_num_heads * head_size
    q_hidden = q_num_heads * head_size
    is_bf16 = ort_type == TensorProto.BFLOAT16
    np_type = numpy.float16 if torch_type == torch.float16 else numpy.float32

    def _randn(*shape):
        return torch.randn(*shape, dtype=torch_type) * std

    # Generate test data as torch tensors seeded for reproducibility. The 4-D BNSH layout places
    # heads on axis 1 and the sequence on axis 2; the 3-D BSNH layout flattens N*H into the last dim
    # with the sequence on axis 1.
    if use_4d:
        key_cache_t = _randn(batch_size, kv_num_heads, total_kv_seq_len, head_size)
        value_cache_t = _randn(batch_size, kv_num_heads, total_kv_seq_len, head_size)
    else:
        key_cache_t = _randn(batch_size, total_kv_seq_len, kv_hidden)
        value_cache_t = _randn(batch_size, total_kv_seq_len, kv_hidden)

    # Zero out padding positions in cache (sequence dim is axis=2 for BNSH, axis=1 for BSNH).
    for b in range(batch_size):
        old_valid = max(0, nonpad_seqlens[b] - q_seq_len)
        if old_valid < total_kv_seq_len:
            if use_4d:
                key_cache_t[b, :, old_valid:, :] = 0
                value_cache_t[b, :, old_valid:, :] = 0
            else:
                key_cache_t[b, old_valid:, :] = 0
                value_cache_t[b, old_valid:, :] = 0

    if use_4d:
        new_k_t = _randn(batch_size, kv_num_heads, q_seq_len, head_size)
        new_v_t = _randn(batch_size, kv_num_heads, q_seq_len, head_size)
        query_t = _randn(batch_size, q_num_heads, q_seq_len, head_size)
    else:
        new_k_t = _randn(batch_size, q_seq_len, kv_hidden)
        new_v_t = _randn(batch_size, q_seq_len, kv_hidden)
        query_t = _randn(batch_size, q_seq_len, q_hidden)

    write_indices_np = numpy.array(scatter_positions, dtype=numpy.int64)
    nonpad_kv_seqlen_np = numpy.array(nonpad_seqlens, dtype=numpy.int64)

    # --- NumPy reference ---
    # Compute reference in float32 from the rounded storage values, so only the compute precision
    # differs across dtypes (bf16 has no native numpy dtype; .float() upcasts the rounded values).
    key_cache_ref = key_cache_t.float().cpu().numpy().copy()
    value_cache_ref = value_cache_t.float().cpu().numpy().copy()
    new_k_ref = new_k_t.float().cpu().numpy()
    new_v_ref = new_v_t.float().cpu().numpy()

    if use_4d:
        # BNSH scatter: write at sequence axis=2 across all heads.
        for b in range(batch_size):
            pos = scatter_positions[b]
            for t in range(q_seq_len):
                key_cache_ref[b, :, pos + t, :] = new_k_ref[b, :, t, :]
                value_cache_ref[b, :, pos + t, :] = new_v_ref[b, :, t, :]

        # numpy_attention_ref expects BSNH; transpose BNSH -> BSNH for Q/K/V.
        q_ref = query_t.float().cpu().numpy().transpose(0, 2, 1, 3)
        k_ref = key_cache_ref.transpose(0, 2, 1, 3)
        v_ref = value_cache_ref.transpose(0, 2, 1, 3)

        ref_output_bsnh = numpy_attention_ref(q_ref, k_ref, v_ref, nonpad_seqlens, is_causal=bool(is_causal))
        # ORT 4-D output is BNSH; transpose the BSNH reference back to [B, q_heads, S, head_size].
        ref_output_arr = ref_output_bsnh.transpose(0, 2, 1, 3)

        # present_key/value are BNSH; the updated cache is already BNSH.
        ref_present_k = key_cache_ref
        ref_present_v = value_cache_ref
    else:
        for b in range(batch_size):
            pos = scatter_positions[b]
            for t in range(q_seq_len):
                key_cache_ref[b, pos + t, :] = new_k_ref[b, t, :]
                value_cache_ref[b, pos + t, :] = new_v_ref[b, t, :]

        # Reshape to BSNH for reference attention
        q_ref = query_t.float().cpu().numpy().reshape(batch_size, q_seq_len, q_num_heads, head_size)
        k_ref = key_cache_ref.reshape(batch_size, total_kv_seq_len, kv_num_heads, head_size)
        v_ref = value_cache_ref.reshape(batch_size, total_kv_seq_len, kv_num_heads, head_size)

        ref_output = numpy_attention_ref(q_ref, k_ref, v_ref, nonpad_seqlens, is_causal=bool(is_causal))
        ref_output_arr = ref_output.reshape(batch_size, q_seq_len, q_hidden)

        # Compute expected present_key/present_value: BSNH → BNSH transpose of updated cache.
        # Attention op with no past_key simply reshapes+transposes K/V to [B, H, S, D].
        ref_present_k = k_ref.transpose(0, 2, 1, 3)  # [B, kv_num_heads, total_kv_seq_len, head_size]
        ref_present_v = v_ref.transpose(0, 2, 1, 3)

    # --- ORT execution with IO Binding ---
    onnx_model_str = build_tensorscatter_attention_graph(
        batch_size=batch_size,
        total_kv_seq_len=total_kv_seq_len,
        q_seq_len=q_seq_len,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        head_size=head_size,
        ort_type=ort_type,
        is_causal=is_causal,
        use_4d=use_4d,
    )

    sess_options = SessionOptions()
    if provider_options is not None:
        session = InferenceSession(onnx_model_str, sess_options, providers=[ep], provider_options=[provider_options])
    else:
        session = InferenceSession(onnx_model_str, sess_options, providers=[ep])

    # Determine device for OrtValue allocation
    ort_device = "cuda" if "CUDA" in ep else "cpu"
    device_id = 0

    present_shape = [batch_size, kv_num_heads, total_kv_seq_len, head_size]
    output_shape = [batch_size, q_num_heads, q_seq_len, head_size] if use_4d else [batch_size, q_seq_len, q_hidden]

    if is_bf16:
        # numpy has no native bfloat16, so (matching test_gqa.py) bf16 tensors live as torch tensors
        # on the device and are bound via their raw data_ptr with an explicit BFLOAT16 tag; the
        # binding C-API takes an explicit element_type + pointer and never inspects the tensor dtype.
        # The device tensors MUST stay alive for the whole run so their buffers are not freed while
        # ORT holds the raw pointers.
        key_cache_dev = key_cache_t.to(ort_device)
        value_cache_dev = value_cache_t.to(ort_device)
        new_k_dev = new_k_t.to(ort_device)
        new_v_dev = new_v_t.to(ort_device)
        query_dev = query_t.to(ort_device)
        write_indices_dev = torch.from_numpy(write_indices_np).to(ort_device)
        nonpad_dev = torch.from_numpy(nonpad_kv_seqlen_np).to(ort_device)

        output_dev = torch.zeros(tuple(output_shape), dtype=torch.bfloat16, device=ort_device)
        present_k_dev = torch.zeros(tuple(present_shape), dtype=torch.bfloat16, device=ort_device)
        present_v_dev = torch.zeros(tuple(present_shape), dtype=torch.bfloat16, device=ort_device)

        io_binding = session.io_binding()
        io_binding.bind_input(
            "key_cache",
            ort_device,
            device_id,
            TensorProto.BFLOAT16,
            tuple(key_cache_dev.shape),
            key_cache_dev.data_ptr(),
        )
        io_binding.bind_input(
            "value_cache",
            ort_device,
            device_id,
            TensorProto.BFLOAT16,
            tuple(value_cache_dev.shape),
            value_cache_dev.data_ptr(),
        )
        io_binding.bind_input(
            "new_k", ort_device, device_id, TensorProto.BFLOAT16, tuple(new_k_dev.shape), new_k_dev.data_ptr()
        )
        io_binding.bind_input(
            "new_v", ort_device, device_id, TensorProto.BFLOAT16, tuple(new_v_dev.shape), new_v_dev.data_ptr()
        )
        io_binding.bind_input(
            "write_indices",
            ort_device,
            device_id,
            TensorProto.INT64,
            tuple(write_indices_dev.shape),
            write_indices_dev.data_ptr(),
        )
        io_binding.bind_input(
            "query", ort_device, device_id, TensorProto.BFLOAT16, tuple(query_dev.shape), query_dev.data_ptr()
        )
        io_binding.bind_input(
            "nonpad_kv_seqlen",
            ort_device,
            device_id,
            TensorProto.INT64,
            tuple(nonpad_dev.shape),
            nonpad_dev.data_ptr(),
        )

        io_binding.bind_output(
            "output", ort_device, device_id, TensorProto.BFLOAT16, tuple(output_shape), output_dev.data_ptr()
        )
        io_binding.bind_output(
            "present_key", ort_device, device_id, TensorProto.BFLOAT16, tuple(present_shape), present_k_dev.data_ptr()
        )
        io_binding.bind_output(
            "present_value", ort_device, device_id, TensorProto.BFLOAT16, tuple(present_shape), present_v_dev.data_ptr()
        )
        # In-place TensorScatter: bind the updated-cache outputs to the SAME device buffers as the
        # cache inputs (TensorScatter declares MayInplace(0, 0)).
        io_binding.bind_output(
            "updated_key_cache",
            ort_device,
            device_id,
            TensorProto.BFLOAT16,
            tuple(key_cache_dev.shape),
            key_cache_dev.data_ptr(),
        )
        io_binding.bind_output(
            "updated_value_cache",
            ort_device,
            device_id,
            TensorProto.BFLOAT16,
            tuple(value_cache_dev.shape),
            value_cache_dev.data_ptr(),
        )

        io_binding.synchronize_inputs()
        session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

        # Read back to float32 (mirroring the fp16/fp32 upcast used for the reference).
        output_result = output_dev.cpu().float().numpy()
        present_k_result = present_k_dev.cpu().float().numpy()
        present_v_result = present_v_dev.cpu().float().numpy()
        # Release the session and device buffers promptly so GPU memory does not accumulate across
        # the parameterized cases.
        del io_binding, session
        gc.collect()
        return output_result, ref_output_arr, present_k_result, present_v_result, ref_present_k, ref_present_v

    # fp16 / fp32 path: stage inputs as numpy arrays and let OrtValue own the device copies.
    key_cache_np = key_cache_t.cpu().numpy()
    value_cache_np = value_cache_t.cpu().numpy()
    new_k_np = new_k_t.cpu().numpy()
    new_v_np = new_v_t.cpu().numpy()
    query_np = query_t.cpu().numpy()

    # Create OrtValues for inputs on target device
    key_cache_ort = OrtValue.ortvalue_from_numpy(key_cache_np, ort_device, device_id)
    value_cache_ort = OrtValue.ortvalue_from_numpy(value_cache_np, ort_device, device_id)
    new_k_ort = OrtValue.ortvalue_from_numpy(new_k_np, ort_device, device_id)
    new_v_ort = OrtValue.ortvalue_from_numpy(new_v_np, ort_device, device_id)
    write_indices_ort = OrtValue.ortvalue_from_numpy(write_indices_np, ort_device, device_id)
    query_ort = OrtValue.ortvalue_from_numpy(query_np, ort_device, device_id)
    nonpad_ort = OrtValue.ortvalue_from_numpy(nonpad_kv_seqlen_np, ort_device, device_id)

    # Pre-allocate output buffers on target device. present_key/value are BNSH regardless of layout;
    # the attention output matches the query rank (4-D BNSH when use_4d, else 3-D BSNH).
    output_ort = OrtValue.ortvalue_from_shape_and_type(output_shape, np_type, ort_device, device_id)
    present_k_ort = OrtValue.ortvalue_from_shape_and_type(present_shape, np_type, ort_device, device_id)
    present_v_ort = OrtValue.ortvalue_from_shape_and_type(present_shape, np_type, ort_device, device_id)

    # Set up IO binding
    io_binding = session.io_binding()

    # Bind all inputs
    io_binding.bind_ortvalue_input("key_cache", key_cache_ort)
    io_binding.bind_ortvalue_input("value_cache", value_cache_ort)
    io_binding.bind_ortvalue_input("new_k", new_k_ort)
    io_binding.bind_ortvalue_input("new_v", new_v_ort)
    io_binding.bind_ortvalue_input("write_indices", write_indices_ort)
    io_binding.bind_ortvalue_input("query", query_ort)
    io_binding.bind_ortvalue_input("nonpad_kv_seqlen", nonpad_ort)

    # Bind Attention outputs to pre-allocated buffers
    io_binding.bind_ortvalue_output("output", output_ort)
    io_binding.bind_ortvalue_output("present_key", present_k_ort)
    io_binding.bind_ortvalue_output("present_value", present_v_ort)

    # Bind TensorScatter outputs to the SAME OrtValues as inputs (in-place update).
    # TensorScatter declares MayInplace(0, 0), so ORT will skip the copy when
    # input and output share the same buffer.
    io_binding.bind_ortvalue_output("updated_key_cache", key_cache_ort)
    io_binding.bind_ortvalue_output("updated_value_cache", value_cache_ort)

    # Execute with IO binding
    io_binding.synchronize_inputs()
    session.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()

    # Read results from pre-bound OrtValues
    output_result = output_ort.numpy()
    present_k_result = present_k_ort.numpy()
    present_v_result = present_v_ort.numpy()

    # Release the session and device buffers promptly so GPU memory does not accumulate across the
    # parameterized cases.
    del io_binding, session
    gc.collect()
    return output_result, ref_output_arr, present_k_result, present_v_result, ref_present_k, ref_present_v


# #################################################################################################
#  Test Case Generator
# #################################################################################################

# Shared test dimensions
_HEAD_SIZE = 64
_TOTAL_KV_SEQ_LEN = 8

_GQA_CASES = [
    # (batch, q_seq, q_heads, kv_heads, scatter_positions, nonpad_seqlens, label)
    (1, 1, 8, 2, [3], [4], "gqa_batch1"),
    (2, 1, 8, 2, [2, 4], [3, 5], "gqa_diff_lens"),
    (2, 1, 8, 2, [4, 4], [5, 5], "gqa_same_lens"),
    (2, 1, 8, 2, [0, 3], [1, 4], "gqa_one_empty"),
    (2, 1, 8, 2, [7, 7], [8, 8], "gqa_full_len"),
    # Additional GQA ratios
    (2, 1, 16, 4, [2, 5], [3, 6], "gqa_16h_4kvh"),
    (2, 1, 6, 3, [3, 3], [4, 4], "gqa_6h_3kvh"),
]

_MHA_CASES = [
    (1, 1, 4, 4, [3], [4], "mha_batch1"),
    (2, 1, 4, 4, [2, 4], [3, 5], "mha_diff_lens"),
    (2, 1, 4, 4, [4, 4], [5, 5], "mha_same_lens"),
    (2, 1, 4, 4, [0, 3], [1, 4], "mha_one_empty"),
    (2, 1, 4, 4, [7, 7], [8, 8], "mha_full_len"),
]


def _make_test_params(cases, is_causal):
    """Convert raw case tuples into parameterized test parameter tuples."""
    causal_str = "causal" if is_causal else "noncausal"
    for batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, label in cases:
        name = f"b{batch}_qs{q_seq}_qh{q_heads}_kvh{kv_heads}_h{_HEAD_SIZE}_{label}_{causal_str}"
        yield (
            name,
            batch,
            q_seq,
            q_heads,
            kv_heads,
            _HEAD_SIZE,
            _TOTAL_KV_SEQ_LEN,
            scatter_pos,
            seqlens,
            is_causal,
        )


def cpu_test_cases():
    """CPU: all modes, non-causal and causal (both GQA and MHA work without restrictions)."""
    yield from _make_test_params(_GQA_CASES + _MHA_CASES, is_causal=0)
    yield from _make_test_params(_GQA_CASES + _MHA_CASES, is_causal=1)


def cuda_fp16_test_cases():
    """CUDA fp16: both GQA and MHA cases. Flash attention handles external KV cache directly.
    TensorScatter manages KV cache externally with nonpad_kv_seqlen bounding the active range.
    Per ONNX spec, is_causal with S_q!=S_kv and no past_key gives upper-left alignment
    (q[0] sees only kv[0]), which is not meaningful for decode. KV bounds are enforced by
    nonpad_kv_seqlen instead, so is_causal=0 is the correct setting for TensorScatter decode."""
    yield from _make_test_params(_GQA_CASES + _MHA_CASES, is_causal=0)


def cuda_fp32_test_cases():
    """CUDA fp32: MHA only. GQA requires fp16/bf16, and flash attention requires fp16/bf16.
    fp32 MHA uses the unfused attention_bias fallback path.
    TensorScatter manages KV cache externally with nonpad_kv_seqlen bounding the active range.
    Per ONNX spec, is_causal with S_q!=S_kv and no past_key gives upper-left alignment
    (q[0] sees only kv[0]), which is not meaningful for decode. KV bounds are enforced by
    nonpad_kv_seqlen instead, so is_causal=0 is the correct setting for TensorScatter decode."""
    yield from _make_test_params(_MHA_CASES, is_causal=0)


# #################################################################################################
#  Test Classes
# #################################################################################################

# Default tolerances (CUDA fp16/fp32 need looser tolerances due to TF32 and reduced precision)
rtol = {"fp16": 5e-3, "fp32": 5e-3, "bf16": 2e-2}
atol = {"fp16": 5e-3, "fp32": 5e-3, "bf16": 2e-2}
# CPU fp32 has no TF32 — use tighter tolerance
cpu_fp32_rtol = 1e-5
cpu_fp32_atol = 1e-5


class TestTensorScatterAttentionCPU(unittest.TestCase):
    """Test TensorScatter + Attention (opset 24) on CPU with float32 and IO Binding."""

    @parameterized.expand(cpu_test_cases())
    def test_tensorscatter_attention_cpu_fp32(
        self,
        name,
        batch,
        q_seq,
        q_heads,
        kv_heads,
        head_size,
        total_kv,
        scatter_pos,
        seqlens,
        is_causal,
    ):
        output, ref_output, present_k, present_v, ref_present_k, ref_present_v = run_tensorscatter_attention(
            batch_size=batch,
            total_kv_seq_len=total_kv,
            q_seq_len=q_seq,
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            head_size=head_size,
            nonpad_seqlens=seqlens,
            scatter_positions=scatter_pos,
            ep="CPUExecutionProvider",
            torch_type=torch.float32,
            ort_type=TensorProto.FLOAT,
            is_causal=is_causal,
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=cpu_fp32_rtol, atol=cpu_fp32_atol)
        numpy.testing.assert_allclose(present_k, ref_present_k, rtol=cpu_fp32_rtol, atol=cpu_fp32_atol)
        numpy.testing.assert_allclose(present_v, ref_present_v, rtol=cpu_fp32_rtol, atol=cpu_fp32_atol)


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestTensorScatterAttentionCUDAFP16(unittest.TestCase):
    """Test TensorScatter + Attention (opset 24) on CUDA with float16 and IO Binding.

    On SM80+ Flash Attention is used; on SM75+ MEA handles the fallback;
    on older GPUs the unfused path runs.  The cascade in attention.cc picks
    the best available backend automatically.
    """

    @parameterized.expand(cuda_fp16_test_cases())
    def test_tensorscatter_attention_cuda_fp16(
        self,
        name,
        batch,
        q_seq,
        q_heads,
        kv_heads,
        head_size,
        total_kv,
        scatter_pos,
        seqlens,
        is_causal,
    ):
        output, ref_output, present_k, present_v, ref_present_k, ref_present_v = run_tensorscatter_attention(
            batch_size=batch,
            total_kv_seq_len=total_kv,
            q_seq_len=q_seq,
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            head_size=head_size,
            nonpad_seqlens=seqlens,
            scatter_positions=scatter_pos,
            ep="CUDAExecutionProvider",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            is_causal=is_causal,
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_allclose(present_k, ref_present_k, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_allclose(present_v, ref_present_v, rtol=rtol["fp16"], atol=atol["fp16"])


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestTensorScatterAttentionCUDAFP32(unittest.TestCase):
    """Test TensorScatter + Attention (opset 24) on CUDA with float32 and IO Binding.

    Only MHA cases: CUDA GQA path requires float16.
    """

    @parameterized.expand(cuda_fp32_test_cases())
    def test_tensorscatter_attention_cuda_fp32(
        self,
        name,
        batch,
        q_seq,
        q_heads,
        kv_heads,
        head_size,
        total_kv,
        scatter_pos,
        seqlens,
        is_causal,
    ):
        output, ref_output, present_k, present_v, ref_present_k, ref_present_v = run_tensorscatter_attention(
            batch_size=batch,
            total_kv_seq_len=total_kv,
            q_seq_len=q_seq,
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            head_size=head_size,
            nonpad_seqlens=seqlens,
            scatter_positions=scatter_pos,
            ep="CUDAExecutionProvider",
            torch_type=torch.float32,
            ort_type=TensorProto.FLOAT,
            is_causal=is_causal,
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp32"], atol=atol["fp32"])
        numpy.testing.assert_allclose(present_k, ref_present_k, rtol=rtol["fp32"], atol=atol["fp32"])
        numpy.testing.assert_allclose(present_v, ref_present_v, rtol=rtol["fp32"], atol=atol["fp32"])


# cuDNN SDPA decode tier (Phase 1, issue #29714). Forces the cuDNN kernel via the sdpa_kernel
# provider option (CUDNN_FLASH_ATTENTION=8 | MATH=16 fallback) so the gated external-cache decode
# path (nonpad_kv_seqlen, q_seq==1, fp16; both is_causal=0 and is_causal=1) routes to cuDNN when
# supported and falls back to the unfused kernel otherwise. Both produce spec-equivalent output,
# so this asserts numeric parity either way — in particular the fully-masked-batch (nonpad==0)
# zero-fill guard, which cuDNN needs but the other tiers get for free.
_CUDNN_DECODE_HEAD_SIZE = 64
_CUDNN_DECODE_TOTAL_KV = 8

# sdpa_kernel bitmask (AttentionBackend in attention_common.h): select cuDNN and keep the unfused
# kernel as a fallback for configs where cuDNN is unsupported.
_SDPA_KERNEL_CUDNN_FLASH_ATTENTION = 8
_SDPA_KERNEL_MATH = 16
_SDPA_KERNEL_CUDNN_WITH_MATH_FALLBACK = _SDPA_KERNEL_CUDNN_FLASH_ATTENTION | _SDPA_KERNEL_MATH

_CUDNN_DECODE_CASES = [
    # (batch, q_seq, q_heads, kv_heads, scatter_positions, nonpad_seqlens, label)
    (1, 1, 8, 8, [3], [4], "mha_batch1"),
    (2, 1, 8, 8, [2, 4], [3, 5], "mha_diff_lens"),
    (1, 1, 8, 1, [5], [6], "mqa_batch1"),
    (2, 1, 8, 2, [2, 4], [3, 5], "gqa_diff_lens"),
    (2, 1, 16, 4, [2, 5], [3, 6], "gqa_16h_4kvh"),
    # Fully-masked batch (nonpad_kv_seqlen[b] == 0): guards the LaunchZeroOutputForFullyMaskedBatches
    # call — cuDNN would otherwise emit NaN for that row while the reference (and every other tier)
    # emit 0.
    (2, 1, 8, 2, [0, 4], [0, 5], "gqa_fully_masked_b0"),
    # Heterogeneous valid lengths across the batch, including one at full capacity.
    (3, 1, 16, 4, [2, 7, 3], [3, 8, 4], "gqa_heterogeneous"),
    (2, 1, 8, 8, [7, 7], [8, 8], "mha_full_len"),
]

# 4-D BNSH cases (M2): the eligibility gate has no rank restriction, so 4-D BNSH inputs reach the
# cuDNN decode tier in production just like 3-D BSNH. These exercise the 4-D-only branches in
# RunCudnnSdpaAttention that the 3-D harness never touches: the input Q transpose (BNSH->BSNH), the
# mixed Q_K_V_BSNH_BNSH_BNSH cuDNN layout, the output transpose (BSNH->BNSH), and the
# device-to-device present-cache copies. A representative subset (MHA, GQA, fully-masked, full-len)
# is enough to cover those code paths without doubling the whole 3-D matrix.
_CUDNN_DECODE_4D_CASES = [
    (1, 1, 8, 8, [3], [4], "mha_batch1_4d"),
    (2, 1, 8, 2, [2, 4], [3, 5], "gqa_diff_lens_4d"),
    # Fully-masked batch under 4-D: guards the device-side zero-fill on the 4-D output layout.
    (2, 1, 8, 2, [0, 4], [0, 5], "gqa_fully_masked_b0_4d"),
    (2, 1, 8, 8, [7, 7], [8, 8], "mha_full_len_4d"),
]


def cudnn_decode_test_cases():
    """cuDNN SDPA decode cases: single-token (q_seq==1), external KV cache.

    Exercised for BOTH is_causal=0 and is_causal=1. For s_q==1 cuDNN drops causal masking, so the
    two collapse to the identical padding-only frontier — is_causal=0 (the repo's documented
    decode contract) must select cuDNN just like is_causal=1, which the eligibility gate now allows.
    Both 3-D BSNH (use_4d=False) and 4-D BNSH (use_4d=True) layouts are exercised.
    """
    for is_causal in (0, 1):
        causal_str = "causal" if is_causal else "noncausal"
        for use_4d, cases in ((False, _CUDNN_DECODE_CASES), (True, _CUDNN_DECODE_4D_CASES)):
            for batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, label in cases:
                name = f"b{batch}_qh{q_heads}_kvh{kv_heads}_h{_CUDNN_DECODE_HEAD_SIZE}_{causal_str}_{label}"
                yield (name, batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, is_causal, use_4d)


# bf16 decode cases (M4): the eligibility gate admits kIsFp16OrBf16, so bf16 reaches the cuDNN decode
# tier. A representative subset plus one 4-D case covers the bf16 dtype across both layouts. Tuple:
# (batch, q_seq, q_heads, kv_heads, scatter, nonpad, use_4d, label).
_CUDNN_DECODE_BF16_CASES = [
    (1, 1, 8, 8, [3], [4], False, "mha_batch1_bf16"),
    (2, 1, 8, 2, [2, 4], [3, 5], False, "gqa_diff_lens_bf16"),
    # Fully-masked batch (nonpad==0): guards the device-side zero-fill on the bf16 output.
    (2, 1, 8, 2, [0, 4], [0, 5], False, "gqa_fully_masked_b0_bf16"),
    (2, 1, 8, 8, [7, 7], [8, 8], False, "mha_full_len_bf16"),
    # bf16 x 4-D BNSH intersection.
    (2, 1, 8, 2, [2, 4], [3, 5], True, "gqa_diff_lens_4d_bf16"),
]


def cudnn_decode_bf16_test_cases():
    """bf16 cuDNN SDPA decode cases (q_seq==1, external KV cache), for is_causal in {0, 1}."""
    for is_causal in (0, 1):
        causal_str = "causal" if is_causal else "noncausal"
        for batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, use_4d, label in _CUDNN_DECODE_BF16_CASES:
            name = f"b{batch}_qh{q_heads}_kvh{kv_heads}_h{_CUDNN_DECODE_HEAD_SIZE}_{causal_str}_{label}"
            yield (name, batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, is_causal, use_4d)


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestTensorScatterAttentionCudnnSdpaDecode(unittest.TestCase):
    """Force the cuDNN SDPA decode tier for the opset-24 external-cache decode path (fp16).

    cuDNN actually runs when SM>=80 and cuDNN>=9.3 (SM>=90 is only the auto-enable heuristic, not a
    hard requirement for the forced path). On such configs (cudnn_decode_supported) the test asserts
    routing landed on CUDNN_FLASH_ATTENTION via the AttentionKernelDebugInfo hook, so a broken cuDNN
    path / fully-masked guard cannot silently pass on the MATH fallback. On other configurations the
    sdpa_kernel selection falls back to the unfused kernel and only the (still spec-equivalent)
    parity assertions apply.
    """

    @parameterized.expand(cudnn_decode_test_cases())
    def test_tensorscatter_attention_cudnn_decode_fp16(
        self,
        name,
        batch,
        q_seq,
        q_heads,
        kv_heads,
        scatter_pos,
        seqlens,
        is_causal,
        use_4d,
    ):
        def run():
            return run_tensorscatter_attention(
                batch_size=batch,
                total_kv_seq_len=_CUDNN_DECODE_TOTAL_KV,
                q_seq_len=q_seq,
                q_num_heads=q_heads,
                kv_num_heads=kv_heads,
                head_size=_CUDNN_DECODE_HEAD_SIZE,
                nonpad_seqlens=seqlens,
                scatter_positions=scatter_pos,
                ep="CUDAExecutionProvider",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                is_causal=is_causal,
                provider_options={"sdpa_kernel": str(_SDPA_KERNEL_CUDNN_WITH_MATH_FALLBACK)},
                use_4d=use_4d,
            )

        (output, ref_output, present_k, present_v, ref_present_k, ref_present_v), sdpa_kernel = (
            _run_capturing_sdpa_kernel(run)
        )

        # On cuDNN-capable platforms the gated decode case MUST route to cuDNN (not silently fall
        # back to MATH), otherwise the cuDNN path and its fully-masked zero-fill guard go untested.
        # Under ORT_TEST_REQUIRE_CUDNN_SDPA=1 (CI gate on a known-good GPU) the assertion is
        # non-skippable so a tier regression fails loudly instead of hiding as an all-green skip.
        if require_cudnn_sdpa() or cudnn_decode_supported(_CUDNN_DECODE_HEAD_SIZE):
            self.assertEqual(
                "CUDNN_FLASH_ATTENTION",
                sdpa_kernel,
                f"Expected cuDNN SDPA decode tier on a cuDNN-capable platform, got {sdpa_kernel}",
            )

        # Fully-masked rows (nonpad==0) must be exactly 0 (no NaN) — assert finiteness explicitly.
        self.assertFalse(numpy.isnan(output).any(), "cuDNN SDPA decode produced NaN output")
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp16"], atol=atol["fp16"])
        # present_key/value are a pure scatter+copy of the storage-precision cache (no math), so they
        # must match the reference bit-for-bit after the shared float32 upcast — assert exactly.
        numpy.testing.assert_array_equal(present_k, ref_present_k)
        numpy.testing.assert_array_equal(present_v, ref_present_v)

    @parameterized.expand(cudnn_decode_bf16_test_cases())
    def test_tensorscatter_attention_cudnn_decode_bf16(
        self,
        name,
        batch,
        q_seq,
        q_heads,
        kv_heads,
        scatter_pos,
        seqlens,
        is_causal,
        use_4d,
    ):
        def run():
            return run_tensorscatter_attention(
                batch_size=batch,
                total_kv_seq_len=_CUDNN_DECODE_TOTAL_KV,
                q_seq_len=q_seq,
                q_num_heads=q_heads,
                kv_num_heads=kv_heads,
                head_size=_CUDNN_DECODE_HEAD_SIZE,
                nonpad_seqlens=seqlens,
                scatter_positions=scatter_pos,
                ep="CUDAExecutionProvider",
                torch_type=torch.bfloat16,
                ort_type=TensorProto.BFLOAT16,
                is_causal=is_causal,
                provider_options={"sdpa_kernel": str(_SDPA_KERNEL_CUDNN_WITH_MATH_FALLBACK)},
                use_4d=use_4d,
            )

        (output, ref_output, present_k, present_v, ref_present_k, ref_present_v), sdpa_kernel = (
            _run_capturing_sdpa_kernel(run)
        )

        # On cuDNN-capable platforms the gated bf16 decode case MUST route to cuDNN (not silently
        # fall back to MATH), otherwise the cuDNN bf16 path and its fully-masked guard go untested.
        # Under ORT_TEST_REQUIRE_CUDNN_SDPA=1 (CI gate on a known-good GPU) the assertion is
        # non-skippable so a tier regression fails loudly instead of hiding as an all-green skip.
        if require_cudnn_sdpa() or cudnn_decode_supported(_CUDNN_DECODE_HEAD_SIZE):
            self.assertEqual(
                "CUDNN_FLASH_ATTENTION",
                sdpa_kernel,
                f"Expected cuDNN SDPA decode tier on a cuDNN-capable platform, got {sdpa_kernel}",
            )

        # Fully-masked rows (nonpad==0) must be exactly 0 (no NaN) — assert finiteness explicitly.
        self.assertFalse(numpy.isnan(output).any(), "cuDNN SDPA decode produced NaN output (bf16)")
        # bf16 has ~8 bits of mantissa, so the attention output (matmul+softmax+matmul) is compared
        # with the looser rtol/atol["bf16"]=2e-2 accumulated-rounding tolerance.
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["bf16"], atol=atol["bf16"])
        # present_key/value are a pure scatter+copy of the storage-precision cache (no math), so they
        # must match the reference bit-for-bit after the shared float32 upcast — assert exactly.
        numpy.testing.assert_array_equal(present_k, ref_present_k)
        numpy.testing.assert_array_equal(present_v, ref_present_v)


class TestTensorScatterAttentionCudnnSdpaDecodeCanary(unittest.TestCase):
    """Canary that fails loudly if the cuDNN SDPA decode tier stops being dispatched.

    cudnn_decode_supported() gates every decode test by OBSERVING dispatch, which is precise but has
    a failure mode: if the tier silently regressed (stopped selecting cuDNN), the observation would
    return False and every decode test would skip green — hiding the regression as all-green skips.
    This canary closes that hole. On the GPU CI leg the operator sets ORT_TEST_REQUIRE_CUDNN_SDPA=1
    on a known-good GPU, and then a MATH fallback / non-dispatch on the minimal known-good config
    FAILS loudly instead of skipping. On dev boxes or unsupported cuDNN (env unset) it falls back to
    the normal cudnn_decode_supported() skip guard so it never false-alarms.
    """

    @unittest.skipIf(not has_cuda_provider(), "CUDA provider not available")
    def test_cudnn_sdpa_decode_tier_is_dispatched(self):
        enforce = require_cudnn_sdpa()
        if not enforce and not cudnn_decode_supported(_CUDNN_DECODE_HEAD_SIZE):
            self.skipTest("cuDNN SDPA decode tier not dispatched; set ORT_TEST_REQUIRE_CUDNN_SDPA=1 to enforce")

        # Known-good minimal decode config: q_seq==1, 3-D BSNH, fp16, small dims.
        def run():
            return run_tensorscatter_attention(
                batch_size=1,
                total_kv_seq_len=2,
                q_seq_len=1,
                q_num_heads=1,
                kv_num_heads=1,
                head_size=_CUDNN_DECODE_HEAD_SIZE,
                nonpad_seqlens=[1],
                scatter_positions=[0],
                ep="CUDAExecutionProvider",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                is_causal=0,
                provider_options={"sdpa_kernel": str(_SDPA_KERNEL_CUDNN_WITH_MATH_FALLBACK)},
            )

        _, sdpa_kernel = _run_capturing_sdpa_kernel(run)
        self.assertEqual(
            "CUDNN_FLASH_ATTENTION",
            sdpa_kernel,
            f"cuDNN SDPA decode tier regressed: expected CUDNN_FLASH_ATTENTION, got {sdpa_kernel}. "
            "The decode fast path is no longer dispatched on a known-good config (this would otherwise "
            "hide as all-green skips because cudnn_decode_supported() gates by observing dispatch).",
        )


#
# PR #29689 was closed because a per-step device-to-host readback of nonpad_kv_seqlen made the
# decode path un-capturable (cudaStreamSynchronize is illegal while a stream is capturing a CUDA
# graph). RunCudnnSdpaAttention reads the valid length device-side, so the tier must (1) capture
# without an illegal sync and (2) re-read nonpad_kv_seqlen from the device buffer on every replay
# rather than baking the capture-time value into the graph. This test proves both by capturing the
# tier, then mutating nonpad_kv_seqlen in place and replaying: the output must track the reference
# recomputed for the mutated length.
#
# CUDA EP captures the graph on Run 3 (min_num_runs_before_cuda_graph_capture_ == 2), so at least
# three Runs are driven before the mutate/replay phase.
_CUDNN_DECODE_CAPTURE_WARMUP_AND_CAPTURE_RUNS = 3
_CUDNN_DECODE_CAPTURE_REPLAY_RUNS = 2


def _make_cudnn_decode_capture_data(batch, total_kv, q_heads, kv_heads, head_size, scatter_positions, seed=7):
    """Generate fixed fp16 decode inputs (q_seq==1) plus fp32 state for the numpy reference.

    The KV cache is fully populated (no padding zero-fill), so mutating nonpad_kv_seqlen across
    graph replays changes which real KV positions are attended — making the recomputed reference
    genuinely differ between replays. This deliberately differs from run_tensorscatter_attention,
    which couples the cache zero-fill to nonpad_seqlen; that coupling would move the fixed device
    buffer's contents when nonpad changes and defeat the mutate-in-place premise of this test.
    """
    torch.manual_seed(seed)
    kv_hidden = kv_heads * head_size
    q_hidden = q_heads * head_size
    std = 0.2

    def randn(*shape):
        return (torch.randn(*shape, dtype=torch.float16) * std).numpy()

    key_cache = randn(batch, total_kv, kv_hidden)
    value_cache = randn(batch, total_kv, kv_hidden)
    new_k = randn(batch, 1, kv_hidden)
    new_v = randn(batch, 1, kv_hidden)
    query = randn(batch, 1, q_hidden)

    inputs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "new_k": new_k,
        "new_v": new_v,
        "write_indices": numpy.asarray(scatter_positions, dtype=numpy.int64),
        "query": query,
    }

    # Post-scatter fp32 caches for the reference: the ORT graph performs this scatter internally via
    # TensorScatter (bound in place), so the reference must attend the scattered contents. Writing
    # the fixed new_k/new_v at the fixed positions is idempotent across replays, matching the device
    # buffers after each Run.
    key_scattered = key_cache.astype(numpy.float32).copy()
    value_scattered = value_cache.astype(numpy.float32).copy()
    for b in range(batch):
        pos = scatter_positions[b]
        key_scattered[b, pos, :] = new_k[b, 0, :].astype(numpy.float32)
        value_scattered[b, pos, :] = new_v[b, 0, :].astype(numpy.float32)

    reference_state = {
        "query": query.astype(numpy.float32),
        "key_scattered": key_scattered,
        "value_scattered": value_scattered,
    }
    return inputs, reference_state


def _cudnn_decode_capture_reference(reference_state, nonpad_seqlens, batch, total_kv, q_heads, kv_heads, head_size):
    """Recompute the decode reference (BSNH, 3-D) for a given nonpad_kv_seqlen using numpy_attention_ref."""
    q_ref = reference_state["query"].reshape(batch, 1, q_heads, head_size)
    k_ref = reference_state["key_scattered"].reshape(batch, total_kv, kv_heads, head_size)
    v_ref = reference_state["value_scattered"].reshape(batch, total_kv, kv_heads, head_size)
    out = numpy_attention_ref(q_ref, k_ref, v_ref, nonpad_seqlens, is_causal=False)
    return out.reshape(batch, 1, q_heads * head_size)


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestTensorScatterAttentionCudnnSdpaDecodeCudaGraph(unittest.TestCase):
    """CUDA-graph capture/replay of the cuDNN SDPA decode tier (q_seq==1, external KV cache).

    On cuDNN-capable platforms (cudnn_decode_supported) the test asserts routing landed on
    CUDNN_FLASH_ATTENTION so a MATH fallback cannot silently pass this acceptance bar. On other
    platforms the capturability + mutate/replay invariant is still validated on the fallback tier.
    """

    def test_cudnn_decode_cuda_graph_capture_replay(self):
        batch = 2
        total_kv = _CUDNN_DECODE_TOTAL_KV
        q_heads = 8
        kv_heads = 2
        head_size = _CUDNN_DECODE_HEAD_SIZE
        scatter_positions = [3, 5]
        # Both frontiers include the scattered token; the mutation shrinks batch 0 (drops the
        # scattered position) and grows batch 1 to full capacity, so the reference changes across
        # the replay for BOTH batches.
        nonpad_initial = [4, 6]
        nonpad_mutated = [2, 8]

        inputs_np, reference_state = _make_cudnn_decode_capture_data(
            batch, total_kv, q_heads, kv_heads, head_size, scatter_positions
        )
        reference_initial = _cudnn_decode_capture_reference(
            reference_state, nonpad_initial, batch, total_kv, q_heads, kv_heads, head_size
        )
        reference_mutated = _cudnn_decode_capture_reference(
            reference_state, nonpad_mutated, batch, total_kv, q_heads, kv_heads, head_size
        )
        # Sanity: the mutation must actually change the expected output, otherwise the replay would
        # pass even if the captured graph baked in the capture-time length.
        self.assertFalse(
            numpy.allclose(reference_initial, reference_mutated, rtol=rtol["fp16"], atol=atol["fp16"]),
            "Test setup error: nonpad mutation does not change the reference output",
        )

        model_bytes = build_tensorscatter_attention_graph(
            batch_size=batch,
            total_kv_seq_len=total_kv,
            q_seq_len=1,
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            head_size=head_size,
            ort_type=TensorProto.FLOAT16,
            is_causal=0,
        )

        provider_options = {
            "enable_cuda_graph": "1",
            "sdpa_kernel": str(_SDPA_KERNEL_CUDNN_WITH_MATH_FALLBACK),
        }

        # The attention-kernel debug info is initialized once per provider at session creation, so
        # the env var must be set BEFORE the session is built (mirrors _run_capturing_sdpa_kernel).
        previous = os.environ.get("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO")
        os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "1"
        try:
            session = InferenceSession(
                model_bytes,
                SessionOptions(),
                providers=[("CUDAExecutionProvider", provider_options)],
            )

            # Fixed-address device buffers: bound once, never rebound. update_inplace() rewrites
            # nonpad_kv_seqlen's contents without moving the buffer (required for CUDA-graph replay).
            input_ortvalues = {name: OrtValue.ortvalue_from_numpy(arr, "cuda", 0) for name, arr in inputs_np.items()}
            nonpad_ort = OrtValue.ortvalue_from_numpy(numpy.asarray(nonpad_initial, dtype=numpy.int64), "cuda", 0)
            input_ortvalues["nonpad_kv_seqlen"] = nonpad_ort

            present_shape = [batch, kv_heads, total_kv, head_size]
            output_ort = OrtValue.ortvalue_from_shape_and_type(
                [batch, 1, q_heads * head_size], numpy.float16, "cuda", 0
            )
            present_k_ort = OrtValue.ortvalue_from_shape_and_type(present_shape, numpy.float16, "cuda", 0)
            present_v_ort = OrtValue.ortvalue_from_shape_and_type(present_shape, numpy.float16, "cuda", 0)

            io_binding = session.io_binding()
            for name, ortvalue in input_ortvalues.items():
                io_binding.bind_ortvalue_input(name, ortvalue)
            io_binding.bind_ortvalue_output("output", output_ort)
            io_binding.bind_ortvalue_output("present_key", present_k_ort)
            io_binding.bind_ortvalue_output("present_value", present_v_ort)
            # In-place TensorScatter: updated cache aliases the input cache buffer.
            io_binding.bind_ortvalue_output("updated_key_cache", input_ortvalues["key_cache"])
            io_binding.bind_ortvalue_output("updated_value_cache", input_ortvalues["value_cache"])

            # First Run: capture the dispatched tier from the native debug-info stdout.
            with _CaptureStdout() as captured:
                io_binding.synchronize_inputs()
                session.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()
            sdpa_kernel = _parse_sdpa_kernel(captured.output.decode(errors="replace"))
        finally:
            if previous is None:
                os.environ.pop("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO", None)
            else:
                os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = previous

        # A MATH fallback under capture must FAIL this acceptance bar, not pass green.
        if require_cudnn_sdpa() or cudnn_decode_supported(head_size):
            self.assertEqual(
                "CUDNN_FLASH_ATTENTION",
                sdpa_kernel,
                f"Expected cuDNN SDPA decode tier under CUDA-graph capture, got {sdpa_kernel}",
            )

        def assert_output(expected, context):
            actual = output_ort.numpy()
            self.assertTrue(numpy.isfinite(actual).all(), f"{context}: produced non-finite output")
            numpy.testing.assert_allclose(actual, expected, rtol=rtol["fp16"], atol=atol["fp16"], err_msg=context)

        # Validate the first (already-executed) Run, then drive the remaining Runs to force capture
        # (capture begins on Run 3). All use the initial nonpad, so all must match reference_initial.
        assert_output(reference_initial, "CUDA-graph run 0 (pre-capture)")
        for run_index in range(1, _CUDNN_DECODE_CAPTURE_WARMUP_AND_CAPTURE_RUNS):
            io_binding.synchronize_inputs()
            session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            assert_output(reference_initial, f"CUDA-graph run {run_index}")

        # Mutate the device-resident valid length in place and replay. The captured graph must
        # re-read nonpad_kv_seqlen from the device buffer each replay, so the output must now track
        # the reference recomputed for the mutated length. If the length were baked into the graph,
        # the output would stay at reference_initial and this assertion would fail.
        nonpad_ort.update_inplace(numpy.asarray(nonpad_mutated, dtype=numpy.int64))
        for replay_index in range(_CUDNN_DECODE_CAPTURE_REPLAY_RUNS):
            io_binding.synchronize_inputs()
            session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            assert_output(reference_mutated, f"CUDA-graph replay {replay_index} after nonpad mutation")


# #################################################################################################
#  TensorScatter + Attention with nonpad_kv_seqlen + attn_mask (T26 / T31)
# #################################################################################################


def build_tensorscatter_attention_graph_with_mask(
    batch_size,
    total_kv_seq_len,
    q_seq_len,
    q_num_heads,
    kv_num_heads,
    head_size,
    ort_type,
    mask_type,
    mask_shape,
    is_causal=0,
):
    """
    Build ONNX graph: TensorScatter(opset 24) → Attention(opset 24) with both
    nonpad_kv_seqlen AND attn_mask inputs.

    Args:
        mask_type: TensorProto type for the mask (BOOL or same as ort_type for additive).
        mask_shape: shape of the attn_mask tensor (e.g., [q_seq, total_kv_seq] for 2D).
    """
    kv_hidden = kv_num_heads * head_size
    q_hidden = q_num_heads * head_size

    scatter_k_node = helper.make_node(
        "TensorScatter",
        inputs=["key_cache", "new_k", "write_indices"],
        outputs=["updated_key_cache"],
        name="TensorScatterKey",
        axis=1,
    )
    scatter_v_node = helper.make_node(
        "TensorScatter",
        inputs=["value_cache", "new_v", "write_indices"],
        outputs=["updated_value_cache"],
        name="TensorScatterValue",
        axis=1,
    )

    attention_node = helper.make_node(
        "Attention",
        inputs=[
            "query",
            "updated_key_cache",
            "updated_value_cache",
            "attn_mask",
            "",  # past_key
            "",  # past_value
            "nonpad_kv_seqlen",
        ],
        outputs=["output", "present_key", "present_value"],
        name="Attention_0",
        is_causal=is_causal,
        kv_num_heads=kv_num_heads,
        q_num_heads=q_num_heads,
        softcap=0.0,
        qk_matmul_output_mode=0,
        domain="",
    )

    cache_shape = [batch_size, total_kv_seq_len, kv_hidden]
    graph_inputs = [
        helper.make_tensor_value_info("key_cache", ort_type, cache_shape),
        helper.make_tensor_value_info("value_cache", ort_type, cache_shape),
        helper.make_tensor_value_info("new_k", ort_type, [batch_size, q_seq_len, kv_hidden]),
        helper.make_tensor_value_info("new_v", ort_type, [batch_size, q_seq_len, kv_hidden]),
        helper.make_tensor_value_info("write_indices", TensorProto.INT64, [batch_size]),
        helper.make_tensor_value_info("query", ort_type, [batch_size, q_seq_len, q_hidden]),
        helper.make_tensor_value_info("nonpad_kv_seqlen", TensorProto.INT64, [batch_size]),
        helper.make_tensor_value_info("attn_mask", mask_type, mask_shape),
    ]

    present_shape = [batch_size, kv_num_heads, total_kv_seq_len, head_size]
    graph_outputs = [
        helper.make_tensor_value_info("output", ort_type, [batch_size, q_seq_len, q_hidden]),
        helper.make_tensor_value_info("present_key", ort_type, present_shape),
        helper.make_tensor_value_info("present_value", ort_type, present_shape),
        helper.make_tensor_value_info("updated_key_cache", ort_type, cache_shape),
        helper.make_tensor_value_info("updated_value_cache", ort_type, cache_shape),
    ]

    graph = helper.make_graph(
        [scatter_k_node, scatter_v_node, attention_node],
        "TensorScatterAttentionWithMask_Graph",
        graph_inputs,
        graph_outputs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])
    return model.SerializeToString()


def run_tensorscatter_attention_with_mask(
    batch_size,
    total_kv_seq_len,
    q_seq_len,
    q_num_heads,
    kv_num_heads,
    head_size,
    nonpad_seqlens,
    scatter_positions,
    mask_positions_to_block,
    use_bool_mask,
    ep,
    torch_type,
    ort_type,
    is_causal=0,
    std=0.2,
):
    """
    Run TensorScatter + Attention test with BOTH nonpad_kv_seqlen AND attn_mask.

    Args:
        mask_positions_to_block: list of KV position indices to mask out via attn_mask
            (applied uniformly across all batches since 2D mask broadcasts).
        use_bool_mask: True for bool mask, False for float additive mask.
    """
    torch.manual_seed(42)
    kv_hidden = kv_num_heads * head_size
    q_hidden = q_num_heads * head_size
    np_type = numpy.float16 if torch_type == torch.float16 else numpy.float32

    # Generate test data
    key_cache_np = (torch.randn(batch_size, total_kv_seq_len, kv_hidden, dtype=torch_type) * std).numpy()
    value_cache_np = (torch.randn(batch_size, total_kv_seq_len, kv_hidden, dtype=torch_type) * std).numpy()

    for b in range(batch_size):
        old_valid = max(0, nonpad_seqlens[b] - q_seq_len)
        if old_valid < total_kv_seq_len:
            key_cache_np[b, old_valid:, :] = 0
            value_cache_np[b, old_valid:, :] = 0

    new_k_np = (torch.randn(batch_size, q_seq_len, kv_hidden, dtype=torch_type) * std).numpy()
    new_v_np = (torch.randn(batch_size, q_seq_len, kv_hidden, dtype=torch_type) * std).numpy()
    query_np = (torch.randn(batch_size, q_seq_len, q_hidden, dtype=torch_type) * std).numpy()
    write_indices_np = numpy.array(scatter_positions, dtype=numpy.int64)
    nonpad_kv_seqlen_np = numpy.array(nonpad_seqlens, dtype=numpy.int64)

    # Create attn_mask: 2D [q_seq, total_kv_seq]
    if use_bool_mask:
        mask_np = numpy.ones((q_seq_len, total_kv_seq_len), dtype=numpy.bool_)
        for pos in mask_positions_to_block:
            mask_np[:, pos] = False
        mask_ort_type = TensorProto.BOOL
        # Reference: convert bool to additive bias for numpy_attention_ref
        ref_bias = numpy.zeros((1, 1, q_seq_len, total_kv_seq_len), dtype=numpy.float32)
        for pos in mask_positions_to_block:
            ref_bias[:, :, :, pos] = -numpy.inf
    else:
        mask_np = numpy.zeros((q_seq_len, total_kv_seq_len), dtype=np_type)
        for pos in mask_positions_to_block:
            mask_np[:, pos] = numpy.finfo(np_type).min
        mask_ort_type = ort_type
        ref_bias = numpy.zeros((1, 1, q_seq_len, total_kv_seq_len), dtype=numpy.float32)
        for pos in mask_positions_to_block:
            ref_bias[:, :, :, pos] = float(numpy.finfo(np_type).min)

    # --- NumPy reference ---
    key_cache_ref = key_cache_np.astype(numpy.float32).copy()
    value_cache_ref = value_cache_np.astype(numpy.float32).copy()
    new_k_ref = new_k_np.astype(numpy.float32)
    new_v_ref = new_v_np.astype(numpy.float32)

    for b in range(batch_size):
        pos = scatter_positions[b]
        for t in range(q_seq_len):
            key_cache_ref[b, pos + t, :] = new_k_ref[b, t, :]
            value_cache_ref[b, pos + t, :] = new_v_ref[b, t, :]

    q_ref = query_np.astype(numpy.float32).reshape(batch_size, q_seq_len, q_num_heads, head_size)
    k_ref = key_cache_ref.reshape(batch_size, total_kv_seq_len, kv_num_heads, head_size)
    v_ref = value_cache_ref.reshape(batch_size, total_kv_seq_len, kv_num_heads, head_size)

    ref_output = numpy_attention_ref(q_ref, k_ref, v_ref, nonpad_seqlens, is_causal=bool(is_causal), attn_bias=ref_bias)
    ref_output_3d = ref_output.reshape(batch_size, q_seq_len, q_hidden)

    # Compute expected present_key/present_value: BSNH → BNSH transpose of updated cache.
    ref_present_k = k_ref.transpose(0, 2, 1, 3)
    ref_present_v = v_ref.transpose(0, 2, 1, 3)

    # --- ORT execution ---
    mask_shape = [q_seq_len, total_kv_seq_len]
    onnx_model_str = build_tensorscatter_attention_graph_with_mask(
        batch_size=batch_size,
        total_kv_seq_len=total_kv_seq_len,
        q_seq_len=q_seq_len,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        head_size=head_size,
        ort_type=ort_type,
        mask_type=mask_ort_type,
        mask_shape=mask_shape,
        is_causal=is_causal,
    )

    sess_options = SessionOptions()
    session = InferenceSession(onnx_model_str, sess_options, providers=[ep])

    ort_device = "cuda" if "CUDA" in ep else "cpu"
    device_id = 0

    key_cache_ort = OrtValue.ortvalue_from_numpy(key_cache_np, ort_device, device_id)
    value_cache_ort = OrtValue.ortvalue_from_numpy(value_cache_np, ort_device, device_id)
    new_k_ort = OrtValue.ortvalue_from_numpy(new_k_np, ort_device, device_id)
    new_v_ort = OrtValue.ortvalue_from_numpy(new_v_np, ort_device, device_id)
    write_indices_ort = OrtValue.ortvalue_from_numpy(write_indices_np, ort_device, device_id)
    query_ort = OrtValue.ortvalue_from_numpy(query_np, ort_device, device_id)
    nonpad_ort = OrtValue.ortvalue_from_numpy(nonpad_kv_seqlen_np, ort_device, device_id)
    mask_ort = OrtValue.ortvalue_from_numpy(mask_np, ort_device, device_id)

    present_shape = [batch_size, kv_num_heads, total_kv_seq_len, head_size]
    output_ort = OrtValue.ortvalue_from_shape_and_type(
        [batch_size, q_seq_len, q_hidden], np_type, ort_device, device_id
    )
    present_k_ort = OrtValue.ortvalue_from_shape_and_type(present_shape, np_type, ort_device, device_id)
    present_v_ort = OrtValue.ortvalue_from_shape_and_type(present_shape, np_type, ort_device, device_id)

    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input("key_cache", key_cache_ort)
    io_binding.bind_ortvalue_input("value_cache", value_cache_ort)
    io_binding.bind_ortvalue_input("new_k", new_k_ort)
    io_binding.bind_ortvalue_input("new_v", new_v_ort)
    io_binding.bind_ortvalue_input("write_indices", write_indices_ort)
    io_binding.bind_ortvalue_input("query", query_ort)
    io_binding.bind_ortvalue_input("nonpad_kv_seqlen", nonpad_ort)
    io_binding.bind_ortvalue_input("attn_mask", mask_ort)

    io_binding.bind_ortvalue_output("output", output_ort)
    io_binding.bind_ortvalue_output("present_key", present_k_ort)
    io_binding.bind_ortvalue_output("present_value", present_v_ort)
    io_binding.bind_ortvalue_output("updated_key_cache", key_cache_ort)
    io_binding.bind_ortvalue_output("updated_value_cache", value_cache_ort)

    io_binding.synchronize_inputs()
    session.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()

    output_result = output_ort.numpy()
    present_k_result = present_k_ort.numpy()
    present_v_result = present_v_ort.numpy()
    return output_result, ref_output_3d, present_k_result, present_v_result, ref_present_k, ref_present_v


# Test cases for nonpad_kv_seqlen + attn_mask combination
# Format: (batch, q_seq, q_heads, kv_heads, scatter_pos, nonpad_seqlens, mask_positions, label)
_NONPAD_MASK_CASES = [
    # Single batch: mask position 1 within valid range
    (1, 1, 4, 4, [3], [4], [1], "mha_b1_mask1pos"),
    # Multi-batch with different valid lengths, mask position 0
    (2, 1, 4, 4, [2, 4], [3, 5], [0], "mha_b2_mask_pos0"),
    # GQA with mask blocking two positions
    (2, 1, 8, 2, [2, 4], [3, 5], [1, 2], "gqa_b2_mask2pos"),
    # Larger batch with varied lengths
    (4, 1, 4, 4, [1, 3, 5, 7], [2, 4, 6, 8], [0, 3], "mha_b4_varied"),
    # GQA with full valid length, mask some positions
    (2, 1, 8, 2, [7, 7], [8, 8], [2, 5], "gqa_b2_full_mask2"),
]


def _make_mask_test_params(cases, use_bool_mask):
    """Generate parameterized test cases for nonpad + mask tests."""
    mask_str = "bool" if use_bool_mask else "float"
    for batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, mask_pos, label in cases:
        name = f"b{batch}_qh{q_heads}_kvh{kv_heads}_{label}_{mask_str}"
        yield (
            name,
            batch,
            q_seq,
            q_heads,
            kv_heads,
            _HEAD_SIZE,
            _TOTAL_KV_SEQ_LEN,
            scatter_pos,
            seqlens,
            mask_pos,
            use_bool_mask,
        )


def nonpad_mask_cpu_test_cases():
    """CPU test cases for nonpad_kv_seqlen + attn_mask, both bool and float masks."""
    yield from _make_mask_test_params(_NONPAD_MASK_CASES, use_bool_mask=True)
    yield from _make_mask_test_params(_NONPAD_MASK_CASES, use_bool_mask=False)


class TestTensorScatterAttentionWithMaskCPU(unittest.TestCase):
    """Test TensorScatter + Attention with both nonpad_kv_seqlen and attn_mask on CPU.

    Exercises the T26 fix: graceful fallback from Flash to MEA when both inputs present.
    On CPU, both masks compose additively in the reference attention implementation.
    """

    @parameterized.expand(nonpad_mask_cpu_test_cases())
    def test_nonpad_with_mask_cpu(
        self,
        name,
        batch,
        q_seq,
        q_heads,
        kv_heads,
        head_size,
        total_kv,
        scatter_pos,
        seqlens,
        mask_pos,
        use_bool_mask,
    ):
        output, ref_output, present_k, present_v, ref_present_k, ref_present_v = run_tensorscatter_attention_with_mask(
            batch_size=batch,
            total_kv_seq_len=total_kv,
            q_seq_len=q_seq,
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            head_size=head_size,
            nonpad_seqlens=seqlens,
            scatter_positions=scatter_pos,
            mask_positions_to_block=mask_pos,
            use_bool_mask=use_bool_mask,
            ep="CPUExecutionProvider",
            torch_type=torch.float32,
            ort_type=TensorProto.FLOAT,
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=cpu_fp32_rtol, atol=cpu_fp32_atol)
        numpy.testing.assert_allclose(present_k, ref_present_k, rtol=cpu_fp32_rtol, atol=cpu_fp32_atol)
        numpy.testing.assert_allclose(present_v, ref_present_v, rtol=cpu_fp32_rtol, atol=cpu_fp32_atol)


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestTensorScatterAttentionWithMaskCUDA(unittest.TestCase):
    """Test TensorScatter + Attention with both nonpad_kv_seqlen and attn_mask on CUDA.

    Exercises the MEA path which supports seqlen_k_ptr + attn_bias simultaneously.
    Flash is excluded when both inputs are present; MEA handles the combination.
    """

    @parameterized.expand(_make_mask_test_params(_NONPAD_MASK_CASES, use_bool_mask=True))
    def test_nonpad_with_bool_mask_cuda_fp16(
        self,
        name,
        batch,
        q_seq,
        q_heads,
        kv_heads,
        head_size,
        total_kv,
        scatter_pos,
        seqlens,
        mask_pos,
        use_bool_mask,
    ):
        output, ref_output, present_k, present_v, ref_present_k, ref_present_v = run_tensorscatter_attention_with_mask(
            batch_size=batch,
            total_kv_seq_len=total_kv,
            q_seq_len=q_seq,
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            head_size=head_size,
            nonpad_seqlens=seqlens,
            scatter_positions=scatter_pos,
            mask_positions_to_block=mask_pos,
            use_bool_mask=use_bool_mask,
            ep="CUDAExecutionProvider",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_allclose(present_k, ref_present_k, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_allclose(present_v, ref_present_v, rtol=rtol["fp16"], atol=atol["fp16"])


class TestCausalTensorScatterBottomRight(unittest.TestCase):
    """Test that is_causal=1 + TensorScatter decode (S_q != S_kv, no past) is SUPPORTED.

    Per onnx/onnx#8068, is_causal with an external KV cache (nonpad_kv_seqlen) and no
    past_key uses BOTTOM-RIGHT alignment: query in-block index i attends key j iff
    j <= i + offset[b], where offset[b] = nonpad_kv_seqlen[b] - S_q. For decode
    (S_q=1, nonpad=5) the offset is 4, so the single query row attends keys 0..4 — all
    valid cache positions — a meaningful, correct decode result.

    This combination previously returned NOT_IMPLEMENTED under the pre-#8068 upper-left
    assumption (where q[0] would have seen only kv[0]); onnxruntime#28904 removed that
    dispatch guard and now computes the bottom-right frontier via Flash (seqlens_k) or the
    CUTLASS memory-efficient fallback (causal_diagonal_offset = num_keys - num_queries).
    The is_causal + nonpad_kv_seqlen + past_key combination remains rejected upstream
    (ORT_ENFORCE in attention_helper.h). Deeper S_q>1 / nonpad<S_q structural-empty-row
    parity is locked by the C++ AttentionTest goldens (Decode_BottomRight,
    StructuralEmptyRows_Zero_CUDA); at S_q=1 the suite's total-kv-relative numpy reference
    coincides with bottom-right, so the parity assertion below is sound.
    """

    @unittest.skipUnless("CUDAExecutionProvider" in get_available_providers(), "CUDA not available")
    def test_is_causal_with_tensorscatter_no_past_bottom_right(self):
        """is_causal=1 + TensorScatter + nonpad_kv_seqlen (no past) runs and matches the bottom-right reference."""
        output, ref_output, present_k, present_v, ref_present_k, ref_present_v = run_tensorscatter_attention(
            batch_size=1,
            total_kv_seq_len=8,
            q_seq_len=1,
            q_num_heads=2,
            kv_num_heads=2,
            head_size=32,
            nonpad_seqlens=[5],
            scatter_positions=[4],
            ep="CUDAExecutionProvider",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            is_causal=1,
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_allclose(present_k, ref_present_k, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_allclose(present_v, ref_present_v, rtol=rtol["fp16"], atol=atol["fp16"])


if __name__ == "__main__":
    unittest.main()
