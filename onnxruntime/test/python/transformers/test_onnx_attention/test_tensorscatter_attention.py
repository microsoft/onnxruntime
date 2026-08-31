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
import warnings
from unittest.mock import patch

import numpy
import torch
from onnx import TensorProto, helper
from parameterized import parameterized

from onnxruntime import (
    InferenceSession,
    OrtValue,
    SessionOptions,
    get_available_providers,
)

# #################################################################################################
#  Helper Functions
# #################################################################################################


_STDOUT_FD = 1
_STDERR_FD = 2


class _CaptureNativeFd:
    """Capture output written by the native ONNX Runtime library to an OS file descriptor.

    Native output cannot be intercepted by Python's contextlib.redirect_stdout/redirect_stderr
    (which only swap sys.stdout/sys.stderr), so fd-level dup2 redirection is used instead.
    Mirrors CaptureStdout in test_gqa.py, generalized over the descriptor:
      * fd 1 (_STDOUT_FD): AttentionKernelDebugInfo::Print writes the SdpaKernel=... tier there.
      * fd 2 (_STDERR_FD): the default logger sink (CLogSink → std::clog) writes ORT log lines
        such as the CUDA EP's "Capturing the cuda graph for this model" there.
    """

    def __init__(self, fd=_STDOUT_FD):
        self.fd = fd
        self.chunk_size = 1024
        self.output = b""

    def _capture(self):
        chunks = []
        while chunk := os.read(self._pipe_reader, self.chunk_size):
            chunks.append(chunk)
        self.output = b"".join(chunks)

    def __enter__(self):
        self._flush()
        self._duped_fd = os.dup(self.fd)
        self._pipe_reader, pipe_writer = os.pipe()
        os.dup2(pipe_writer, self.fd)
        os.close(pipe_writer)
        self._capture_thread = threading.Thread(target=self._capture)
        self._capture_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._flush()
        os.dup2(self._duped_fd, self.fd)
        self._capture_thread.join()
        os.close(self._pipe_reader)
        os.close(self._duped_fd)

    def _flush(self):
        sys.stdout.flush()
        sys.stderr.flush()

    @property
    def text(self):
        return self.output.decode(errors="replace")


def _parse_sdpa_kernel(captured_text):
    """Extract the SdpaKernel=... tier name emitted by AttentionKernelDebugInfo::Print."""
    match = re.search(r"SdpaKernel=(?P<kernel>[A-Z_]+)", captured_text)
    return match.group("kernel") if match is not None else None


_CUDNN_DECODE_SUPPORTED_CACHE = {}


def _observe_cudnn_decode_dispatch(q_num_heads, kv_num_heads, head_size):
    """Build a minimal q_seq==1 external-KV decode graph, run it once requesting the cuDNN SDPA
    kernel with dispatch debug info enabled, and return the observed SdpaKernel string (or None).

    This asks the ORT build itself which tier it selects, so the test's notion of "supported"
    exactly matches the kernel's own cuDNN gate (cudnn_sdpa::is_stable, the cuDNN version ORT
    actually dlopened, and is_supported()'s head-count/head-size checks) with no Python-side
    reimplementation. The probe uses the same (q_num_heads, kv_num_heads, head_size) as the case
    being asserted, so the observation can't drift from what that case will actually dispatch.
    """
    batch_size, total_kv_seq_len = 1, 2
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
    session = None
    try:
        session = InferenceSession(model, SessionOptions(), providers=[("CUDAExecutionProvider", provider_options)])
        with _CaptureNativeFd(_STDOUT_FD) as captured:
            session.run(None, feeds)
    finally:
        if previous is None:
            os.environ.pop("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO", None)
        else:
            os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = previous
        # Release the probe session's device memory eagerly (the probe runs once per unique
        # (q_num_heads, kv_num_heads, head_size)), mirroring run_tensorscatter_attention.
        del session
        gc.collect()
    return _parse_sdpa_kernel(captured.text)


def cudnn_decode_supported(q_num_heads, kv_num_heads, head_size):
    """Return True iff this ORT build actually dispatches the cuDNN SDPA decode tier on this machine
    for the given (q_num_heads, kv_num_heads, head_size) configuration.

    Instead of re-implementing the kernel's cuDNN version rule in Python (which would drift from the
    C++ cudnn_sdpa::is_stable gate and from whichever cuDNN ORT dlopened), this OBSERVES the real
    dispatch: it runs a minimal q_seq==1 decode graph with the requested head configuration once
    with the cuDNN kernel requested and the attention-kernel debug info enabled, and reports True
    only when the observed tier is CUDNN_FLASH_ATTENTION. The result is cached per
    (q_num_heads, kv_num_heads, head_size) so the reported support can never drift from the actual
    case being asserted (e.g. a non-divisible head ratio or a different head_size that
    is_supported() would reject). Any failure (no CUDA provider, unsupported cuDNN, etc.) yields
    False so dependent tests skip cleanly rather than false-failing; the exception is always
    surfaced as a warning so a genuine harness/kernel bug shows up in the CI log instead of
    silently masquerading as "cuDNN unavailable, skip cleanly".
    """
    cache_key = (q_num_heads, kv_num_heads, head_size)
    if cache_key in _CUDNN_DECODE_SUPPORTED_CACHE:
        return _CUDNN_DECODE_SUPPORTED_CACHE[cache_key]
    supported = False
    if has_cuda_provider():
        try:
            supported = _observe_cudnn_decode_dispatch(q_num_heads, kv_num_heads, head_size) == "CUDNN_FLASH_ATTENTION"
        except Exception as exc:
            # Deliberately broad: the probe builds a graph, creates a session and runs it, so the
            # failure modes span pybind Fail/InvalidArgument, OOM, ONNX build errors, etc. Rather
            # than guess the exact set (and silently swallow the rest), report every failure.
            warnings.warn(
                f"cudnn_decode_supported probe failed for "
                f"(q_num_heads={q_num_heads}, kv_num_heads={kv_num_heads}, head_size={head_size}): {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            supported = False
    _CUDNN_DECODE_SUPPORTED_CACHE[cache_key] = supported
    return supported


def require_cudnn_sdpa():
    """Return True when the environment demands the cuDNN SDPA decode tier be dispatched.

    ORT_TEST_REQUIRE_CUDNN_SDPA=1 is intended for an operator to set on a known-good GPU CI leg
    once one exists. As of this PR no pipeline definition exports it, so it currently has no effect
    in this project's CI and only matters for manual/local runs where a developer sets it
    explicitly (the same limitation applies to the analogous GQA/MHA cuDNN paths, which likewise
    have no Hopper+ CI leg enforcing them today).

    When set, the decode dispatch assertions become NON-skippable: a MATH fallback / non-dispatch
    fails loudly instead of being hidden as an all-green skip by the observe-dispatch gating in
    cudnn_decode_supported(). When unset, tests fall back to the normal cudnn_decode_supported()
    skip guard.
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
        with _CaptureNativeFd(_STDOUT_FD) as captured:
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


def get_compute_capability():
    """Return the CUDA compute capability as major*10+minor (0 when no CUDA device is usable).

    Mirrors test_mha.py::get_compute_capability so gating reads the same way across the
    transformers test suite.
    """
    if torch.cuda.is_available() and has_cuda_provider():
        # Device 0 explicitly: this file's sessions/OrtValues all use device_id 0, while
        # torch.cuda.get_device_capability() with no argument would read torch's *current*
        # device, which can differ from the GPU the CUDA EP actually runs on.
        major, minor = torch.cuda.get_device_capability(0)
        return major * 10 + minor
    return 0


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
_SDPA_KERNEL_FLASH_ATTENTION = 1
_SDPA_KERNEL_EFFICIENT_ATTENTION = 2
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
        # Under ORT_TEST_REQUIRE_CUDNN_SDPA=1 (opt-in, manual/local today — see require_cudnn_sdpa) the assertion is
        # non-skippable so a tier regression fails loudly instead of hiding as an all-green skip.
        if require_cudnn_sdpa() or cudnn_decode_supported(q_heads, kv_heads, _CUDNN_DECODE_HEAD_SIZE):
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
        # Under ORT_TEST_REQUIRE_CUDNN_SDPA=1 (opt-in, manual/local today — see require_cudnn_sdpa) the assertion is
        # non-skippable so a tier regression fails loudly instead of hiding as an all-green skip.
        if require_cudnn_sdpa() or cudnn_decode_supported(q_heads, kv_heads, _CUDNN_DECODE_HEAD_SIZE):
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


# Fused zero-fill + BSNH->BNSH transpose (cuDNN decode tier, 4-D output path). On the 4-D path the
# fully-masked-batch zero-fill and the output transpose are performed by ONE kernel
# (LaunchTransposeBSNHtoBNSHWithZeroMask) instead of two back-to-back launches. That kernel
# recomputes the batch index from the flat BNSH output index, so a batch-index off-by-one or a
# transposed-address mistake would either zero the wrong batch or scramble heads/head_size within a
# batch. These cases place the fully-masked batch (nonpad==0) at the first, middle, and last batch
# position, across several batch sizes and head counts, and assert per batch that masked batches are
# EXACTLY zero while unmasked batches are both correct and NOT accidentally zeroed.
# Tuple: (batch, q_seq, q_heads, kv_heads, scatter_positions, nonpad_seqlens, label).
_CUDNN_DECODE_FUSED_4D_CASES = [
    # Masked batch first / last / middle — catches an off-by-one in the recomputed batch index.
    (2, 1, 8, 2, [0, 4], [0, 5], "masked_first"),
    (2, 1, 8, 2, [2, 0], [3, 0], "masked_last"),
    (3, 1, 8, 2, [2, 0, 5], [3, 0, 6], "masked_middle"),
    # More than one masked batch, non-adjacent, batch > 2.
    (4, 1, 8, 8, [0, 1, 0, 5], [0, 2, 0, 6], "masked_b0_b2_mha"),
    # Different head counts / GQA ratios: the fused kernel's index math divides by
    # num_heads * head_size, so a wrong head count shows up as cross-batch corruption.
    (3, 1, 16, 4, [0, 3, 7], [0, 4, 8], "masked_first_16h_4kvh"),
    (2, 1, 8, 1, [4, 0], [5, 0], "masked_last_mqa"),
    # No masked batch at all: the fused kernel must behave as a pure transpose here.
    (3, 1, 8, 2, [1, 3, 7], [2, 4, 8], "no_masked_batch"),
    # Every batch masked: whole output must be zero.
    (2, 1, 8, 2, [0, 0], [0, 0], "all_masked"),
]


def cudnn_decode_fused_4d_test_cases():
    """4-D BNSH decode cases for the fused zero-fill + transpose kernel, for is_causal in {0, 1}."""
    for is_causal in (0, 1):
        causal_str = "causal" if is_causal else "noncausal"
        for batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, label in _CUDNN_DECODE_FUSED_4D_CASES:
            name = f"b{batch}_qh{q_heads}_kvh{kv_heads}_h{_CUDNN_DECODE_HEAD_SIZE}_{causal_str}_{label}_4d"
            yield (name, batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, is_causal)


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestTensorScatterAttentionCudnnSdpaDecodeFused4D(unittest.TestCase):
    """Per-batch assertions for the fused zero-fill + BSNH->BNSH transpose on the 4-D decode path.

    The generic decode tests already compare the whole output against the reference; these add the
    per-batch discrimination that a fused kernel needs: masked batches exactly zero, unmasked
    batches non-zero AND matching the reference elementwise. A transpose-indexing bug that swaps
    heads with sequence/head_size, or a batch-index bug that zeroes a neighbouring batch, fails here
    with a precise message instead of a bulk allclose diff.
    """

    @parameterized.expand(cudnn_decode_fused_4d_test_cases())
    def test_cudnn_decode_fused_zero_and_transpose_4d(
        self,
        name,
        batch,
        q_seq,
        q_heads,
        kv_heads,
        scatter_pos,
        seqlens,
        is_causal,
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
                use_4d=True,
            )

        (output, ref_output, present_k, present_v, ref_present_k, ref_present_v), sdpa_kernel = (
            _run_capturing_sdpa_kernel(run)
        )

        if require_cudnn_sdpa() or cudnn_decode_supported(q_heads, kv_heads, _CUDNN_DECODE_HEAD_SIZE):
            self.assertEqual(
                "CUDNN_FLASH_ATTENTION",
                sdpa_kernel,
                f"Expected cuDNN SDPA decode tier on a cuDNN-capable platform, got {sdpa_kernel}",
            )

        # Output is 4-D BNSH: [batch, q_heads, q_seq, head_size].
        self.assertEqual((batch, q_heads, q_seq, _CUDNN_DECODE_HEAD_SIZE), output.shape)
        self.assertFalse(numpy.isnan(output).any(), "cuDNN SDPA decode produced NaN output")

        for b in range(batch):
            if seqlens[b] == 0:
                # Fully-masked batch: the fused kernel must write exact zeros for this batch only.
                numpy.testing.assert_array_equal(
                    output[b],
                    numpy.zeros_like(output[b]),
                    err_msg=f"batch {b} is fully masked (nonpad==0) but output is not exactly zero",
                )
            else:
                # Not masked: must not be zeroed by a batch-index bug, and must match the reference
                # elementwise (a wrong transpose stride would still be non-zero but misplaced).
                self.assertTrue(
                    numpy.any(output[b] != 0),
                    f"batch {b} has nonpad={seqlens[b]} but output was zeroed",
                )
                numpy.testing.assert_allclose(
                    output[b],
                    ref_output[b],
                    rtol=rtol["fp16"],
                    atol=atol["fp16"],
                    err_msg=f"batch {b} output mismatch (fused transpose indexing?)",
                )

        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_array_equal(present_k, ref_present_k)
        numpy.testing.assert_array_equal(present_v, ref_present_v)


# Auto-enable (default) dispatch cases: no explicit sdpa_kernel provider option, so
# AttentionKernelOptions::has_explicit_kernel_selection_ stays false and
# AllowCudnnFlashAttentionAuto() can take effect (the kernel auto-prefers cuDNN on SM>=90).
# A representative fp16 subset of _CUDNN_DECODE_CASES is enough: this arm is about the *dispatch
# path*, not re-covering the whole shape matrix (already covered by the forced-kernel arm above).
_CUDNN_DECODE_AUTO_CASE_LABELS = ("mha_batch1", "gqa_fully_masked_b0")

# Compute capability at/above which the kernel auto-prefers cuDNN
# (attention.cc: auto_enable_cudnn_flash_attention_ && device_prop.major >= 9).
_CUDNN_AUTO_ENABLE_MIN_COMPUTE_CAPABILITY = 90


def cudnn_decode_auto_test_cases():
    """Auto-dispatch decode cases (fp16, q_seq==1, external KV cache), for is_causal in {0, 1}."""
    selected = [case for case in _CUDNN_DECODE_CASES if case[-1] in _CUDNN_DECODE_AUTO_CASE_LABELS]
    assert len(selected) == len(_CUDNN_DECODE_AUTO_CASE_LABELS), "auto-dispatch case labels drifted from cases"
    for is_causal in (0, 1):
        causal_str = "causal" if is_causal else "noncausal"
        for batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, label in selected:
            name = f"b{batch}_qh{q_heads}_kvh{kv_heads}_h{_CUDNN_DECODE_HEAD_SIZE}_{causal_str}_{label}_auto"
            yield (name, batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, is_causal)


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestTensorScatterAttentionCudnnSdpaDecodeAuto(unittest.TestCase):
    """Exercise the DEFAULT (auto-enable) dispatch path — no explicit sdpa_kernel selection.

    Every other cuDNN decode test passes provider_options={"sdpa_kernel": ...}, which sets
    AttentionKernelOptions::has_explicit_kernel_selection_ and therefore makes
    AllowCudnnFlashAttentionAuto() return false. That leaves the auto-enable path — the one that
    actually runs in production on Hopper+ when nothing is configured — untested. These tests omit
    provider_options entirely so the real default cascade (cuDNN -> Flash -> MEA -> Unfused) decides.

    Assertions:
      * Always: numeric parity with the reference. Every tier is spec-equivalent, so this must hold
        no matter which one the cascade picked (SM<90 runners fall through to Flash/MEA/Unfused).
      * On SM>=90 (where auto-enable can fire) and when the build/machine really does dispatch the
        tier: assert the selected tier is CUDNN_FLASH_ATTENTION, i.e. the auto path is wired up.

    Note the strict assertion also requires SM>=90, not just ORT_TEST_REQUIRE_CUDNN_SDPA=1: that env
    gate marks a GPU where the *forced* cuDNN path is known good (true from SM80), but auto-enable
    is SM>=90 only, so enforcing it on an SM80 CI leg would false-fail.
    """

    @parameterized.expand(cudnn_decode_auto_test_cases())
    def test_tensorscatter_attention_cudnn_decode_auto_fp16(
        self,
        name,
        batch,
        q_seq,
        q_heads,
        kv_heads,
        scatter_pos,
        seqlens,
        is_causal,
    ):
        def run():
            # No provider_options: leave kernel selection entirely to the default cascade.
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
            )

        # Clear any ambient ORT_ENABLE_CUDNN_FLASH_ATTENTION for the duration of the run. The
        # strict-probe helpers below (require_cudnn_sdpa / cudnn_decode_supported) force cuDNN via
        # an explicit sdpa_kernel provider option and therefore ignore this env var, but the run
        # under test omits provider_options and DOES honor it (attention_kernel_options.cc: an
        # explicit "0" disables the auto path even on SM90+). Without this scoping, a shell with
        # ORT_ENABLE_CUDNN_FLASH_ATTENTION=0 would make the probe say "cuDNN available" while the
        # legitimate auto-dispatch picks Flash, failing the assertion spuriously.
        previous_cudnn_env = os.environ.get("ORT_ENABLE_CUDNN_FLASH_ATTENTION")
        os.environ.pop("ORT_ENABLE_CUDNN_FLASH_ATTENTION", None)
        try:
            (output, ref_output, present_k, present_v, ref_present_k, ref_present_v), sdpa_kernel = (
                _run_capturing_sdpa_kernel(run)
            )
        finally:
            if previous_cudnn_env is not None:
                os.environ["ORT_ENABLE_CUDNN_FLASH_ATTENTION"] = previous_cudnn_env

        if get_compute_capability() >= _CUDNN_AUTO_ENABLE_MIN_COMPUTE_CAPABILITY and (
            require_cudnn_sdpa() or cudnn_decode_supported(q_heads, kv_heads, _CUDNN_DECODE_HEAD_SIZE)
        ):
            self.assertEqual(
                "CUDNN_FLASH_ATTENTION",
                sdpa_kernel,
                "Expected the auto-enable path to select the cuDNN SDPA decode tier on SM>=90 "
                f"with no explicit sdpa_kernel, got {sdpa_kernel}",
            )

        # Spec-equivalence holds for every tier, so parity is asserted unconditionally.
        self.assertFalse(numpy.isnan(output).any(), "Default-dispatch decode produced NaN output")
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_array_equal(present_k, ref_present_k)
        numpy.testing.assert_array_equal(present_v, ref_present_v)


class TestTensorScatterAttentionCudnnSdpaDecodeCanary(unittest.TestCase):
    """Canary that fails loudly if the cuDNN SDPA decode tier stops being dispatched.

    cudnn_decode_supported() gates every decode test by OBSERVING dispatch, which is precise but has
    a failure mode: if the tier silently regressed (stopped selecting cuDNN), the observation would
    return False and every decode test would skip green — hiding the regression as all-green skips.
    This canary closes that hole. When ORT_TEST_REQUIRE_CUDNN_SDPA=1 is set (opt-in; intended for a
    known-good GPU CI leg once one exists — no pipeline exports it today, so this is manual/local
    only, see require_cudnn_sdpa), a MATH fallback / non-dispatch on the minimal known-good config
    FAILS loudly instead of skipping. On dev boxes or unsupported cuDNN (env unset) it falls back to
    the normal cudnn_decode_supported() skip guard so it never false-alarms.
    """

    @unittest.skipIf(not has_cuda_provider(), "CUDA provider not available")
    def test_cudnn_sdpa_decode_tier_is_dispatched(self):
        enforce = require_cudnn_sdpa()
        if not enforce and not cudnn_decode_supported(1, 1, _CUDNN_DECODE_HEAD_SIZE):
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
# CUDA EP allows capture once min_num_runs_before_cuda_graph_capture_ (== 2) regular Runs have
# happened, so at least three Runs are driven before the mutate/replay phase. The Run index at
# which capture actually engages is an implementation detail (this build was observed beginning
# capture on the very first Run), so the capture-engagement assertion scans the ORT log of every
# warmup Run instead of a single expected Run.
_CUDNN_DECODE_CAPTURE_WARMUP_AND_CAPTURE_RUNS = 3
_CUDNN_DECODE_CAPTURE_REPLAY_RUNS = 2

# Log line emitted by CUDAExecutionProvider::OnRunStart (cuda_execution_provider.cc) when it begins
# capturing. It is the only capture-engagement signal reachable from Python, and it needs INFO
# severity (logging::Severity::kINFO == 1) plus fd-2 capture (the default CLogSink writes std::clog).
_CUDA_GRAPH_CAPTURE_LOG = "Capturing the cuda graph for this model"
_ORT_LOG_SEVERITY_INFO = 1

# Every ORT log line includes its session's logid (ostream_sink.cc: "[severity:category:logger_id,
# location] message"), so a unique logid lets the assertion below require BOTH the capture message
# AND this test's own logid on the SAME line. fd 2 is process-global: without this, a concurrently
# running session (e.g. a parallel test worker) logging the identical capture message during the
# redirected window would satisfy a bare substring match for the wrong session.
_CUDA_GRAPH_CAPTURE_TEST_LOGID = "test_cudnn_decode_cuda_graph_capture_replay"


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

    Capture engagement is VERIFIED, not assumed: the session runs at INFO severity and the test
    asserts the CUDA EP's "Capturing the cuda graph for this model" log line (emitted from
    CUDAExecutionProvider::OnRunStart) appears on the native stderr fd during the warmup/capture
    Runs. Without that check, an ORT build that declined to capture would still pass every
    assertion below (the mutate-and-replay check would degrade to "eager Runs read the current
    device buffer", which is trivially true) — a false green on the exact regression that closed
    issue #29689.
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
        # INFO severity is required to surface the CUDA EP's capture log line, which is the only
        # signal exposed to Python that graph capture actually engaged.
        session_options = SessionOptions()
        session_options.log_severity_level = _ORT_LOG_SEVERITY_INFO
        session_options.logid = _CUDA_GRAPH_CAPTURE_TEST_LOGID
        capture_log_chunks = []
        try:
            session = InferenceSession(
                model_bytes,
                session_options,
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

            # First Run: capture the dispatched tier from the native debug-info stdout (fd 1) and,
            # simultaneously, the ORT log stream on stderr (fd 2) where the CUDA EP announces graph
            # capture. Both fds are redirected independently, so nesting is safe.
            with (
                _CaptureNativeFd(_STDERR_FD) as captured_log,
                _CaptureNativeFd(_STDOUT_FD) as captured,
            ):
                io_binding.synchronize_inputs()
                session.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()
            capture_log_chunks.append(captured_log.text)
            sdpa_kernel = _parse_sdpa_kernel(captured.text)
        finally:
            if previous is None:
                os.environ.pop("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO", None)
            else:
                os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = previous

        # A MATH fallback under capture must FAIL this acceptance bar, not pass green.
        if require_cudnn_sdpa() or cudnn_decode_supported(q_heads, kv_heads, head_size):
            self.assertEqual(
                "CUDNN_FLASH_ATTENTION",
                sdpa_kernel,
                f"Expected cuDNN SDPA decode tier under CUDA-graph capture, got {sdpa_kernel}",
            )

        def assert_output(expected, context):
            actual = output_ort.numpy()
            self.assertTrue(numpy.isfinite(actual).all(), f"{context}: produced non-finite output")
            numpy.testing.assert_allclose(actual, expected, rtol=rtol["fp16"], atol=atol["fp16"], err_msg=context)

        # Validate the first (already-executed) Run, then drive the remaining warmup Runs so capture
        # is guaranteed to have engaged. All use the initial nonpad, so all must match
        # reference_initial.
        assert_output(reference_initial, "CUDA-graph run 0 (pre-capture)")
        for run_index in range(1, _CUDNN_DECODE_CAPTURE_WARMUP_AND_CAPTURE_RUNS):
            with _CaptureNativeFd(_STDERR_FD) as captured_log:
                io_binding.synchronize_inputs()
                session.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()
            capture_log_chunks.append(captured_log.text)
            assert_output(reference_initial, f"CUDA-graph run {run_index}")

        # Prove CUDA-graph capture actually engaged. Otherwise the mutate/replay check below would
        # degrade to "eager Runs read the current device buffer" and pass green even if ORT declined
        # to capture, leaving the #29689 regression unverified. The log line is emitted from
        # CUDAExecutionProvider::OnRunStart at INFO severity; every Run up to and including the
        # capture Run is scanned so the assertion does not depend on which Run index captures.
        #
        # Require the capture message and this test's own session logid on the SAME line (not just
        # the message alone): fd 2 is a process-global descriptor, so a bare substring match could
        # false-pass on a capture line logged by an unrelated concurrently-running session (e.g.
        # under parallel test execution). The logid anchors the match to this test's own session
        # (ostream_sink.cc formats each line as "[severity:category:logger_id, location] message").
        combined_log = "".join(capture_log_chunks)
        capture_line_pattern = re.compile(
            rf"^.*{re.escape(_CUDA_GRAPH_CAPTURE_TEST_LOGID)}.*{re.escape(_CUDA_GRAPH_CAPTURE_LOG)}.*$",
            re.MULTILINE,
        )
        self.assertRegex(
            combined_log,
            capture_line_pattern,
            "CUDA-graph capture never engaged for this test's own session: no captured stderr line "
            f"during the first {_CUDNN_DECODE_CAPTURE_WARMUP_AND_CAPTURE_RUNS} Runs contains both "
            f"this session's logid ('{_CUDA_GRAPH_CAPTURE_TEST_LOGID}') and "
            f"'{_CUDA_GRAPH_CAPTURE_LOG}', so the replay assertions below would not actually "
            "exercise a captured graph.",
        )

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


# #################################################################################################
#  present_key/present_value redundant-copy skip (external-cache 4-D BNSH aliasing)
# #################################################################################################
# Attention::Run{Flash,CudnnSdpa,MemoryEfficient,Unfused}Attention populate present_key/present_value
# for 4-D BNSH external-cache inputs via llm_attention_detail::CopyKVToPresent (attention.cc), which
# skips the D2D copy entirely when present_key/present_value is bound to the SAME device buffer as
# the K/V cache input (mirroring TensorScatter's own .MayInplace(0, 0) self-copy skip). These tests:
#   1. Prove the skip actually fires when present_* aliases the cache buffer, via the greppable
#      "present_copy_skipped" log tag emitted at VERBOSE severity — without this, a regression that
#      re-introduced the unconditional copy would still pass on output correctness alone (the copy
#      is redundant, not wrong, when src == dst).
#   2. Prove the non-aliased path still runs the copy (tag absent) and both paths remain correct,
#      across all four backends (Flash, cuDNN, Memory-Efficient, unfused/MATH).
#
# Only reachable for 4-D BNSH inputs (use_4d=True): the 3-D BSNH path always needs a
# layout-changing transpose into present_*, so src and dst can never alias there.
_PRESENT_COPY_SKIPPED_TAG = "present_copy_skipped"
_ORT_LOG_SEVERITY_VERBOSE = 0

# CopyKVToPresent logs through this session's logger: the plugin build uses the kernel-info logger,
# while the legacy build uses the logger assigned to the EP. Thus the tag is controlled purely by
# session_options.log_severity_level — no process-global logger mutation is needed.
# Every ORT log line carries its session's logid, so a unique logid lets the assertions below
# require BOTH the tag AND this test's own logid on the SAME line. fd 2 is a process-global
# descriptor: without the logid anchor, a concurrently running session (e.g. a parallel test
# worker) emitting the same tag during the redirected window would satisfy a bare substring match
# for the wrong session — which would make the negative (tag-absent) assertion racy.
_PRESENT_COPY_SKIP_TEST_LOGID = "test_attention_present_kv_copy_skip"

# Matches a captured log line that carries BOTH this test's session logid and the skip tag.
# OStreamSink writes "<timestamp> [<severity>:<category>:<logger_id>, <location>] <message>"
# (onnxruntime/core/common/logging/sinks/ostream_sink.cc), so the logid is delimited by ':' before
# and ', ' after. Matching those delimiters (rather than a bare substring) keeps a different
# session whose logid merely CONTAINS this one from false-matching.
_PRESENT_COPY_SKIPPED_LINE = re.compile(
    rf"^.*:{re.escape(_PRESENT_COPY_SKIP_TEST_LOGID)}, .*{re.escape(_PRESENT_COPY_SKIPPED_TAG)}.*$",
    re.MULTILINE,
)


# Env overrides applied (per backend case) before the session — and therefore
# AttentionKernelOptions — is created, so an observed MATH fallback cannot be caused by an ambient
# ORT_DISABLE_* env var and TestAttentionPresentKVCopySkip._check_dispatched_tier's skip stays as
# narrow as possible. (These cases also pass an explicit sdpa_kernel provider option, which already
# bypasses the env vars in AttentionKernelOptions::Initialize; this is defense-in-depth against an
# ambient environment.) cuDNN has no ORT_DISABLE_* switch in this cascade (only the opt-in
# ORT_ENABLE_CUDNN_FLASH_ATTENTION, superseded by the sdpa_kernel option), and the unfused/MATH
# kernel cannot be disabled at all, so those two cases need no override.
_FORCE_ENABLE_ENV = {
    "flash": {"ORT_DISABLE_FLASH_ATTENTION": "0"},
    "efficient": {"ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION": "0"},
}


def _run_tensorscatter_attention_4d(
    batch_size,
    total_kv_seq_len,
    q_seq_len,
    q_num_heads,
    kv_num_heads,
    head_size,
    nonpad_seqlens,
    scatter_positions,
    provider_options,
    alias_present,
):
    """4-D BNSH TensorScatter + Attention, with present_key/value optionally aliased to the K/V
    cache buffer (the pattern llm_attention_detail::CopyKVToPresent's skip targets).

    Runs at VERBOSE log severity and captures native stderr (present_copy_skipped tag) and
    stdout (SdpaKernel=... dispatch tier, via ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO) so callers
    can assert on both. Returns (output, present_k, present_v, ref_output, ref_present_k,
    ref_present_v, log_text, sdpa_kernel).
    """
    torch.manual_seed(123)
    std = 0.2
    key_cache_t = torch.randn(batch_size, kv_num_heads, total_kv_seq_len, head_size, dtype=torch.float16) * std
    value_cache_t = torch.randn(batch_size, kv_num_heads, total_kv_seq_len, head_size, dtype=torch.float16) * std
    for b in range(batch_size):
        old_valid = max(0, nonpad_seqlens[b] - q_seq_len)
        if old_valid < total_kv_seq_len:
            key_cache_t[b, :, old_valid:, :] = 0
            value_cache_t[b, :, old_valid:, :] = 0
    new_k_t = torch.randn(batch_size, kv_num_heads, q_seq_len, head_size, dtype=torch.float16) * std
    new_v_t = torch.randn(batch_size, kv_num_heads, q_seq_len, head_size, dtype=torch.float16) * std
    query_t = torch.randn(batch_size, q_num_heads, q_seq_len, head_size, dtype=torch.float16) * std

    key_cache_ref = key_cache_t.float().cpu().numpy().copy()
    value_cache_ref = value_cache_t.float().cpu().numpy().copy()
    new_k_ref = new_k_t.float().cpu().numpy()
    new_v_ref = new_v_t.float().cpu().numpy()
    for b in range(batch_size):
        pos = scatter_positions[b]
        for t in range(q_seq_len):
            key_cache_ref[b, :, pos + t, :] = new_k_ref[b, :, t, :]
            value_cache_ref[b, :, pos + t, :] = new_v_ref[b, :, t, :]
    q_ref = query_t.float().cpu().numpy().transpose(0, 2, 1, 3)
    k_ref = key_cache_ref.transpose(0, 2, 1, 3)
    v_ref = value_cache_ref.transpose(0, 2, 1, 3)
    ref_output_bsnh = numpy_attention_ref(q_ref, k_ref, v_ref, nonpad_seqlens, is_causal=False)
    ref_output = ref_output_bsnh.transpose(0, 2, 1, 3)
    ref_present_k = key_cache_ref
    ref_present_v = value_cache_ref

    onnx_model_str = build_tensorscatter_attention_graph(
        batch_size=batch_size,
        total_kv_seq_len=total_kv_seq_len,
        q_seq_len=q_seq_len,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        head_size=head_size,
        ort_type=TensorProto.FLOAT16,
        is_causal=0,
        use_4d=True,
    )

    # CopyKVToPresent uses the kernel-info logger in plugin builds and the EP logger in legacy
    # builds. Both are session-anchored, so the unique logid anchors assertions to this session.
    # ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO is still process-global state and must be restored even
    # if session creation itself throws, so everything from here to the run is wrapped in one
    # try/finally.
    previous_debug_info_env = os.environ.get("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO")
    os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "1"
    session = None
    io_binding = None
    try:
        session_options = SessionOptions()
        session_options.log_severity_level = _ORT_LOG_SEVERITY_VERBOSE
        session_options.logid = _PRESENT_COPY_SKIP_TEST_LOGID
        session = InferenceSession(
            onnx_model_str,
            session_options,
            providers=["CUDAExecutionProvider"],
            provider_options=[provider_options],
        )

        key_cache_ort = OrtValue.ortvalue_from_numpy(key_cache_t.cpu().numpy(), "cuda", 0)
        value_cache_ort = OrtValue.ortvalue_from_numpy(value_cache_t.cpu().numpy(), "cuda", 0)
        new_k_ort = OrtValue.ortvalue_from_numpy(new_k_t.cpu().numpy(), "cuda", 0)
        new_v_ort = OrtValue.ortvalue_from_numpy(new_v_t.cpu().numpy(), "cuda", 0)
        write_indices_ort = OrtValue.ortvalue_from_numpy(numpy.array(scatter_positions, dtype=numpy.int64), "cuda", 0)
        query_ort = OrtValue.ortvalue_from_numpy(query_t.cpu().numpy(), "cuda", 0)
        nonpad_ort = OrtValue.ortvalue_from_numpy(numpy.array(nonpad_seqlens, dtype=numpy.int64), "cuda", 0)

        output_shape = [batch_size, q_num_heads, q_seq_len, head_size]
        output_ort = OrtValue.ortvalue_from_shape_and_type(output_shape, numpy.float16, "cuda", 0)

        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input("key_cache", key_cache_ort)
        io_binding.bind_ortvalue_input("value_cache", value_cache_ort)
        io_binding.bind_ortvalue_input("new_k", new_k_ort)
        io_binding.bind_ortvalue_input("new_v", new_v_ort)
        io_binding.bind_ortvalue_input("write_indices", write_indices_ort)
        io_binding.bind_ortvalue_input("query", query_ort)
        io_binding.bind_ortvalue_input("nonpad_kv_seqlen", nonpad_ort)
        io_binding.bind_ortvalue_output("output", output_ort)
        # In-place TensorScatter: the updated cache always aliases the cache input buffer.
        io_binding.bind_ortvalue_output("updated_key_cache", key_cache_ort)
        io_binding.bind_ortvalue_output("updated_value_cache", value_cache_ort)
        if alias_present:
            # Full 3-way alias: key_cache input == updated_key_cache output == present_key output.
            # This is the production pattern CopyKVToPresent's skip targets.
            io_binding.bind_ortvalue_output("present_key", key_cache_ort)
            io_binding.bind_ortvalue_output("present_value", value_cache_ort)
        else:
            present_shape = [batch_size, kv_num_heads, total_kv_seq_len, head_size]
            present_k_ort = OrtValue.ortvalue_from_shape_and_type(present_shape, numpy.float16, "cuda", 0)
            present_v_ort = OrtValue.ortvalue_from_shape_and_type(present_shape, numpy.float16, "cuda", 0)
            io_binding.bind_ortvalue_output("present_key", present_k_ort)
            io_binding.bind_ortvalue_output("present_value", present_v_ort)

        # fd 2 (stderr) carries the present_copy_skipped VERBOSE tag; fd 1 (stdout) carries the
        # AttentionKernelDebugInfo SdpaKernel=... dispatch tier. Both fds are redirected
        # independently, so nesting (as elsewhere in this file) is safe.
        with (
            _CaptureNativeFd(_STDERR_FD) as captured_log,
            _CaptureNativeFd(_STDOUT_FD) as captured_stdout,
        ):
            io_binding.synchronize_inputs()
            session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
        log_text = captured_log.text
        sdpa_kernel = _parse_sdpa_kernel(captured_stdout.text)

        output = output_ort.numpy()
        if alias_present:
            present_k = key_cache_ort.numpy()
            present_v = value_cache_ort.numpy()
        else:
            present_k = present_k_ort.numpy()
            present_v = present_v_ort.numpy()
    finally:
        if previous_debug_info_env is None:
            os.environ.pop("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO", None)
        else:
            os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = previous_debug_info_env

    del io_binding, session
    gc.collect()
    return output, present_k, present_v, ref_output, ref_present_k, ref_present_v, log_text, sdpa_kernel


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestAttentionPresentKVCopySkip(unittest.TestCase):
    """present_key/present_value D2D-copy skip when aliased to the external KV cache buffer."""

    # (name, provider_options, expected SdpaKernel=... tier, "is this tier expected to be
    # available on this HW/build" predicate). The predicate gates the dispatch assertion the
    # same way the existing cuDNN decode tests do (see cudnn_decode_supported/
    # require_cudnn_sdpa): a "PASSED" on numeric correctness alone does NOT prove which backend
    # ran, since flash/efficient/cudnn all OR in a MATH fallback bit — without this assertion a
    # regression that broke 3 of the 4 backends' call sites while leaving unfused/MATH intact
    # would still pass all 8 tests green. The predicates only see HW capability, so a tier that is
    # compiled out of this build is detected from the observed MATH fallback instead and turned
    # into a skip (see _check_dispatched_tier and _FORCE_ENABLE_ENV).
    _CASES = (
        (
            "flash",
            {"sdpa_kernel": str(_SDPA_KERNEL_FLASH_ATTENTION | _SDPA_KERNEL_MATH)},
            "FLASH_ATTENTION",
            has_flash_attention,
        ),
        (
            "efficient",
            {"sdpa_kernel": str(_SDPA_KERNEL_EFFICIENT_ATTENTION | _SDPA_KERNEL_MATH)},
            "EFFICIENT_ATTENTION",
            lambda: True,  # cutlass memory-efficient attention supports this class's SM53+ floor.
        ),
        (
            "cudnn",
            {"sdpa_kernel": str(_SDPA_KERNEL_CUDNN_WITH_MATH_FALLBACK)},
            "CUDNN_FLASH_ATTENTION",
            lambda: require_cudnn_sdpa() or cudnn_decode_supported(8, 2, 64),
        ),
        (
            "math",
            {"sdpa_kernel": str(_SDPA_KERNEL_MATH)},
            "MATH",
            lambda: True,  # the unfused/MATH kernel has no HW/build gating.
        ),
    )

    def _check_dispatched_tier(self, name, expected_kernel, sdpa_kernel, is_supported):
        """Assert the expected backend actually dispatched, skipping when it is not available.

        Numeric correctness alone does NOT prove which backend ran (flash/efficient/cudnn all OR in
        a MATH fallback bit), hence the assertion. But a fused tier can also be compiled out
        (onnxruntime_USE_FLASH_ATTENTION / onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION), which the
        HW-only predicates above cannot see. The runtime ORT_DISABLE_* switches are already ruled
        out by _FORCE_ENABLE_ENV, so observing MATH where a fused tier was expected means "this
        tier is not compiled into this build": skip instead of failing, since the aliasing logic
        under test is backend-independent and the "math" case still covers it. (Residual
        ambiguity: a genuine backend regression would also surface as MATH here. That limitation
        is shared with the other backend-gated tests in this file and is accepted.)
        """
        if not is_supported():
            return
        if expected_kernel != "MATH" and sdpa_kernel == "MATH":
            self.skipTest(
                f"[{name}] the {expected_kernel} tier fell back to MATH even with its "
                "ORT_DISABLE_* override forced off, so it is not compiled into this build."
            )
        self.assertEqual(
            expected_kernel,
            sdpa_kernel,
            f"[{name}] expected the {expected_kernel} tier to dispatch on this HW/build, "
            f"got {sdpa_kernel} — the per-backend coverage this test claims is not real.",
        )

    @parameterized.expand(_CASES)
    def test_copy_skipped_when_present_aliases_cache(self, name, provider_options, expected_kernel, is_supported):
        batch, total_kv, q_seq, q_heads, kv_heads, head_size = 2, 8, 1, 8, 2, 64
        nonpad_seqlens = [4, 6]
        scatter_positions = [3, 5]

        with patch.dict(os.environ, _FORCE_ENABLE_ENV.get(name, {})):
            output, present_k, present_v, ref_output, ref_present_k, ref_present_v, log_text, sdpa_kernel = (
                _run_tensorscatter_attention_4d(
                    batch,
                    total_kv,
                    q_seq,
                    q_heads,
                    kv_heads,
                    head_size,
                    nonpad_seqlens,
                    scatter_positions,
                    provider_options,
                    alias_present=True,
                )
            )

        self._check_dispatched_tier(name, expected_kernel, sdpa_kernel, is_supported)
        skipped_copy_records = _PRESENT_COPY_SKIPPED_LINE.findall(log_text)
        self.assertEqual(
            2,
            len(skipped_copy_records),
            f"[{name}] present_key/value aliased the cache buffer, so exactly two captured log records "
            f"(one for K and one for V) must carry both this session's logid "
            f"('{_PRESENT_COPY_SKIP_TEST_LOGID}') and the '{_PRESENT_COPY_SKIPPED_TAG}' tag; "
            f"got {len(skipped_copy_records)}.",
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_allclose(present_k, ref_present_k, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_allclose(present_v, ref_present_v, rtol=rtol["fp16"], atol=atol["fp16"])

    @parameterized.expand(_CASES)
    def test_copy_still_runs_when_present_not_aliased(self, name, provider_options, expected_kernel, is_supported):
        batch, total_kv, q_seq, q_heads, kv_heads, head_size = 2, 8, 1, 8, 2, 64
        nonpad_seqlens = [4, 6]
        scatter_positions = [3, 5]

        with patch.dict(os.environ, _FORCE_ENABLE_ENV.get(name, {})):
            output, present_k, present_v, ref_output, ref_present_k, ref_present_v, log_text, sdpa_kernel = (
                _run_tensorscatter_attention_4d(
                    batch,
                    total_kv,
                    q_seq,
                    q_heads,
                    kv_heads,
                    head_size,
                    nonpad_seqlens,
                    scatter_positions,
                    provider_options,
                    alias_present=False,
                )
            )

        self._check_dispatched_tier(name, expected_kernel, sdpa_kernel, is_supported)
        skipped_copy_records = _PRESENT_COPY_SKIPPED_LINE.findall(log_text)
        self.assertEqual(
            0,
            len(skipped_copy_records),
            f"[{name}] present_key/value used SEPARATE buffers from the cache, but "
            f"{len(skipped_copy_records)} log record(s) with this session's logid "
            f"('{_PRESENT_COPY_SKIP_TEST_LOGID}') carry the '{_PRESENT_COPY_SKIPPED_TAG}' tag — "
            "the aliasing check may have a false positive.",
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_allclose(present_k, ref_present_k, rtol=rtol["fp16"], atol=atol["fp16"])
        numpy.testing.assert_allclose(present_v, ref_present_v, rtol=rtol["fp16"], atol=atol["fp16"])


if __name__ == "__main__":
    unittest.main()
