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

CUDA limitations:
  - GQA path (kv_num_heads != q_num_heads) requires is_causal=1 and float16
  - MHA path (kv_num_heads == q_num_heads) supports float32 and non-causal
"""

import math
import unittest

import numpy
import torch
from onnx import TensorProto, helper
from parameterized import parameterized

from onnxruntime import InferenceSession, OrtValue, SessionOptions, get_available_providers

# #################################################################################################
#  Helper Functions
# #################################################################################################


def has_cuda_provider():
    return "CUDAExecutionProvider" in get_available_providers()


def has_cuda_device(min_capability: int = 53):
    if not has_cuda_provider() or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= min_capability


def numpy_attention_ref(q, k, v, nonpad_kv_seqlen, is_causal=False):
    """
    NumPy reference implementation of scaled dot-product attention with padding mask.

    Args:
        q: Query [batch, q_seq, num_heads, head_size]
        k: Key [batch, kv_seq, kv_num_heads, head_size]
        v: Value [batch, kv_seq, kv_num_heads, head_size]
        nonpad_kv_seqlen: [batch] — number of valid KV positions per batch
        is_causal: whether to apply causal masking

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

    # Apply causal mask
    if is_causal:
        for sq in range(q_seq):
            offset = kv_seq - q_seq
            for sk in range(kv_seq):
                if sk > sq + offset:
                    scores[:, :, sq, sk] = -numpy.inf

    # Softmax along last axis
    # Handle all-masked rows: if entire row is -inf, softmax gives nan; we want 0
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
):
    """
    Build ONNX graph: TensorScatter(opset 24) → Attention(opset 24).

    TensorScatter uses write_indices [B] to scatter new KV entries into cache
    at per-batch positions. Attention uses updated cache with nonpad_kv_seqlen
    to mask padding.

    The graph exposes updated_key_cache and updated_value_cache as graph outputs
    to enable in-place buffer binding via IO Binding.

    Inputs:
      0: key_cache        [B, total_kv_seq_len, kv_hidden]
      1: value_cache      [B, total_kv_seq_len, kv_hidden]
      2: new_k            [B, q_seq_len, kv_hidden]
      3: new_v            [B, q_seq_len, kv_hidden]
      4: write_indices    [B]   (int64 — per-batch write position)
      5: query            [B, q_seq_len, q_hidden]
      6: nonpad_kv_seqlen [B]   (int64 — valid KV length after scatter)

    Outputs:
      0: output              [B, q_seq_len, q_hidden]
      1: present_key         [B, kv_num_heads, total_kv_seq_len, head_size]
      2: present_value       [B, kv_num_heads, total_kv_seq_len, head_size]
      3: updated_key_cache   [B, total_kv_seq_len, kv_hidden]
      4: updated_value_cache [B, total_kv_seq_len, kv_hidden]
    """
    kv_hidden = kv_num_heads * head_size
    q_hidden = q_num_heads * head_size

    # TensorScatter for key cache update (axis=1: sequence dim in [B, S, H])
    scatter_k_node = helper.make_node(
        "TensorScatter",
        inputs=["key_cache", "new_k", "write_indices"],
        outputs=["updated_key_cache"],
        name="TensorScatterKey",
        axis=1,
    )

    # TensorScatter for value cache update
    scatter_v_node = helper.make_node(
        "TensorScatter",
        inputs=["value_cache", "new_v", "write_indices"],
        outputs=["updated_value_cache"],
        name="TensorScatterValue",
        axis=1,
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
    cache_shape = [batch_size, total_kv_seq_len, kv_hidden]
    graph_inputs = [
        helper.make_tensor_value_info("key_cache", ort_type, cache_shape),
        helper.make_tensor_value_info("value_cache", ort_type, cache_shape),
        helper.make_tensor_value_info("new_k", ort_type, [batch_size, q_seq_len, kv_hidden]),
        helper.make_tensor_value_info("new_v", ort_type, [batch_size, q_seq_len, kv_hidden]),
        helper.make_tensor_value_info("write_indices", TensorProto.INT64, [batch_size]),
        helper.make_tensor_value_info("query", ort_type, [batch_size, q_seq_len, q_hidden]),
        helper.make_tensor_value_info("nonpad_kv_seqlen", TensorProto.INT64, [batch_size]),
    ]

    # Graph outputs: Attention outputs + TensorScatter outputs for in-place binding
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
    device,
    torch_type,
    ort_type,
    is_causal=0,
    std=0.2,
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
    """
    torch.manual_seed(42)
    kv_hidden = kv_num_heads * head_size
    q_hidden = q_num_heads * head_size
    np_type = numpy.float16 if torch_type == torch.float16 else numpy.float32

    # Generate test data as numpy arrays via torch for reproducible seeding
    key_cache_np = (torch.randn(batch_size, total_kv_seq_len, kv_hidden, dtype=torch_type) * std).numpy()
    value_cache_np = (torch.randn(batch_size, total_kv_seq_len, kv_hidden, dtype=torch_type) * std).numpy()

    # Zero out padding positions in cache
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

    # --- NumPy reference ---
    # Compute reference in float32 for accuracy
    key_cache_ref = key_cache_np.astype(numpy.float32).copy()
    value_cache_ref = value_cache_np.astype(numpy.float32).copy()
    new_k_ref = new_k_np.astype(numpy.float32)
    new_v_ref = new_v_np.astype(numpy.float32)

    for b in range(batch_size):
        pos = scatter_positions[b]
        for t in range(q_seq_len):
            key_cache_ref[b, pos + t, :] = new_k_ref[b, t, :]
            value_cache_ref[b, pos + t, :] = new_v_ref[b, t, :]

    # Reshape to BSNH for reference attention
    q_ref = query_np.astype(numpy.float32).reshape(batch_size, q_seq_len, q_num_heads, head_size)
    k_ref = key_cache_ref.reshape(batch_size, total_kv_seq_len, kv_num_heads, head_size)
    v_ref = value_cache_ref.reshape(batch_size, total_kv_seq_len, kv_num_heads, head_size)

    ref_output = numpy_attention_ref(q_ref, k_ref, v_ref, nonpad_seqlens, is_causal=bool(is_causal))
    ref_output_3d = ref_output.reshape(batch_size, q_seq_len, q_hidden)

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
    )

    sess_options = SessionOptions()
    session = InferenceSession(onnx_model_str, sess_options, providers=[ep])

    # Determine device for OrtValue allocation
    ort_device = "cuda" if "CUDA" in ep else "cpu"
    device_id = 0

    # Create OrtValues for inputs on target device
    key_cache_ort = OrtValue.ortvalue_from_numpy(key_cache_np, ort_device, device_id)
    value_cache_ort = OrtValue.ortvalue_from_numpy(value_cache_np, ort_device, device_id)
    new_k_ort = OrtValue.ortvalue_from_numpy(new_k_np, ort_device, device_id)
    new_v_ort = OrtValue.ortvalue_from_numpy(new_v_np, ort_device, device_id)
    write_indices_ort = OrtValue.ortvalue_from_numpy(write_indices_np, ort_device, device_id)
    query_ort = OrtValue.ortvalue_from_numpy(query_np, ort_device, device_id)
    nonpad_ort = OrtValue.ortvalue_from_numpy(nonpad_kv_seqlen_np, ort_device, device_id)

    # Pre-allocate output buffers on target device
    present_shape = [batch_size, kv_num_heads, total_kv_seq_len, head_size]
    output_ort = OrtValue.ortvalue_from_shape_and_type(
        [batch_size, q_seq_len, q_hidden], np_type, ort_device, device_id
    )
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

    return output_result, ref_output_3d, present_k_result, present_v_result


# #################################################################################################
#  Test Case Generator
# #################################################################################################

# Shared test dimensions
_HEAD_SIZE = 64
_TOTAL_KV_SEQ_LEN = 8

_GQA_CASES = [
    # (batch, q_seq, q_heads, kv_heads, scatter_positions, nonpad_seqlens, label)
    (2, 1, 8, 2, [2, 4], [3, 5], "gqa_diff_lens"),
    (2, 1, 8, 2, [4, 4], [5, 5], "gqa_same_lens"),
    (2, 1, 8, 2, [0, 3], [1, 4], "gqa_one_empty"),
    (2, 1, 8, 2, [7, 7], [8, 8], "gqa_full_len"),
]

_MHA_CASES = [
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
    """CUDA fp16: MHA only. CUDA GQA path requires self-attention (kv_seq == q_seq)
    which is incompatible with the decode-step TensorScatter pattern."""
    yield from _make_test_params(_MHA_CASES, is_causal=0)
    yield from _make_test_params(_MHA_CASES, is_causal=1)


def cuda_fp32_test_cases():
    """CUDA fp32: MHA only (CUDA GQA path requires float16)."""
    yield from _make_test_params(_MHA_CASES, is_causal=0)
    yield from _make_test_params(_MHA_CASES, is_causal=1)


# #################################################################################################
#  Test Classes
# #################################################################################################

# Default tolerances
rtol = {"fp16": 5e-3, "fp32": 5e-3}
atol = {"fp16": 5e-3, "fp32": 5e-3}


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
        output, ref_output, _, _ = run_tensorscatter_attention(
            batch_size=batch,
            total_kv_seq_len=total_kv,
            q_seq_len=q_seq,
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            head_size=head_size,
            nonpad_seqlens=seqlens,
            scatter_positions=scatter_pos,
            ep="CPUExecutionProvider",
            device="cpu",
            torch_type=torch.float32,
            ort_type=TensorProto.FLOAT,
            is_causal=is_causal,
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp32"], atol=atol["fp32"])


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestTensorScatterAttentionCUDAFP16(unittest.TestCase):
    """Test TensorScatter + Attention (opset 24) on CUDA with float16 and IO Binding."""

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
        output, ref_output, _, _ = run_tensorscatter_attention(
            batch_size=batch,
            total_kv_seq_len=total_kv,
            q_seq_len=q_seq,
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            head_size=head_size,
            nonpad_seqlens=seqlens,
            scatter_positions=scatter_pos,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            is_causal=is_causal,
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp16"], atol=atol["fp16"])


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
        output, ref_output, _, _ = run_tensorscatter_attention(
            batch_size=batch,
            total_kv_seq_len=total_kv,
            q_seq_len=q_seq,
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            head_size=head_size,
            nonpad_seqlens=seqlens,
            scatter_positions=scatter_pos,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float32,
            ort_type=TensorProto.FLOAT,
            is_causal=is_causal,
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp32"], atol=atol["fp32"])


if __name__ == "__main__":
    unittest.main()
