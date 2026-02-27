# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Tests for TensorScatter(opset 24) + Attention(opset 24) pattern.

Demonstrates a decode step where new KV entries are scattered into a
pre-allocated cache via ScatterElements, then Attention uses the updated
KV cache with nonpad_kv_seqlen to mask out padding positions.

The graph looks like:

  kv_cache (B, S, kv_num_heads*head_size)  ─┐
  scatter_indices (B, 1, kv_num_heads*head_size) ─┤
  new_kv (B, 1, kv_num_heads*head_size)      ─┤
                                                ├─ ScatterElements(axis=1) ─→ updated_kv
                                                                               │
  Q (B, 1, q_num_heads*head_size) ────────────┬─ Attention(opset 24)  ←────────┘
  nonpad_kv_seqlen (B,)  ────────────────────┘          │
                                                        ├─ output
                                                        ├─ present_key
                                                        └─ present_value
"""

import math
import unittest

import numpy
import torch
from onnx import TensorProto, helper
from parameterized import parameterized

from onnxruntime import InferenceSession, SessionOptions, get_available_providers

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
    # q: [b,s,h,d] -> [b,h,s,d];  k: [b,s,h,d] -> [b,h,d,s]
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


def build_scatter_attention_graph(
    batch_size,
    total_kv_seq_len,
    q_seq_len,
    q_num_heads,
    kv_num_heads,
    head_size,
    ort_type,
):
    """
    Build ONNX graph: ScatterElements → Attention (opset 24).

    ScatterElements updates KV cache in-place at specific positions.
    Attention uses updated cache with nonpad_kv_seqlen to mask padding.

    Inputs:
      0: key_cache   [B, total_kv_seq_len, kv_hidden]
      1: value_cache  [B, total_kv_seq_len, kv_hidden]
      2: scatter_indices_k [B, q_seq_len, kv_hidden]
      3: scatter_indices_v [B, q_seq_len, kv_hidden]
      4: new_k        [B, q_seq_len, kv_hidden]
      5: new_v        [B, q_seq_len, kv_hidden]
      6: query        [B, q_seq_len, q_hidden]
      7: nonpad_kv_seqlen [B]

    Outputs:
      0: output       [B, q_seq_len, q_hidden]
      1: present_key  [B, kv_num_heads, total_kv_seq_len, head_size]
      2: present_value [B, kv_num_heads, total_kv_seq_len, head_size]
    """
    kv_hidden = kv_num_heads * head_size
    q_hidden = q_num_heads * head_size

    # ScatterElements for key cache update (axis=1, scatter along seq dim)
    scatter_k_node = helper.make_node(
        "ScatterElements",
        inputs=["key_cache", "scatter_indices_k", "new_k"],
        outputs=["updated_key_cache"],
        name="ScatterKey",
        axis=1,
    )

    # ScatterElements for value cache update
    scatter_v_node = helper.make_node(
        "ScatterElements",
        inputs=["value_cache", "scatter_indices_v", "new_v"],
        outputs=["updated_value_cache"],
        name="ScatterValue",
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
        is_causal=0,
        kv_num_heads=kv_num_heads,
        q_num_heads=q_num_heads,
        softcap=0.0,
        qk_matmul_output_mode=0,
        domain="",
    )

    # Graph inputs
    graph_inputs = [
        helper.make_tensor_value_info("key_cache", ort_type, [batch_size, total_kv_seq_len, kv_hidden]),
        helper.make_tensor_value_info("value_cache", ort_type, [batch_size, total_kv_seq_len, kv_hidden]),
        helper.make_tensor_value_info("scatter_indices_k", TensorProto.INT64, [batch_size, q_seq_len, kv_hidden]),
        helper.make_tensor_value_info("scatter_indices_v", TensorProto.INT64, [batch_size, q_seq_len, kv_hidden]),
        helper.make_tensor_value_info("new_k", ort_type, [batch_size, q_seq_len, kv_hidden]),
        helper.make_tensor_value_info("new_v", ort_type, [batch_size, q_seq_len, kv_hidden]),
        helper.make_tensor_value_info("query", ort_type, [batch_size, q_seq_len, q_hidden]),
        helper.make_tensor_value_info("nonpad_kv_seqlen", TensorProto.INT64, [batch_size]),
    ]

    # Graph outputs
    present_shape = [batch_size, kv_num_heads, total_kv_seq_len, head_size]
    graph_outputs = [
        helper.make_tensor_value_info("output", ort_type, [batch_size, q_seq_len, q_hidden]),
        helper.make_tensor_value_info("present_key", ort_type, present_shape),
        helper.make_tensor_value_info("present_value", ort_type, present_shape),
    ]

    graph = helper.make_graph(
        [scatter_k_node, scatter_v_node, attention_node],
        "ScatterAttention_Graph",
        graph_inputs,
        graph_outputs,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])
    return model.SerializeToString()


def run_scatter_attention(
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
    std=0.2,
):
    """
    Run TensorScatter + Attention test and compare against NumPy reference.

    Args:
        scatter_positions: list of ints per batch — the seq position(s) to scatter new KV into.
            For a single decode step, this is one position per batch.
        nonpad_seqlens: list of ints per batch — valid KV length AFTER scatter.
    """
    torch.manual_seed(42)
    kv_hidden = kv_num_heads * head_size
    q_hidden = q_num_heads * head_size

    # Pre-allocated KV cache (partially filled with valid data, rest is padding/zeros)
    key_cache = torch.randn(batch_size, total_kv_seq_len, kv_hidden, dtype=torch_type, device=device) * std
    value_cache = torch.randn(batch_size, total_kv_seq_len, kv_hidden, dtype=torch_type, device=device) * std

    # Zero out padding positions in cache
    for b in range(batch_size):
        old_valid = max(0, nonpad_seqlens[b] - q_seq_len)
        if old_valid < total_kv_seq_len:
            key_cache[b, old_valid:, :] = 0
            value_cache[b, old_valid:, :] = 0

    # New KV entries to scatter
    new_k = torch.randn(batch_size, q_seq_len, kv_hidden, dtype=torch_type, device=device) * std
    new_v = torch.randn(batch_size, q_seq_len, kv_hidden, dtype=torch_type, device=device) * std

    # Build scatter indices: for each batch, scatter new_k[b, t, :] into key_cache[b, pos, :]
    # ScatterElements with axis=1 needs indices of shape [B, q_seq_len, kv_hidden]
    # where each element along dim=1 is the target seq position, broadcast across hidden dim
    scatter_indices_k = torch.zeros(batch_size, q_seq_len, kv_hidden, dtype=torch.int64, device=device)
    scatter_indices_v = torch.zeros(batch_size, q_seq_len, kv_hidden, dtype=torch.int64, device=device)
    for b in range(batch_size):
        for t in range(q_seq_len):
            pos = scatter_positions[b] + t
            scatter_indices_k[b, t, :] = pos
            scatter_indices_v[b, t, :] = pos

    # Query
    query = torch.randn(batch_size, q_seq_len, q_hidden, dtype=torch_type, device=device) * std

    nonpad_kv_seqlen = torch.tensor(nonpad_seqlens, dtype=torch.int64, device=device)

    # --- NumPy reference ---
    # Apply scatter manually
    key_cache_np = key_cache.float().cpu().numpy().copy()
    value_cache_np = value_cache.float().cpu().numpy().copy()
    new_k_np = new_k.float().cpu().numpy()
    new_v_np = new_v.float().cpu().numpy()

    for b in range(batch_size):
        for t in range(q_seq_len):
            pos = scatter_positions[b] + t
            key_cache_np[b, pos, :] = new_k_np[b, t, :]
            value_cache_np[b, pos, :] = new_v_np[b, t, :]

    # Reshape to BSNH for reference
    q_ref = query.float().cpu().numpy().reshape(batch_size, q_seq_len, q_num_heads, head_size)
    k_ref = key_cache_np.reshape(batch_size, total_kv_seq_len, kv_num_heads, head_size)
    v_ref = value_cache_np.reshape(batch_size, total_kv_seq_len, kv_num_heads, head_size)

    ref_output = numpy_attention_ref(q_ref, k_ref, v_ref, nonpad_seqlens, is_causal=False)
    ref_output_3d = ref_output.reshape(batch_size, q_seq_len, q_hidden)

    # --- ORT execution ---
    onnx_model_str = build_scatter_attention_graph(
        batch_size=batch_size,
        total_kv_seq_len=total_kv_seq_len,
        q_seq_len=q_seq_len,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        head_size=head_size,
        ort_type=ort_type,
    )

    sess_options = SessionOptions()
    session = InferenceSession(onnx_model_str, sess_options, providers=[ep])

    feed = {
        "key_cache": key_cache.cpu().numpy(),
        "value_cache": value_cache.cpu().numpy(),
        "scatter_indices_k": scatter_indices_k.cpu().numpy(),
        "scatter_indices_v": scatter_indices_v.cpu().numpy(),
        "new_k": new_k.cpu().numpy(),
        "new_v": new_v.cpu().numpy(),
        "query": query.cpu().numpy(),
        "nonpad_kv_seqlen": nonpad_kv_seqlen.cpu().numpy(),
    }

    output, present_k, present_v = session.run(None, feed)

    return output, ref_output_3d, present_k, present_v


# #################################################################################################
#  Test Case Generator
# #################################################################################################


def scatter_attention_test_cases():
    """
    Generate test cases for ScatterElements + Attention pattern.

    Simulates decode steps: new KV entries scattered into cache,
    then attention computed with nonpad_kv_seqlen masking.
    """
    head_size = 64
    total_kv_seq_len = 8

    # (batch, q_seq, q_heads, kv_heads, scatter_positions, nonpad_seqlens, label)
    cases = [
        # GQA mode: different valid lengths per batch
        (2, 1, 8, 2, [2, 4], [3, 5], "gqa_diff_lens"),
        # GQA mode: same valid length
        (2, 1, 8, 2, [4, 4], [5, 5], "gqa_same_lens"),
        # GQA mode: one batch empty, scatter into position 0
        (2, 1, 8, 2, [0, 3], [1, 4], "gqa_one_empty"),
        # GQA mode: full length
        (2, 1, 8, 2, [7, 7], [8, 8], "gqa_full_len"),
        # MHA mode: different valid lengths
        (2, 1, 4, 4, [2, 4], [3, 5], "mha_diff_lens"),
        # MHA mode: same valid length
        (2, 1, 4, 4, [4, 4], [5, 5], "mha_same_lens"),
        # MHA mode: one batch empty
        (2, 1, 4, 4, [0, 3], [1, 4], "mha_one_empty"),
        # MHA mode: full length
        (2, 1, 4, 4, [7, 7], [8, 8], "mha_full_len"),
    ]

    for batch, q_seq, q_heads, kv_heads, scatter_pos, seqlens, label in cases:
        name = f"b{batch}_qs{q_seq}_qh{q_heads}_kvh{kv_heads}_h{head_size}_{label}"
        yield (name, batch, q_seq, q_heads, kv_heads, head_size, total_kv_seq_len, scatter_pos, seqlens)


# #################################################################################################
#  Test Classes
# #################################################################################################

# Default tolerances
rtol = {"fp16": 5e-3, "fp32": 5e-3}
atol = {"fp16": 5e-3, "fp32": 5e-3}


class TestTensorScatterAttentionCPU(unittest.TestCase):
    """Test ScatterElements + Attention (opset 24) on CPU with float32."""

    @parameterized.expand(scatter_attention_test_cases())
    def test_scatter_attention_cpu_fp32(
        self, name, batch, q_seq, q_heads, kv_heads, head_size, total_kv, scatter_pos, seqlens
    ):
        output, ref_output, _, _ = run_scatter_attention(
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
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp32"], atol=atol["fp32"])


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestTensorScatterAttentionCUDAFP16(unittest.TestCase):
    """Test ScatterElements + Attention (opset 24) on CUDA with float16."""

    @parameterized.expand(scatter_attention_test_cases())
    def test_scatter_attention_cuda_fp16(
        self, name, batch, q_seq, q_heads, kv_heads, head_size, total_kv, scatter_pos, seqlens
    ):
        output, ref_output, _, _ = run_scatter_attention(
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
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp16"], atol=atol["fp16"])


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
class TestTensorScatterAttentionCUDAFP32(unittest.TestCase):
    """Test ScatterElements + Attention (opset 24) on CUDA with float32."""

    @parameterized.expand(scatter_attention_test_cases())
    def test_scatter_attention_cuda_fp32(
        self, name, batch, q_seq, q_heads, kv_heads, head_size, total_kv, scatter_pos, seqlens
    ):
        output, ref_output, _, _ = run_scatter_attention(
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
        )
        numpy.testing.assert_allclose(output, ref_output, rtol=rtol["fp32"], atol=atol["fp32"])


if __name__ == "__main__":
    unittest.main()
