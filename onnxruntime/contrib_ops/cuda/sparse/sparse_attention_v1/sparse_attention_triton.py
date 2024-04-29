# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import triton
import triton.language as tl


# This kernel is for prompt only and assume that past sequence length is 0. It only supports right padding.
@triton.jit
def block_sparse_attention_kernel(
    out,  # output [B, H, M, D]. Note that B is batch_size, H is num_heads, M is q_seq_len, and D is head_size
    Q,  # query [B, H, M, D]
    K,  # key [B, H_kv, N, D]. Note that N is max_seq_len for kv cache, H_kv is num_kv_heads
    V,  # value [B, H_kv, N, D]
    layout_csr_row_indices,  # block mask CSR format. Shape is [L, num_rows + 1] where num_rows = max_seq_len / BLOCK_M
    layout_csr_col_indices,  # block mask CSR format. Shape is [L, num_rows * num_cols] where num_cols = max_seq_len / BLOCK_N
    layout_csr_row_stride_h,  # stride per head for csr_row_indices, i.e. num_rows + 1
    layout_csr_col_stride_h,  # stride per head for csr_col_indices, i.e. num_rows * num_cols
    num_layout,  # number of sparse layout (L)
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    num_heads,
    num_kv_heads,
    total_seq_len,  # Total sequence length including past sequence length and query sequence length.
    BLOCK_M: tl.constexpr,  # block size for q_seq_len
    EVEN_M: tl.constexpr,  # whether q_seq_len % BLOCK_M == 0
    BLOCK_N: tl.constexpr,  # block size for k_seq_len
    EVEN_N: tl.constexpr,  # whether k_seq_len % BLOCK_N == 0
    BLOCK_D: tl.constexpr,  # block size for D
    NUM_D_BLOCKS: tl.constexpr,  # number of data blocks =  D / BLOCK_D
):
    tl.static_print(f"{BLOCK_M=} {BLOCK_N=} {BLOCK_D=} {EVEN_M=} {EVEN_N=} {NUM_D_BLOCKS=}")

    # Past sequence length is 0 since this kernel is for prompt only.
    q_seq_len = total_seq_len

    # Grid is [CDiv(q_seq_len, BLOCK_M), batch_size * num_heads]
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)

    off_h = off_bh % num_heads
    off_b = off_bh // num_heads

    # For group query attention, map the query head index to the corresponding one for key and value.
    head_groups = num_heads // num_kv_heads
    off_h_kv = off_h // head_groups

    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h_kv * stride_kh
    V += off_b * stride_vb + off_h_kv * stride_vh

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :]  # [BLOCK_M, BLOCK_D]
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None]  # [BLOCK_D, BLOCK_N]
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :]  # [BLOCK_N, BLOCK_D]

    # Initialize pointers to query, key, value
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    if NUM_D_BLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load q: it will stay in SRAM throughout
    if EVEN_M:
        q = tl.load(q_ptrs)
        if NUM_D_BLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_D)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len)
        if NUM_D_BLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_D, mask=offs_m[:, None] < q_seq_len)

    layout_h = off_h % num_layout

    # This assumes that past sequence length is 0, otherwise need + (past_seq_len + 1) // BLOCK_M.
    layout_ptr = layout_csr_row_indices + layout_h * layout_csr_row_stride_h + start_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + 1).to(tl.int32)

    # Loop over k, v and update accumulator
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_csr_col_indices + layout_h * layout_csr_col_stride_h + col_idx_idx).to(tl.int32)
        start_n = col_idx * BLOCK_N
        # -- compute qk ----
        if EVEN_N:
            k = tl.load(k_ptrs + start_n * stride_kn)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_n[None, :] + start_n < total_seq_len)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if NUM_D_BLOCKS >= 2:
            if EVEN_N:
                k = tl.load(k_ptrs + start_n * stride_kn + BLOCK_D)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn + BLOCK_D, mask=offs_n[None, :] + start_n < total_seq_len)
            qk += tl.dot(q2, k)

        qk *= softmax_scale

        # This assumes that past sequence length is 0, otherwise need offs_m[:, None] + past_seq_len >= ...
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        if NUM_D_BLOCKS >= 2:
            acc2 = acc2 * acc_scale[:, None]
        p = p.to(Q.dtype.element_ty)
        # update acc
        if EVEN_N:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[:, None] + start_n < total_seq_len)
        acc += tl.dot(p, v)

        if NUM_D_BLOCKS >= 2:
            if EVEN_N:
                v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_D)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_D, mask=offs_n[:, None] + start_n < total_seq_len)
            acc2 += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    off_o = off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    out_ptrs = out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
    if NUM_D_BLOCKS >= 2:
        tl.store(out_ptrs + BLOCK_D, acc2, mask=offs_m[:, None] < q_seq_len)
