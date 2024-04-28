# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import triton
import triton.language as tl


@triton.jit
def block_sparse_attention(
    Out,  # output [B, M, H, D]. Note that B is batch_size, M is q_seq_len, H is num_heads, and D is head_size
    Q,  # query [B, M, H, D]
    K,  # key [B, N, H_kv, D]. Note that N is max_seq_len for kv cache, H_kv is num_kv_heads
    V,  # value [B, N, H_kv, D]
    q_batch_starts,  # [B], start position (excluding the past) of query in the sequence for each batch
    q_batch_ends,  # [B], end position (excluding the past) of query in the sequence for each batch
    k_batch_starts,  # [B], start position (including the past) of key in the sequence for each batch
    k_batch_ends,  # [B], end position (including the past) of key in the sequence for each batch
    q_batch_ids,  # [G], batch id for each query block; G is the total number of query blocks
    q_start_sids,  # [G], start position (excluding the past) of each query block
    layout_crow_ptr,  # block mask CSR format. Shape is [H, num_rows + 1] where num_rows = max_seq_len / BLOCK_M
    layout_col_ptr,  # block mask CSR format. Shape is [H, num_rows * num_cols] where num_cols = max_seq_len / BLOCK_N
    layout_crow_stride_h,  # stride per head for csr_row_indices, i.e. num_rows + 1
    layout_col_stride_h,  # stride per head for csr_col_indices, i.e. num_rows * num_cols
    stride_qb,
    stride_qt,
    stride_qh,  # strides for query (excluding the stride for last hidden dim, which is always 1)
    stride_kb,
    stride_kt,
    stride_kh,  # strides for key (excluding the stride for last hidden dim, which is always 1)
    stride_vb,
    stride_vt,
    stride_vh,  # strides for value (excluding the stride for last hidden dim, which is always 1)
    stride_ob,
    stride_ot,
    stride_oh,  # strides for output (excluding the stride for last hidden dim, which is always 1)
    q_k_ratio,  # num_heads / num_kv_heads
    num_layout,  # number of sparse layout (H)
    softmax_scale,  # scaling factor applied prior to softmax
    HAS_BATCH_DIM: tl.constexpr,  # whether batch dim is present
    D_HEAD: tl.constexpr,  # head size
    BLOCK_M: tl.constexpr,  # block size for q_seq_len
    BLOCK_N: tl.constexpr,  # block size for k_seq_len
    BLOCK_D: tl.constexpr,  # block size for D
    BLOCK_M_LOADING: tl.constexpr,  # block size for loading q
    EVEN_D: tl.constexpr,  # whether D is divisible by BLOCK_D
    M_LT_N: tl.constexpr,  # whether BLOCK_M < BLOCK_N
):
    tl.static_print(
        f"{HAS_BATCH_DIM=} {D_HEAD=} {BLOCK_M=} {BLOCK_N=} {BLOCK_D=} {BLOCK_M_LOADING=} {EVEN_D=} {M_LT_N=}"
    )
    # The grid is [G, num_heads] where G is number of query blocks.
    off_g = tl.program_id(0)
    off_h = tl.program_id(1)

    off_h_for_kv = off_h // q_k_ratio
    off_b = tl.load(q_batch_ids + off_g).to(tl.int32)
    q_start_sid = tl.load(q_start_sids + off_g)
    start_m = q_start_sid // BLOCK_M

    if HAS_BATCH_DIM:
        Q += off_b * stride_qb
        K += off_b * stride_kb
        V += off_b * stride_vb
        Out += off_b * stride_ob

    # offs_m, offs_n: storage offsets of m-dim(q, row) and n-dim(k, col)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M_LOADING)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_cu_start = tl.load(q_batch_starts + off_b).to(tl.int32)
    q_seqlen = tl.load(q_batch_ends + off_b).to(tl.int32) - q_cu_start

    k_cu_start = tl.load(k_batch_starts + off_b).to(tl.int32)
    k_seqlen = tl.load(k_batch_ends + off_b).to(tl.int32) - k_cu_start

    past_len = k_seqlen - q_seqlen

    Q += q_cu_start * stride_qt + off_h * stride_qh
    K += k_cu_start * stride_kt + off_h_for_kv * stride_kh
    V += k_cu_start * stride_vt + off_h_for_kv * stride_vh
    Out += q_cu_start * stride_ot + off_h * stride_oh

    if EVEN_D:
        q = tl.load(Q + offs_m[:, None] * stride_qt + offs_d[None, :], mask=offs_m[:, None] < q_seqlen)
    else:
        q = tl.load(
            Q + offs_m[:, None] * stride_qt + offs_d[None, :],
            mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
            other=0,
        )

    q_row = (past_len + q_start_sid) // BLOCK_M

    layout_h = off_h % num_layout
    sparse_crow_ptr = layout_crow_ptr + layout_h * layout_crow_stride_h + q_row

    # TODO: load at once, supported in new Triton
    k_block_start = tl.load(sparse_crow_ptr).to(tl.int32)
    k_block_end = tl.load(sparse_crow_ptr + 1).to(tl.int32)

    m_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M_LOADING, BLOCK_D], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kt + offs_d[:, None]
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_d[None, :]

    for k_block_col_idx in range(k_block_start, k_block_end - 1):
        k_block_id = tl.load(layout_col_ptr + layout_h * layout_col_stride_h + k_block_col_idx).to(tl.int32)
        start_n = k_block_id * BLOCK_N

        # -- compute qk ----
        if EVEN_D:
            k = tl.load(k_ptrs + start_n * stride_kt)
        else:
            k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_d[:, None] < D_HEAD)

        qk = tl.zeros([BLOCK_M_LOADING, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= softmax_scale
        if M_LT_N:
            qk += tl.where(offs_m[:, None] + past_len >= (start_n + offs_n[None, :]), 0, float("-inf"))

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

        p = p.to(Q.dtype.element_ty)

        # update acc
        if EVEN_D:
            v = tl.load(v_ptrs + start_n * stride_vt)
        else:
            v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_d[None, :] < D_HEAD)

        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # Process the last k block
    k_block_col_idx = k_block_end - 1
    k_block_id = tl.load(layout_col_ptr + layout_h * layout_col_stride_h + k_block_col_idx).to(tl.int32)
    start_n = k_block_id * BLOCK_N
    # -- compute qk ----
    if EVEN_D:
        k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < k_seqlen)
    else:
        # mask = mask & (offs_d[:, ])
        k = tl.load(
            k_ptrs + start_n * stride_kt, mask=(offs_n[None, :] + start_n < k_seqlen) & (offs_d[:, None] < D_HEAD)
        )

    qk = tl.zeros([BLOCK_M_LOADING, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= softmax_scale
    qk += tl.where(offs_m[:, None] + past_len >= (start_n + offs_n[None, :]), 0, float("-inf"))

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

    p = p.to(Q.dtype.element_ty)
    # update acc
    if EVEN_D:
        v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < k_seqlen)
    else:
        v = tl.load(
            v_ptrs + start_n * stride_vt, mask=(offs_n[:, None] + start_n < k_seqlen) & (offs_d[None, :] < D_HEAD)
        )

    acc += tl.dot(p, v)
    # l_i = l_i_new
    # m_i = m_i_new

    # write output
    if EVEN_D:
        tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :], acc, mask=offs_m[:, None] < q_seqlen)
    else:
        tl.store(
            Out + offs_m[:, None] * stride_ot + offs_d[None, :],
            acc,
            mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
        )
