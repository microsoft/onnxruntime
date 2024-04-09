# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math
from itertools import product

import triton
import triton.language as tl


@triton.jit
def block_sparse_attention_kernel(
    out,  # [B, H, M, D]. Note that B is batch_size, H is num_heads, M is q_seq_len, and D is head_size
    query,  # [B, H, M, D]
    key,  # [B, H, N, D]. Note that N is k_seq_len (i.e. total_seq_len), k_num_heads need to expand to num_heads
    value,  # [B, H, N, D]
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h,
    layout_crow_stride_m,
    layout_col_stride_h,
    layout_col_stride_m,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,  # TODO: remove strides for D since it is always 1
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    num_heads,
    total_seq_len,
    past_seq_len,
    BLOCK_M: tl.constexpr,  # block size for q_seq_len
    BLOCK_D: tl.constexpr,  # block size for D
    BLOCK_N: tl.constexpr,  # block size for k_seq_len
    EVEN_M: tl.constexpr,  # whether q_seq_len % BLOCK_M == 0
    EVEN_N: tl.constexpr,  # whether k_seq_len % BLOCK_M == 0
    NUM_D_BLOCKS: tl.constexpr,  # number of data blocks =  D / BLOCK_D
):
    q_seq_len = total_seq_len - past_seq_len

    # Grid is [CDiv(q_seq_len, BLOCK_M), batch_size * num_heads]
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)

    off_h = off_bh % num_heads
    off_b = off_bh // num_heads
    query += off_b * stride_qb + off_h * stride_qh
    key += off_b * stride_kb + off_h * stride_kh
    value += off_b * stride_vb + off_h * stride_vh

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    # off_k = offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

    # Initialize pointers to Q, K, V
    q_ptrs = query + off_q
    k_ptrs = key + off_k
    v_ptrs = value + off_v

    # Initialize pointer to m and l
    # TMP/L/M are float32 (batch_size * num_heads, CDiv(q_seq_len, BLOCK_M) * BLOCK_M)
    # t_ptrs = TMP + off_bh * Q_ROUNDED_LEN + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    if NUM_D_BLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load q: it will stay in SRAM throughout
    if EVEN_M:
        q = tl.load(q_ptrs)
        if NUM_D_BLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_D * stride_qd)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len)
        if NUM_D_BLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_D * stride_qd, mask=offs_m[:, None] < q_seq_len)

    layout_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + start_m * layout_crow_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_crow_stride_m).to(tl.int32)

    # loop over k, v and update accumulator
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_col_ptr + off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
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
                k = tl.load(k_ptrs + start_n * stride_kn + BLOCK_D * stride_kd)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn + BLOCK_D * stride_kd, mask=offs_n[None, :] + start_n < total_seq_len
                )
            qk += tl.dot(q2, k)

        qk *= softmax_scale
        qk += tl.where(offs_m[:, None] + past_seq_len >= (start_n + offs_n[None, :]), 0, float("-inf"))

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

        # tl.store(t_ptrs, acc_scale)
        # acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        if NUM_D_BLOCKS >= 2:
            acc2 = acc2 * acc_scale[:, None]
        p = p.to(query.dtype.element_ty)

        # update acc
        if EVEN_N:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[:, None] + start_n < total_seq_len)
        acc += tl.dot(p, v)

        if NUM_D_BLOCKS >= 2:
            if EVEN_N:
                v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_D * stride_vd)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn + BLOCK_D * stride_vd, mask=offs_n[:, None] + start_n < total_seq_len
                )
            acc2 += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # initialize pointers to output
    # offs_n = tl.arange(0, BLOCK_D)
    off_o = off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_ptrs = out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
    if NUM_D_BLOCKS >= 2:
        tl.store(out_ptrs + BLOCK_D * stride_od, acc2, mask=offs_m[:, None] < q_seq_len)


dtypes = ["fp16"]
block_n_values = [16, 32, 64, 128]
block_d_values = [64]  # support head_size = 128
num_block_d_values = [2]
even_m_values = [True, False]  # TODO: shall we use padding to make it True always?
even_n_values = [True, False]  # TODO: shall we use padding to make it True always?
name_pattern = "BlockSparseAttentionTriton_{}_m{}_n{}_d{}_{}_em{}_en{}"
sig_pattern = "*{},*{},*{},*{},*i32,*i32,i32,i32,i32,i32,fp32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32"
group_pattern = "BlockSparseAttentionTriton_{}"


def get_function_table():
    func_table = []

    for dtype, block_n, block_d, num_blocks_d, even_m, even_n in product(
        dtypes, block_n_values, block_d_values, num_block_d_values, even_m_values, even_n_values
    ):
        for block_m in [16, block_n]:
            name = name_pattern.format(dtype, block_m, block_n, block_d, num_blocks_d, int(even_m), int(even_n))

            # head_size = block_d * num_blocks_d
            # group = group_pattern.format(head_size, dtype)
            group = group_pattern.format(dtype)

            sig = sig_pattern.format(dtype, dtype, dtype, dtype)
            kwargs = {
                "num_warps": max(1, 2 ** int(math.log2(min(block_m, block_n, block_d) / 16))),
                "constants": {
                    "BLOCK_M": block_m,
                    "BLOCK_D": block_d,
                    "BLOCK_N": block_n,
                    "EVEN_M": int(even_m),
                    "EVEN_N": int(even_n),
                    "NUM_D_BLOCKS": num_blocks_d,
                },
            }

            func_desc = {
                "name": name,
                "group": group,
                "func": block_sparse_attention_kernel,
                "sig": sig,
                "kwargs": kwargs,
            }
            func_table.append(func_desc)
    return func_table


if __name__ == "__main__":
    func_table = get_function_table()
    for func_desc in func_table:
        print(func_desc)
