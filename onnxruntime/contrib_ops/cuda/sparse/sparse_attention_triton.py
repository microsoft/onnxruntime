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
    Out,  # [Z, H, M, D]. Note that Z is batch_size, H is num_heads, M is q_seq_len, N is k_seq_len, and D is head_size.
    Q,    # [Z, H, M, D]
    K,    # [Z, H, N, D]
    V,    # [Z, H, N, D]
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h,
    layout_crow_stride_m,
    layout_col_stride_h,
    layout_col_stride_m,
    softmax_scale,  # scaling factor for softmax
    # TMP, L, M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug. TMP, L, M are assumed to have contiguous layouts
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    H,  # number of heads
    N_CTX,  # context length is past sequence length + current sequence length
    PAST_LEN,  # past sequence length
    BLOCK_M: tl.constexpr,  # block size for Q
    BLOCK_DMODEL: tl.constexpr,  # block size for D
    BLOCK_N: tl.constexpr,  # block size for K, V
    EVEN_M_BLOCK: tl.constexpr,  # whether q_seq_len % BLOCK_M == 0
    EVEN_N_BLOCK: tl.constexpr,  # whether k_seq_len % BLOCK_M == 0
    NUM_DBLOCKS: tl.constexpr,  # number of data blocks =  D / BLOCK_DMODEL
):
    Q_LEN = N_CTX - PAST_LEN

    # grid is [CDiv(q_seq_len, BLOCK_M), batch_size * num_heads]
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    # off_k = offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    # TMP/L/M are float32 (batch_size * num_heads, CDiv(q_seq_len, BLOCK_M) * BLOCK_M)
    # t_ptrs = TMP + off_hz * Q_ROUNDED_LEN + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if NUM_DBLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q: it will stay in SRAM throughout
    if EVEN_M_BLOCK:
        q = tl.load(q_ptrs)
        if NUM_DBLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)
        if NUM_DBLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd, mask=offs_m[:, None] < Q_LEN)

    layout_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + start_m * layout_crow_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_crow_stride_m).to(tl.int32)

    # loop over k, v and update accumulator
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_col_ptr + off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
        start_n = col_idx * BLOCK_N
        # -- compute qk ----
        if EVEN_N_BLOCK:
            k = tl.load(k_ptrs + start_n * stride_kn)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_n[None, :] + start_n < N_CTX)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if NUM_DBLOCKS >= 2:
            if EVEN_N_BLOCK:
                k = tl.load(k_ptrs + start_n * stride_kn + BLOCK_DMODEL * stride_kd)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn + BLOCK_DMODEL * stride_kd, mask=offs_n[None, :] + start_n < N_CTX
                )
            qk += tl.dot(q2, k)

        qk *= softmax_scale
        qk += tl.where(offs_m[:, None] + PAST_LEN >= (start_n + offs_n[None, :]), 0, float("-inf"))

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
        if NUM_DBLOCKS >= 2:
            acc2 = acc2 * acc_scale[:, None]
        p = p.to(Q.dtype.element_ty)

        # update acc
        if EVEN_N_BLOCK:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[:, None] + start_n < N_CTX)
        acc += tl.dot(p, v)

        if NUM_DBLOCKS >= 2:
            if EVEN_N_BLOCK:
                v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_DMODEL * stride_vd)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn + BLOCK_DMODEL * stride_vd, mask=offs_n[:, None] + start_n < N_CTX
                )
            acc2 += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # initialize pointers to output
    # offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < Q_LEN)
    if NUM_DBLOCKS >= 2:
        tl.store(out_ptrs + BLOCK_DMODEL * stride_od, acc2, mask=offs_m[:, None] < Q_LEN)


dtypes = ["fp16"]
block_n_values = [16, 32, 64, 128]
block_d_values = [64]  # support head_size = 128
num_block_d_values = [2]
even_m_values = [True, False]  # TODO: shall we use padding to make it True always?
even_n_values = [True, False]  # TODO: shall we use padding to make it True always?
name_pattern = "BlockSparseAttentionTriton_{}_m{}_n{}_d{}_{}_em{}_en{}"
sig_pattern = "*{},*{},*{},*{},*{},*fp32,*fp32,i32,i32,i32,fp32,i1,i1,i1"
group_pattern = "BlockSparseAttentionTriton_{}"


def get_function_table():
    func_table = []

    for dtype, block_n, block_d, num_blocks_d, even_m, even_n in product(
        dtypes, block_n_values, block_d_values, num_block_d_values, even_m_values, even_n_values
    ):
        for block_m in [16, block_n]:
            name = name_pattern.format(dtype, block_m, block_n, block_d, num_blocks_d, even_m, even_n)

            # head_size = block_d * num_blocks_d
            # group = group_pattern.format(head_size, dtype)
            group = group_pattern.format(dtype)

            sig = sig_pattern.format(dtype, dtype, dtype, dtype, dtype)
            kwargs = {
                "num_warps": max(1, 2 ** int(math.log2(min(block_m, block_n, block_d) / 16))),
                "constants": {
                    "BLOCK_M": block_m,
                    "BLOCK_DMODEL": block_d,
                    "BLOCK_N": block_n,
                    "EVEN_M_BLOCK": int(even_m),
                    "EVEN_N_BLOCK": int(even_n),
                    "NUM_DBLOCKS": num_blocks_d,
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
