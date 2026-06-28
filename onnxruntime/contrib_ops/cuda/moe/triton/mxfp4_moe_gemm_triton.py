# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""MXFP4 (e2m1, group=32) grouped GEMM Triton kernel — tensor-core spike.

Tensor-core grouped GEMM with in-mainloop e2m1 dequant, matching the ORT QMoE
FP4 weight layout so an AOT cubin can later drop into moe_quantization.cc
(like SparseAttention v1=prefill / v2=decode). One program = one expert's
[BLOCK_M rows, BLOCK_N] output tile, full-K mainloop. Decode uses BLOCK_M=16
(4 active rows padded); prefill can reuse with larger M tiles.

Weight layout (mirrors ORT prepack):
  - W: [E, N, K//2] uint8 row-major, two e2m1 codes/byte, even-K in low nibble.
  - S: [E, K//32, N] fp16 block scales (already folded with per-expert global).
"""

import triton
import triton.language as tl


@triton.jit
def _e2m1_to_f32(nib):
    sign = (nib & 0x8) != 0
    m = nib & 0x7
    v = tl.where(m == 0, 0.0, tl.where(m == 1, 0.5, tl.where(m == 2, 1.0,
        tl.where(m == 3, 1.5, tl.where(m == 4, 2.0, tl.where(m == 5, 3.0,
        tl.where(m == 6, 4.0, 6.0)))))))
    return tl.where(sign, -v, v)


@triton.jit
def mxfp4_grouped_gemm(
    A, W, S, rows_to_expert, Out,
    M, N, K,
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_se, stride_sk, stride_sn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    e0 = tl.load(rows_to_expert + pid_m * BLOCK_M)  # all rows in an M-tile share one expert
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        a = tl.load(A + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
        byte = tl.load(W + e0 * stride_we + offs_n[None, :] * stride_wn + (kk[:, None] // 2) * stride_wk,
                       mask=(offs_n[None, :] < N) & (kk[:, None] < K), other=0)
        nib = tl.where((kk[:, None] & 1) == 0, byte & 0xF, (byte >> 4) & 0xF)
        s = tl.load(S + e0 * stride_se + (kk[:, None] // 32) * stride_sk + offs_n[None, :] * stride_sn,
                    mask=(offs_n[None, :] < N) & (kk[:, None] < K), other=0.0)
        w = (_e2m1_to_f32(nib) * s).to(tl.float16)
        acc += tl.dot(a, w)
    tl.store(Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
