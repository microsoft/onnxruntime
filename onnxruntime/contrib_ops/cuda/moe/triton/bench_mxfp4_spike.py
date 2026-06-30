# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Validate + benchmark the MXFP4 grouped GEMM Triton spike on GPT-OSS decode shapes.

Builds random MXFP4 weights in the ORT prepack layout, runs the Triton kernel,
checks against a dequant reference, and times per-expert FC1 to compare with the
fused CUDA GEMV (~65 us/expert measured by ncu, ~264 e2e tps).
"""

import torch
import triton
from mxfp4_moe_gemm_triton import mxfp4_grouped_gemm

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def make_mxfp4(E, N, K, dev):
    codes = torch.randint(0, 16, (E, N, K), device=dev, dtype=torch.int32)
    lut = _E2M1.to(dev)
    scales = (torch.rand(E, K // 32, N, device=dev) * 0.1 + 0.05).to(torch.float16)
    packed = (codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)).to(torch.uint8)  # [E,N,K//2] low=even
    decoded = lut[codes]  # [E,N,K]
    s_full = scales.transpose(1, 2).repeat_interleave(32, dim=2).to(torch.float32)  # [E,N,K]
    W = (decoded * s_full).to(torch.float16)  # ref dense weight [E,N,K]
    return packed, scales, W


def ref(A, W, r2e):
    return torch.stack([A[i].float() @ W[r2e[i]].T.float() for i in range(A.shape[0])]).to(torch.float16)


def run(M, N, K, E, BM=16, BN=64, BK=64, w=4, stages=3):
    dev = "cuda"
    A = (torch.randn(M, K, device=dev) * 0.05).to(torch.float16)
    r2e = torch.zeros(M, device=dev, dtype=torch.int32)  # one expert per M-tile (grouped)
    packed, scales, W = make_mxfp4(E, N, K, dev)
    Out = torch.empty(M, N, device=dev, dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    mxfp4_grouped_gemm[grid](
        A,
        packed,
        scales,
        r2e,
        Out,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        *packed.stride(),
        *scales.stride(),
        Out.stride(0),
        Out.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=w,
        num_stages=stages,
    )
    R = ref(A, W, r2e)
    err = (Out.float() - R.float()).abs().max().item()
    rel = err / (R.float().abs().max().item() + 1e-6)
    ms = triton.testing.do_bench(
        lambda: mxfp4_grouped_gemm[grid](
            A,
            packed,
            scales,
            r2e,
            Out,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            *packed.stride(),
            *scales.stride(),
            Out.stride(0),
            Out.stride(1),
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            num_warps=w,
            num_stages=stages,
        )
    )
    print(
        f"M={M} N={N} K={K} E={E} BN={BN} BK={BK} w={w} st={stages} | rel={rel:.4f} | {ms * 1000:.1f} us "
        f"({ms * 1000 / M:.1f} us/row)"
    )


if __name__ == "__main__":
    # GPT-OSS-20B: hidden=2880, inter=2880, topk=4. Decode FC1 N=2*inter, FC2 N=hidden.
    run(M=4, N=5760, K=2880, E=32)  # FC1 gate+up, 4 active experts
    run(M=4, N=2880, K=2880, E=32)  # FC2
    print("--- sweep (FC1 m=4) ---")
    for bn in (32, 64, 128):
        for bk in (32, 64, 128):
            for w in (4, 8):
                for stages in (2, 3):
                    try:
                        run(M=4, N=5760, K=2880, E=32, BM=16, BN=bn, BK=bk, w=w, stages=stages)
                    except Exception as ex:
                        print(f"M=4 N=5760 K=2880 BN={bn} BK={bk} w={w} st={stages} | fail: {type(ex).__name__}: {ex}")
