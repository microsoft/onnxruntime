/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon_fp16.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON specific to
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE CompFp16.

--*/

#include <arm_neon.h>

#include <cassert>

#include "fp16_common.h"
#include "sqnbitgemm.h"
#include "sqnbitgemm_kernel_neon.h"

namespace sqnbitgemm_neon
{

MLAS_FORCEINLINE void
Transpose8x8(uint8x8_t& v0, uint8x8_t& v1, uint8x8_t& v2, uint8x8_t& v3,
             uint8x8_t& v4, uint8x8_t& v5, uint8x8_t& v6, uint8x8_t& v7)
{
    // v0: | B00 B10 | B20 B30 | B40 B50 | B60 B70 | B80 B90 | Ba0 Bb0 | Bc0 Bd0 | Be0 Bf0 |
    // v1: | B01 B11 | B21 B31 | B41 B51 | B61 B71 | B81 B91 | Ba1 Bb1 | Bc1 Bd1 | Be1 Bf1 |
    // v2: | B02 B12 | B22 B32 | B42 B52 | B62 B72 | B82 B92 | Ba2 Bb2 | Bc2 Bd2 | Be2 Bf2 |
    // v3: | B03 B13 | B23 B33 | B43 B53 | B63 B73 | B83 B93 | Ba3 Bb3 | Bc3 Bd3 | Be3 Bf3 |
    // v4: | B04 B14 | B24 B34 | B44 B54 | B64 B74 | B84 B94 | Ba4 Bb4 | Bc4 Bd4 | Be4 Bf4 |
    // v5: | B05 B15 | B25 B35 | B45 B55 | B65 B75 | B85 B95 | Ba5 Bb5 | Bc5 Bd5 | Be5 Bf5 |
    // v6: | B06 B16 | B26 B36 | B46 B56 | B66 B76 | B86 B96 | Ba6 Bb6 | Bc6 Bd6 | Be6 Bf6 |
    // v7: | B07 B17 | B27 B37 | B47 B57 | B67 B77 | B87 B97 | Ba7 Bb7 | Bc7 Bd7 | Be7 Bf7 |

    uint8x8x2_t a0 = vtrn_u8(v0, v1);
    uint8x8x2_t a1 = vtrn_u8(v2, v3);
    uint8x8x2_t a2 = vtrn_u8(v4, v5);
    uint8x8x2_t a3 = vtrn_u8(v6, v7);

    // a0[0]: | B00 B10 | B01 B11 | B40 B50 | B41 B51 | B80 B90 | B81 B91 | Bc0 Bd0 | Bc1 Bd1 |
    // a0[1]: | B20 B30 | B21 B31 | B60 B70 | B61 B71 | Ba0 Bb0 | Ba1 Bb1 | Be0 Bf0 | Be1 Bf1 |
    // a1[0]: | B02 B12 | B03 B13 | B42 B52 | B43 B53 | B82 B92 | B83 B93 | Bc2 Bd2 | Bc3 Bd3 |
    // a1[1]: | B22 B32 | B23 B33 | B62 B72 | B63 B73 | Ba2 Bb2 | Ba3 Bb3 | Be2 Bf2 | Be3 Bf3 |
    // a2[0]: | B04 B14 | B05 B15 | B44 B54 | B45 B55 | B84 B94 | B85 B95 | Bc4 Bd4 | Bc5 Bd5 |
    // a2[1]: | B24 B34 | B25 B35 | B64 B74 | B65 B75 | Ba4 Bb4 | Ba5 Bb5 | Be4 Bf4 | Be5 Bf5 |
    // a3[0]: | B06 B16 | B07 B17 | B46 B56 | B47 B57 | B86 B96 | B87 B97 | Bc6 Bd6 | Bc7 Bd7 |
    // a3[1]: | B26 B36 | B27 B37 | B66 B76 | B67 B77 | Ba6 Bb6 | Ba7 Bb7 | Be6 Bf6 | Be7 Bf7 |

    uint16x4x2_t b0 = vtrn_u16(vreinterpret_u16_u8(a0.val[0]), vreinterpret_u16_u8(a1.val[0]));
    uint16x4x2_t b1 = vtrn_u16(vreinterpret_u16_u8(a0.val[1]), vreinterpret_u16_u8(a1.val[1]));
    uint16x4x2_t b2 = vtrn_u16(vreinterpret_u16_u8(a2.val[0]), vreinterpret_u16_u8(a3.val[0]));
    uint16x4x2_t b3 = vtrn_u16(vreinterpret_u16_u8(a2.val[1]), vreinterpret_u16_u8(a3.val[1]));

    // b0[0]: | B00 B10 | B01 B11 | B02 B12 | B03 B13 | B80 B90 | B81 B91 | B82 B92 | B83 B93 |
    // b0[1]: | B40 B50 | B41 B51 | B42 B52 | B43 B53 | Bc0 Bd0 | Bc1 Bd1 | Bc2 Bd2 | Bc3 Bd3 |
    // b1[0]: | B20 B30 | B21 B31 | B22 B32 | B23 B33 | Ba0 Bb0 | Ba1 Bb1 | Ba2 Bb2 | Ba3 Bb3 |
    // b1[1]: | B60 B70 | B61 B71 | B62 B72 | B63 B73 | Be0 Bf0 | Be1 Bf1 | Be2 Bf2 | Be3 Bf3 |
    // b2[0]: | B04 B14 | B05 B15 | B06 B16 | B07 B17 | B84 B94 | B85 B95 | B86 B96 | B87 B97 |
    // b2[1]: | B44 B54 | B45 B55 | B46 B56 | B47 B57 | Bc4 Bd4 | Bc5 Bd5 | Bc6 Bd6 | Bc7 Bd7 |
    // b3[0]: | B24 B34 | B25 B35 | B26 B36 | B27 B37 | Ba4 Bb4 | Ba5 Bb5 | Ba6 Bb6 | Ba7 Bb7 |
    // b3[1]: | B64 B74 | B65 B75 | B66 B76 | B67 B77 | Be4 Bf4 | Be5 Bf5 | Be6 Bf6 | Be7 Bf7 |

    uint32x2x2_t c0 = vtrn_u32(vreinterpret_u32_u16(b0.val[0]), vreinterpret_u32_u16(b2.val[0]));
    uint32x2x2_t c1 = vtrn_u32(vreinterpret_u32_u16(b0.val[1]), vreinterpret_u32_u16(b2.val[1]));
    uint32x2x2_t c2 = vtrn_u32(vreinterpret_u32_u16(b1.val[0]), vreinterpret_u32_u16(b3.val[0]));
    uint32x2x2_t c3 = vtrn_u32(vreinterpret_u32_u16(b1.val[1]), vreinterpret_u32_u16(b3.val[1]));

    // c0[0]: | B00 B10 | B01 B11 | B02 B12 | B03 B13 | B04 B14 | B05 B15 | B06 B16 | B07 B17 |
    // c0[1]: | B80 B90 | B81 B91 | B92 B92 | B83 B93 | B84 B94 | B85 B95 | B86 B96 | B87 B97 |
    // c1[0]: | B40 B50 | B41 B51 | B42 B52 | B43 B53 | B44 B54 | B45 B55 | B46 B56 | B47 B57 |
    // c1[1]: | Bc0 Bd0 | Bc1 Bd1 | Bc2 Bd2 | Bc3 Bd3 | Bc4 Bd4 | Bc5 Bd5 | Bc6 Bd6 | Bc7 Bd7 |
    // c2[0]: | B20 B30 | B21 B31 | B22 B32 | B23 B33 | B24 B34 | B25 B35 | B26 B36 | B27 B37 |
    // c2[1]: | Ba0 Bb0 | Ba1 Bb1 | Ba2 Bb2 | Ba3 Bb3 | Ba4 Bb4 | Ba5 Bb5 | Ba6 Bb6 | Ba7 Bb7 |
    // c3[0]: | B60 B70 | B61 B71 | B62 B72 | B63 B73 | B64 B74 | B65 B75 | B66 B76 | B67 B77 |
    // c3[1]: | Be0 Bf0 | Be1 Bf1 | Be2 Bf2 | Be3 Bf3 | Be4 Bf4 | Be5 Bf5 | Be6 Bf6 | Be7 Bf7 |

    v0 = vreinterpret_u8_u32(c0.val[0]);
    v1 = vreinterpret_u8_u32(c2.val[0]);
    v2 = vreinterpret_u8_u32(c1.val[0]);
    v3 = vreinterpret_u8_u32(c3.val[0]);
    v4 = vreinterpret_u8_u32(c0.val[1]);
    v5 = vreinterpret_u8_u32(c2.val[1]);
    v6 = vreinterpret_u8_u32(c1.val[1]);
    v7 = vreinterpret_u8_u32(c3.val[1]);
}

void
SQ4BitGemmPackQuantBData_CompFp16(
    size_t N,
    size_t K,
    size_t BlkLen,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    constexpr size_t nbits = 4;
    constexpr size_t k_blk_dim = 16;
    constexpr size_t n_blk_dim = 8;
    assert(BlkLen > 0 && BlkLen % k_blk_dim == 0);

    const size_t k_blk_num = MlasDivRoundup(K, k_blk_dim);
    const size_t n_blk_num = MlasDivRoundup(N, n_blk_dim);
    const size_t k_blk_bytes = MlasQNBitBlkDataSizeInBytes(nbits, BlkLen);
    const size_t iterations = k_blk_num * n_blk_num; // one iteration per block
    const size_t ld = k_blk_num * k_blk_bytes;

    //
    // For blocks 16_K * 8_N, transpose bytes in 8x8 blocks like this:
    // src B_k_n:
    // | B00 B10 | B20 B30 | B40 B50 | B60 B70 | B80 B90 | Ba0 Bb0 | Bc0 Bd0 | Be0 Bf0 |
    // | B01 B11 | B21 B31 | B41 B51 | B61 B71 | B81 B91 | Ba1 Bb1 | Bc1 Bd1 | Be1 Bf1 |
    // | B02 B12 | B22 B32 | B42 B52 | B62 B72 | B82 B92 | Ba2 Bb2 | Bc2 Bd2 | Be2 Bf2 |
    // | B03 B13 | B23 B33 | B43 B53 | B63 B73 | B83 B93 | Ba3 Bb3 | Bc3 Bd3 | Be3 Bf3 |
    // | B04 B14 | B24 B34 | B44 B54 | B64 B74 | B84 B94 | Ba4 Bb4 | Bc4 Bd4 | Be4 Bf4 |
    // | B05 B15 | B25 B35 | B45 B55 | B65 B75 | B85 B95 | Ba5 Bb5 | Bc5 Bd5 | Be5 Bf5 |
    // | B06 B16 | B26 B36 | B46 B56 | B66 B76 | B86 B96 | Ba6 Bb6 | Bc6 Bd6 | Be6 Bf6 |
    // | B07 B17 | B27 B37 | B47 B57 | B67 B77 | B87 B97 | Ba7 Bb7 | Bc7 Bd7 | Be7 Bf7 |
    // => dst:
    // | B00 B10 | B01 B11 | B02 B12 | B03 B13 | B04 B14 | B05 B15 | B06 B16 | B07 B17 |
    // | B20 B30 | B21 B31 | B22 B32 | B23 B33 | B24 B34 | B25 B35 | B26 B36 | B27 B37 |
    // | B40 B50 | B41 B51 | B42 B52 | B43 B53 | B44 B54 | B45 B55 | B46 B56 | B47 B57 |
    // | B60 B70 | B61 B71 | B62 B72 | B63 B73 | B64 B74 | B65 B75 | B66 B76 | B67 B77 |
    // | B80 B90 | B81 B91 | B92 B92 | B83 B93 | B84 B94 | B85 B95 | B86 B96 | B87 B97 |
    // | Ba0 Bb0 | Ba1 Bb1 | Ba2 Bb2 | Ba3 Bb3 | Ba4 Bb4 | Ba5 Bb5 | Ba6 Bb6 | Ba7 Bb7 |
    // | Bc0 Bd0 | Bc1 Bd1 | Bc2 Bd2 | Bc3 Bd3 | Bc4 Bd4 | Bc5 Bd5 | Bc6 Bd6 | Bc7 Bd7 |
    // | Be0 Bf0 | Be1 Bf1 | Be2 Bf2 | Be3 Bf3 | Be4 Bf4 | Be5 Bf5 | Be6 Bf6 | Be7 Bf7 |
    //

    //
    // For blocks < 8_N:
    // src: | v0 v1 | v2 v3 | v4 v5 | v6 v7 | v8 v9 | vA vB | vC vD | vE vF |
    //   =>
    // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
    //

    MlasTrySimpleParallel(
        ThreadPool, iterations,
        [&](ptrdiff_t tid) {
            const size_t n_blk = tid / k_blk_num;
            const size_t k_blk = tid % k_blk_num;
            const size_t n = n_blk * n_blk_dim;

            if (n + n_blk_dim <= N) {
                const size_t data_offset = n * ld + k_blk * k_blk_bytes * n_blk_dim;
                const uint8_t* src = reinterpret_cast<const uint8_t*>(QuantBDataBegin) + data_offset;
                uint8_t* dst = reinterpret_cast<uint8_t*>(PackedQuantBDataBegin) + data_offset;

                uint8x8_t v0 = vld1_u8(src);
                uint8x8_t v1 = vld1_u8(src + ld);
                uint8x8_t v2 = vld1_u8(src + 2*ld);
                uint8x8_t v3 = vld1_u8(src + 3*ld);
                uint8x8_t v4 = vld1_u8(src + 4*ld);
                uint8x8_t v5 = vld1_u8(src + 5*ld);
                uint8x8_t v6 = vld1_u8(src + 6*ld);
                uint8x8_t v7 = vld1_u8(src + 7*ld);

                Transpose8x8(v0, v1, v2, v3, v4, v5, v6, v7);

                vst1_u8(dst, v0);
                vst1_u8(dst + 8, v1);
                vst1_u8(dst + 16, v2);
                vst1_u8(dst + 24, v3);
                vst1_u8(dst + 32, v4);
                vst1_u8(dst + 40, v5);
                vst1_u8(dst + 48, v6);
                vst1_u8(dst + 56, v7);
            } else {
                const size_t data_offset = n * ld + k_blk * k_blk_bytes;
                const uint8_t* src = reinterpret_cast<const uint8_t*>(QuantBDataBegin) + data_offset;
                uint8_t* dst = reinterpret_cast<uint8_t*>(PackedQuantBDataBegin) + data_offset;

                for (; n < N; ++n, src += ld, dst += ld) {
                    uint8x8_t v0 = vld1_u8(src);
                    uint8x8_t v_even = vand_u8(v0, vdup_n_u8(0x0F));
                    uint8x8_t v_odd = vshr_n_u8(v0, 4);
                    uint8x8x2_t v1 = vzip_u8(v_even, v_odd);
                    uint8x8_t v2 = vorr_u8(v1.val[0], vshl_n_u8(v1.val[1], 4));
                    vst1_u8(dst, v2);
                }
            }
        }
    );
}

MLAS_FORCEINLINE
void
Q4BitBlkDequantB_16K_8N(
    const std::byte* src_ptr,
    const float16x8_t& scale,
    const float16x8_t& neg_scaled_zp,
    MLAS_FP16* dst_ptr
) {
    constexpr uint8x8_t low_mask = vdup_n_u8(0x0F);

    uint8x8_t b01 = vld1_u8(src_ptr);
    uint8x8_t b23 = vld1_u8(src_ptr + 8);
    uint8x8_t b45 = vld1_u8(src_ptr + 16);
    uint8x8_t b67 = vld1_u8(src_ptr + 24);
    uint8x8_t b89 = vld1_u8(src_ptr + 32);
    uint8x8_t bab = vld1_u8(src_ptr + 40);
    uint8x8_t bcd = vld1_u8(src_ptr + 48);
    uint8x8_t bef = vld1_u8(src_ptr + 56);

    float16x8_t b0 = vcvtq_f16_u16(vshll_n_u8(vand_u8(b01, low_mask), 0));
    float16x8_t b1 = vcvtq_f16_u16(vshll_n_u8(vshr_n_u8(b01, 4), 0));
    float16x8_t b2 = vcvtq_f16_u16(vshll_n_u8(vand_u8(b23, low_mask), 0));
    float16x8_t b3 = vcvtq_f16_u16(vshll_n_u8(vshr_n_u8(b23, 4), 0));
    float16x8_t b4 = vcvtq_f16_u16(vshll_n_u8(vand_u8(b45, low_mask), 0));
    float16x8_t b5 = vcvtq_f16_u16(vshll_n_u8(vshr_n_u8(b45, 4), 0));
    float16x8_t b6 = vcvtq_f16_u16(vshll_n_u8(vand_u8(b67, low_mask), 0));
    float16x8_t b7 = vcvtq_f16_u16(vshll_n_u8(vshr_n_u8(b67, 4), 0));
    float16x8_t b8 = vcvtq_f16_u16(vshll_n_u8(vand_u8(b89, low_mask), 0));
    float16x8_t b9 = vcvtq_f16_u16(vshll_n_u8(vshr_n_u8(b89, 4), 0));
    float16x8_t ba = vcvtq_f16_u16(vshll_n_u8(vand_u8(bab, low_mask), 0));
    float16x8_t bb = vcvtq_f16_u16(vshll_n_u8(vshr_n_u8(bab, 4), 0));
    float16x8_t bc = vcvtq_f16_u16(vshll_n_u8(vand_u8(bcd, low_mask), 0));
    float16x8_t bd = vcvtq_f16_u16(vshll_n_u8(vshr_n_u8(bcd, 4), 0));
    float16x8_t be = vcvtq_f16_u16(vshll_n_u8(vand_u8(bef, low_mask), 0));
    float16x8_t bf = vcvtq_f16_u16(vshll_n_u8(vshr_n_u8(bef, 4), 0));

    float16x8_t c0 = vfmaq_f16(neg_scaled_zp, b0, scale);
    float16x8_t c1 = vfmaq_f16(neg_scaled_zp, b1, scale);
    float16x8_t c2 = vfmaq_f16(neg_scaled_zp, b2, scale);
    float16x8_t c3 = vfmaq_f16(neg_scaled_zp, b3, scale);
    float16x8_t c4 = vfmaq_f16(neg_scaled_zp, b4, scale);
    float16x8_t c5 = vfmaq_f16(neg_scaled_zp, b5, scale);
    float16x8_t c6 = vfmaq_f16(neg_scaled_zp, b6, scale);
    float16x8_t c7 = vfmaq_f16(neg_scaled_zp, b7, scale);
    float16x8_t c8 = vfmaq_f16(neg_scaled_zp, b8, scale);
    float16x8_t c9 = vfmaq_f16(neg_scaled_zp, b9, scale);
    float16x8_t ca = vfmaq_f16(neg_scaled_zp, ba, scale);
    float16x8_t cb = vfmaq_f16(neg_scaled_zp, bb, scale);
    float16x8_t cc = vfmaq_f16(neg_scaled_zp, bc, scale);
    float16x8_t cd = vfmaq_f16(neg_scaled_zp, bd, scale);
    float16x8_t ce = vfmaq_f16(neg_scaled_zp, be, scale);
    float16x8_t cf = vfmaq_f16(neg_scaled_zp, bf, scale);

    vst1q_f16(dst_ptr, c0);
    vst1q_f16(dst_ptr + 8, c1);
    vst1q_f16(dst_ptr + 16, c2);
    vst1q_f16(dst_ptr + 24, c3);
    vst1q_f16(dst_ptr + 32, c4);
    vst1q_f16(dst_ptr + 40, c5);
    vst1q_f16(dst_ptr + 48, c6);
    vst1q_f16(dst_ptr + 56, c7);
    vst1q_f16(dst_ptr + 64, c8);
    vst1q_f16(dst_ptr + 72, c9);
    vst1q_f16(dst_ptr + 80, ca);
    vst1q_f16(dst_ptr + 88, cb);
    vst1q_f16(dst_ptr + 96, cc);
    vst1q_f16(dst_ptr + 104, cd);
    vst1q_f16(dst_ptr + 112, ce);
    vst1q_f16(dst_ptr + 120, cf);
}

MLAS_FORCEINLINE
void
Q4BitBlkDequantB_16K_1N(
    const std::byte* src_ptr,
    const float16x8_t& scale,
    const float16x8_t& neg_scaled_zp,
    MLAS_FP16* dst_ptr
) {
    constexpr uint8x8_t low_mask = vdup_n_u8(0x0F);

    uint8x8_t v0 = vld1_u8(src_ptr);

    float16x8_t f_low = vcvtq_f16_u16(vshll_n_u8(vand_u8(v0, low_mask), 0));
    float16x8_t f_high = vcvtq_f16_u16(vshll_n_u8(vshr_n_u8(v0, 4), 0));

    float16x8_t c0 = vfmaq_f16(neg_scaled_zp, f_low, scale);
    float16x8_t c1 = vfmaq_f16(neg_scaled_zp, f_high, scale);

    vst1q_f16(dst_ptr, c0);
    vst1q_f16(dst_ptr + 8, c1);
}

void
Q4BitBlkDequantBForSgemm_CompFp16(
    size_t BlkLen,
    MLAS_FP16* FpData,
    const std::byte* QuantBData,
    const MLAS_FP16* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t K,
    size_t BlockCountK
) {
    constexpr size_t nbits = 4;
    constexpr size_t kk_blk_dim = 16;
    constexpr size_t n_blk_dim = 8;
    assert(BlkLen > 0 && BlkLen % kk_blk_dim == 0);

    const size_t kk_blk_num = BlockCountK * BlkLen / kk_blk_dim;
    const size_t kk_blk_bytes = MlasQNBitBlkDataSizeInBytes(nbits, kk_blk_dim);
    const size_t kk_n_src_bytes = kk_blk_bytes * n_blk_dim;
    const size_t kk_n_dst_size = kk_blk_dim * n_blk_dim;
    const size_t ld_blk_src = kk_blk_num * kk_n_src_bytes;
    const size_t ld_blk_dst = BlkLen * BlockCountK * n_blk_dim;
    const size_t ld_blk_scale = BlockCountK * n_blk_dim;
    const size_t ld_zp = (BlockCountK + 1) / 2;
    const size_t ld_blk_zp = ld_zp * n_blk_dim;
    constexpr float16x8_t zp_mid_point_vec = vdupq_n_f16(8.0f);

    int n = 0;
    for (; n + n_blk_dim <= CountN; n += n_blk_dim) {
        const MLAS_FP16* scales_ptr = QuantBScale;
        const std::byte* zero_points_ptr = QuantBZeroPoint;
        const std::byte* src_ptr = QuantBData;
        MLAS_FP16* dst_ptr = FpData;

        for (int k_blk_i = 0; k_blk_i < BlockCountK; ++k_blk_i) {
            // prepare scales and zero_points for the block
            MLAS_FP16 scales[n_blk_dim];
            uint16_t zero_points[n_blk_dim];
            float16x8_t scale_vec;
            float16x8_t neg_scaled_zp_vec;

            UnrolledLoop<n_blk_dim>([&](int nn){
                scales[nn] = scales_ptr[nn * BlockCountK];
            });
            scale_vec = vld1q_f16(scales);

            if (QuantBZeroPoint) {
                UnrolledLoop<n_blk_dim>([&](int nn){
                    uint8_t zp = zero_points_ptr[nn * ld_zp];
                    zp = (k_blk_i & 1) ? (zp >> 4) : (zp & 0x0F);
                    zero_points[nn] = static_cast<uint16_t>(zp);
                });
                uint16x8_t zp_u16_vec = vld1_u16(zero_points);
                neg_scaled_zp_vec = vcvtq_f16_u16(zp_u16_vec);
            } else {
                neg_scaled_zp_vec = zp_mid_point_vec;
            }
            neg_scaled_zp_vec = vnegq_f16(vmulq_f16(scale_vec, neg_scaled_zp_vec));

            for (int kk = 0; kk < BlkLen; kk += kk_blk_dim) {
                Q4BitBlkDequantB_16K_8N(src_ptr, scale_vec, neg_scaled_zp_vec, dst_ptr);

                src_ptr += kk_n_src_bytes;
                dst_ptr += kk_n_dst_size;
            }

            ++scales_ptr;
            if (QuantBZeroPoint) {
                zero_points_ptr += k_blk_i & 1;
            }
        }

        QuantBData += ld_blk_src;
        FpData += ld_blk_dst;
        QuantBScale += ld_blk_scale;
        QuantBZeroPoint = QuantBZeroPoint ? QuantBZeroPoint + ld_blk_zp : nullptr;
    }

    // remaining N
    for (; n < CountN; ++n) {
        for (int k_blk_i = 0; k_blk_i < BlockCountK; ++k_blk_i) {
            MLAS_FP16 scale = QuantBScale[0];
            float16x8_t scale_vec = vdupq_n_f16(scale);
            float16x8_t neg_scaled_zp_vec;

            if (QuantBZeroPoint) {
                uint8_t zero_point = QuantBZeroPoint[0];
                zero_point = (k_blk_i & 1) ? (zero_point >> 4) : (zero_point & 0x0F);
                uint16x8_t zp_u16_vec = vdupq_n_u16(static_cast<uint16_t>(zero_point));
                neg_scaled_zp_vec = vcvtq_f16_u16(zp_u16_vec);
            } else {
                neg_scaled_zp_vec = zp_mid_point_vec;
            }
            neg_scaled_zp_vec = vnegq_f16(vmulq_f16(scale_vec, neg_scaled_zp_vec));

            for (int kk = 0; kk < BlkLen; kk += kk_blk_dim) {
                Q4BitBlkDequantB_16K_1N(QuantBData, scale_vec, neg_scaled_zp_vec, FpData);

                QuantBData += kk_blk_bytes;
                FpData += kk_blk_dim;
            }

            ++QuantBScale;
            if (QuantBZeroPoint) {
                QuantBZeroPoint += k_blk_i & 1;
            }
        }
    }
}
}  // namespace sqnbitgemm_neon
