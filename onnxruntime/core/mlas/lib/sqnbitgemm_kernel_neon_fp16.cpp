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
#include <cstring>

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

MLAS_FORCEINLINE void
Transpose4x8(float16x8_t& v0, float16x8_t& v1, float16x8_t& v2, float16x8_t& v3)
{
    // |v00|v01|v02|v03|v04|v05|v06|v07|
    // |v10|v11|v12|v13|v14|v15|v16|v17|
    // |v20|v21|v22|v23|v24|v25|v26|v27|
    // |v30|v31|v32|v33|v34|v35|v36|v37|
    //  =>
    // |v00|v10|v20|v30|v04|v14|v24|v34|
    // |v01|v11|v21|v31|v05|v15|v25|v35|
    // |v02|v12|v22|v32|v06|v16|v26|v36|
    // |v03|v13|v23|v33|v07|v17|v27|v37|
    float16x8x2_t t01 = vtrnq_f16(v0, v1);
    float16x8x2_t t23 = vtrnq_f16(v2, v3);

    v0 = vreinterpretq_f16_f32(vtrn1q_f32(vreinterpretq_f32_f16(t01.val[0]), vreinterpretq_f32_f16(t23.val[0])));
    v1 = vreinterpretq_f16_f32(vtrn1q_f32(vreinterpretq_f32_f16(t01.val[1]), vreinterpretq_f32_f16(t23.val[1])));
    v2 = vreinterpretq_f16_f32(vtrn2q_f32(vreinterpretq_f32_f16(t01.val[0]), vreinterpretq_f32_f16(t23.val[0])));
    v3 = vreinterpretq_f16_f32(vtrn2q_f32(vreinterpretq_f32_f16(t01.val[1]), vreinterpretq_f32_f16(t23.val[1])));
}

MLAS_FORCEINLINE void
Transpose4x4(float16x4_t& v0, float16x4_t& v1, float16x4_t& v2, float16x4_t& v3)
{
    float16x4x2_t t01 = vtrn_f16(v0, v1);
    float16x4x2_t t23 = vtrn_f16(v2, v3);

    v0 = vreinterpret_f16_f32(vtrn1_f32(vreinterpret_f32_f16(t01.val[0]), vreinterpret_f32_f16(t23.val[0])));
    v1 = vreinterpret_f16_f32(vtrn1_f32(vreinterpret_f32_f16(t01.val[1]), vreinterpret_f32_f16(t23.val[1])));
    v2 = vreinterpret_f16_f32(vtrn2_f32(vreinterpret_f32_f16(t01.val[0]), vreinterpret_f32_f16(t23.val[0])));
    v3 = vreinterpret_f16_f32(vtrn2_f32(vreinterpret_f32_f16(t01.val[1]), vreinterpret_f32_f16(t23.val[1])));
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
    constexpr size_t k_blk_bytes = MlasQNBitBlkDataSizeInBytes(nbits, k_blk_dim);
    const size_t iterations = k_blk_num * n_blk_num; // one iteration per block
    const size_t ld = MlasDivRoundup(K, BlkLen) * MlasQNBitBlkDataSizeInBytes(nbits, BlkLen);

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
            size_t n = n_blk * n_blk_dim;
            const size_t src_offset = n * ld + k_blk * k_blk_bytes;

            if (n + n_blk_dim <= N) {
                const size_t dst_offset = n * ld + k_blk * k_blk_bytes * n_blk_dim;
                const uint8_t* src = reinterpret_cast<const uint8_t*>(QuantBDataBegin) + src_offset;
                uint8_t* dst = reinterpret_cast<uint8_t*>(PackedQuantBDataBegin) + dst_offset;

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
                const uint8_t* src = reinterpret_cast<const uint8_t*>(QuantBDataBegin) + src_offset;
                uint8_t* dst = reinterpret_cast<uint8_t*>(PackedQuantBDataBegin) + src_offset;

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

template<size_t N, size_t K>
MLAS_FORCEINLINE
typename std::enable_if_t<(N == 8 && K == 16), void>
Q4BitBlkDequantBKernel(
    const std::uint8_t* src_ptr,
    const float16x8_t& scale,
    const float16x8_t& neg_scaled_zp,
    MLAS_FP16* dst_ptr
) {
    const uint8x8_t low_mask = vdup_n_u8(0x0F);

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

    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr), c0);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 8), c1);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 16), c2);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 24), c3);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 32), c4);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 40), c5);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 48), c6);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 56), c7);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 64), c8);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 72), c9);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 80), ca);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 88), cb);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 96), cc);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 104), cd);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 112), ce);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 120), cf);
}

template<size_t N, size_t K>
MLAS_FORCEINLINE
typename std::enable_if_t<(N == 1 && K == 16), void>
Q4BitBlkDequantBKernel(
    const std::uint8_t* src_ptr,
    const float16x8_t& scale,
    const float16x8_t& neg_scaled_zp,
    MLAS_FP16* dst_ptr
) {
    const uint8x8_t low_mask = vdup_n_u8(0x0F);

    uint8x8_t v0 = vld1_u8(src_ptr);

    float16x8_t f_low = vcvtq_f16_u16(vshll_n_u8(vand_u8(v0, low_mask), 0));
    float16x8_t f_high = vcvtq_f16_u16(vshll_n_u8(vshr_n_u8(v0, 4), 0));

    float16x8_t c0 = vfmaq_f16(neg_scaled_zp, f_low, scale);
    float16x8_t c1 = vfmaq_f16(neg_scaled_zp, f_high, scale);

    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr), c0);
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(dst_ptr + 8), c1);
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
    MLAS_UNREFERENCED_PARAMETER(K);
    constexpr size_t nbits = 4;
    constexpr size_t kk_blk_dim = 16;
    constexpr size_t n_blk_dim = 8;
    assert(BlkLen > 0 && BlkLen % kk_blk_dim == 0);

    const size_t kk_blk_num = BlockCountK * BlkLen / kk_blk_dim;
    constexpr size_t kk_blk_bytes = MlasQNBitBlkDataSizeInBytes(nbits, kk_blk_dim);
    const size_t kk_n_src_bytes = kk_blk_bytes * n_blk_dim;
    const size_t kk_n_dst_size = kk_blk_dim * n_blk_dim;
    const size_t ld_blk_src = kk_blk_num * kk_n_src_bytes;
    const size_t ld_blk_dst = BlkLen * BlockCountK * n_blk_dim;
    const size_t ld_blk_scale = BlockCountK * n_blk_dim;
    const size_t ld_zp = (BlockCountK + 1) / 2;
    const size_t ld_blk_zp = ld_zp * n_blk_dim;
    const float16x8_t zp_mid_point_vec = MlasBroadcastFloat16x8(MLAS_FP16(8.0f).val);
    const bool has_zp = QuantBZeroPoint != nullptr;

    size_t n = 0;
    for (; n + n_blk_dim <= CountN; n += n_blk_dim) {
        const MLAS_FP16* scales_ptr = QuantBScale;
        const std::uint8_t* zero_points_ptr = reinterpret_cast<const uint8_t*>(QuantBZeroPoint);
        const std::uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(QuantBData);
        MLAS_FP16* dst_ptr = FpData;

        for (size_t k_blk_i = 0; k_blk_i < BlockCountK; ++k_blk_i) {
            // prepare scales and zero_points for the block
            MLAS_FP16 scales[n_blk_dim];
            uint16_t zero_points[n_blk_dim];
            float16x8_t scale_vec;
            float16x8_t neg_scaled_zp_vec;

            UnrolledLoop<n_blk_dim>([&](int nn){
                scales[nn] = scales_ptr[nn * BlockCountK];
            });
            scale_vec = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(scales));

            if (has_zp) {
                UnrolledLoop<n_blk_dim>([&](int nn){
                    uint8_t zp = zero_points_ptr[nn * ld_zp];
                    zp = (k_blk_i & 1) ? (zp >> 4) : (zp & 0x0F);
                    zero_points[nn] = static_cast<uint16_t>(zp);
                });
                uint16x8_t zp_u16_vec = vld1q_u16(zero_points);
                neg_scaled_zp_vec = vcvtq_f16_u16(zp_u16_vec);
            } else {
                neg_scaled_zp_vec = zp_mid_point_vec;
            }
            neg_scaled_zp_vec = vnegq_f16(vmulq_f16(scale_vec, neg_scaled_zp_vec));

            for (size_t kk = 0; kk < BlkLen; kk += kk_blk_dim) {
                Q4BitBlkDequantBKernel<8, 16>(src_ptr, scale_vec, neg_scaled_zp_vec, dst_ptr);

                src_ptr += kk_n_src_bytes;
                dst_ptr += kk_n_dst_size;
            }

            ++scales_ptr;
            if (has_zp) {
                zero_points_ptr += k_blk_i & 1;
            }
        }

        QuantBData += ld_blk_src;
        FpData += ld_blk_dst;
        QuantBScale += ld_blk_scale;
        QuantBZeroPoint = has_zp ? QuantBZeroPoint + ld_blk_zp : nullptr;
    }

    // remaining N
    for (; n < CountN; ++n) {
        const MLAS_FP16* scales_ptr = QuantBScale;
        const std::uint8_t* zero_points_ptr = reinterpret_cast<const uint8_t*>(QuantBZeroPoint);
        for (size_t k_blk_i = 0; k_blk_i < BlockCountK; ++k_blk_i) {
            MLAS_FP16 scale = scales_ptr[0];
            float16x8_t scale_vec = MlasBroadcastFloat16x8(scale.val);
            float16x8_t neg_scaled_zp_vec;

            if (has_zp) {
                uint8_t zero_point = static_cast<uint8_t>(zero_points_ptr[0]);
                zero_point = (k_blk_i & 1) ? (zero_point >> 4) : (zero_point & 0x0F);
                uint16x8_t zp_u16_vec = vdupq_n_u16(static_cast<uint16_t>(zero_point));
                neg_scaled_zp_vec = vcvtq_f16_u16(zp_u16_vec);
            } else {
                neg_scaled_zp_vec = zp_mid_point_vec;
            }
            neg_scaled_zp_vec = vnegq_f16(vmulq_f16(scale_vec, neg_scaled_zp_vec));

            for (size_t kk = 0; kk < BlkLen; kk += kk_blk_dim) {
                Q4BitBlkDequantBKernel<1, 16>(
                    reinterpret_cast<const uint8_t*>(QuantBData), scale_vec, neg_scaled_zp_vec, FpData
                );

                QuantBData += kk_blk_bytes;
                FpData += kk_blk_dim;
            }

            ++scales_ptr;
            if (has_zp) {
                zero_points_ptr += k_blk_i & 1;
            }
        }

        QuantBScale += BlockCountK;
        if (has_zp) {
            QuantBZeroPoint += ld_zp;
        }
    }
}

template <bool Initialize, bool UseBias, size_t N, size_t M>
MLAS_FORCEINLINE
typename std::enable_if_t<(N == 8 && M == 1), float16x8_t>
PrepareAccumulator(
    const MLAS_FP16* Bias,
    const MLAS_FP16* C
) {
    if constexpr (Initialize) {
        if constexpr (UseBias) {
            return MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(Bias));
        } else {
            return MlasZeroFloat16x8();
        }
    } else {
        return MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(C));
    }
}

template <bool Initialize, bool UseBias, size_t N, size_t M>
MLAS_FORCEINLINE
typename std::enable_if_t<(N == 4 && M == 1), float16x4_t>
PrepareAccumulator(
    const MLAS_FP16* Bias,
    const MLAS_FP16* C
) {
    if constexpr (Initialize) {
        if constexpr (UseBias) {
            return MlasLoadFloat16x4(reinterpret_cast<const _mlas_fp16_*>(Bias));
        } else {
            return MlasZeroFloat16x4();
        }
    } else {
        return MlasLoadFloat16x4(reinterpret_cast<const _mlas_fp16_*>(C));
    }
}

template <bool Initialize, bool UseBias, size_t N, size_t M>
MLAS_FORCEINLINE
typename std::enable_if_t<((N == 2 || N == 1) && M == 1), float16x4_t>
PrepareAccumulator(
    const MLAS_FP16* Bias,
    const MLAS_FP16* C
) {
    float16x4_t v = MlasZeroFloat16x4();

    if constexpr (Initialize) {
        if constexpr (UseBias) {
            v = MlasLoadLaneFloat16x4<0>(Bias, v);
            if constexpr (N == 2) {
                v = MlasLoadLaneFloat16x4<1>(Bias + 1, v);
            }
            return v;
        } else {
            return v;
        }
    } else {
        v = MlasLoadLaneFloat16x4<0>(C, v);
        if constexpr (N == 2) {
            v = MlasLoadLaneFloat16x4<1>(C + 1, v);
        }
        return v;
    }
}

template<size_t N, size_t M, size_t K>
MLAS_FORCEINLINE
typename std::enable_if_t<(N == 8 && M == 1 && K == 8), float16x8_t>
SQ4BitGemmMicroKernel(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    float16x8_t accumulator
) {
    float16x8_t a0 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(A));
    float16x8_t b0 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B));
    float16x8_t b1 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 8));
    float16x8_t b2 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 16));
    float16x8_t b3 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 24));
    float16x8_t b4 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 32));
    float16x8_t b5 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 40));
    float16x8_t b6 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 48));
    float16x8_t b7 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 56));

    float16x8_t c0 = vfmaq_laneq_f16(accumulator, b0, a0, 0);
    float16x8_t c01 = vfmaq_laneq_f16(c0, b1, a0, 1);
    float16x8_t c2 = vmulq_laneq_f16(b2, a0, 2);
    float16x8_t c23 = vfmaq_laneq_f16(c2, b3, a0, 3);
    float16x8_t c4 = vmulq_laneq_f16(b4, a0, 4);
    float16x8_t c45 = vfmaq_laneq_f16(c4, b5, a0, 5);
    float16x8_t c6 = vmulq_laneq_f16(b6, a0, 6);
    float16x8_t c67 = vfmaq_laneq_f16(c6, b7, a0, 7);

    float16x8_t c0123 = vaddq_f16(c01, c23);
    float16x8_t c4567 = vaddq_f16(c45, c67);

    return vaddq_f16(c0123, c4567);
}

template<size_t N, size_t M, size_t K>
MLAS_FORCEINLINE
typename std::enable_if_t<(N == 8 && M == 1 && K == 4), float16x8_t>
SQ4BitGemmMicroKernel(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    float16x8_t accumulator
) {
    float16x4_t a0 = MlasLoadFloat16x4(reinterpret_cast<const _mlas_fp16_*>(A));
    float16x8_t b0 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B));
    float16x8_t b1 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 8));
    float16x8_t b2 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 16));
    float16x8_t b3 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 24));

    float16x8_t c0 = vfmaq_lane_f16(accumulator, b0, a0, 0);
    float16x8_t c01 = vfmaq_lane_f16(c0, b1, a0, 1);
    float16x8_t c2 = vmulq_lane_f16(b2, a0, 2);
    float16x8_t c23 = vfmaq_lane_f16(c2, b3, a0, 3);

    return vaddq_f16(c01, c23);
}

template<size_t N, size_t M, size_t K>
MLAS_FORCEINLINE
typename std::enable_if_t<(N == 8 && M == 1 && (K == 2 || K == 1)), float16x8_t>
SQ4BitGemmMicroKernel(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    float16x8_t accumulator
) {
    float16x4_t a0 = MlasZeroFloat16x4();
    a0 = MlasLoadLaneFloat16x4<0>(A, a0);
    if constexpr (K == 2) a0 = MlasLoadLaneFloat16x4<1>(A + 1, a0);
    float16x8_t b0 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B)), b1;
    if constexpr (K == 2) b1 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + 8));

    float16x8_t c0 = vfmaq_lane_f16(accumulator, b0, a0, 0), c01;
    if constexpr (K == 2) c01 = vfmaq_lane_f16(c0, b1, a0, 1);

    if constexpr (K == 1)
        return c0;
    else
        return c01;
}

template <size_t N, size_t M, size_t K>
MLAS_FORCEINLINE
    typename std::enable_if_t<((N > 0 && N <= 4) && M == 1 && K == 8), float16x4_t>
    SQ4BitGemmMicroKernel(
        const MLAS_FP16* A,
        const MLAS_FP16* B,
        const size_t ldb,
        float16x4_t accumulator
    )
{
    float16x8_t a0 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(A));

    float16x8_t b0, b1, b2, b3;
    b0 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B));
    if constexpr (N > 1) b1 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + ldb));
    if constexpr (N > 2) b2 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + ldb * 2));
    if constexpr (N > 3) b3 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(B + ldb * 3));

    float16x8_t c00, c01, c02, c03;
    c00 = vmulq_f16(b0, a0);
    if constexpr (N > 1)
        c01 = vmulq_f16(b1, a0);
    else
        c01 = MlasZeroFloat16x8();
    if constexpr (N > 2)
        c02 = vmulq_f16(b2, a0);
    else
        c02 = MlasZeroFloat16x8();
    if constexpr (N > 3)
        c03 = vmulq_f16(b3, a0);
    else
        c03 = MlasZeroFloat16x8();

    Transpose4x8(c00, c01, c02, c03);

    float16x8_t c_low_high = vaddq_f16(vaddq_f16(c00, c01), vaddq_f16(c02, c03));
    float16x4_t c_low = vget_low_f16(c_low_high);
    float16x4_t c_high = vget_high_f16(c_low_high);
    float16x4_t c = vadd_f16(c_low, c_high);

    return vadd_f16(c, accumulator);
}

template <size_t N, size_t M, size_t K>
MLAS_FORCEINLINE
    typename std::enable_if_t<((N > 0 && N <= 4) && M == 1 && (K > 0 && K <= 4)), float16x4_t>
    SQ4BitGemmMicroKernel(
        const MLAS_FP16* A,
        const MLAS_FP16* B,
        const size_t ldb,
        float16x4_t accumulator
    )
{
    float16x4_t a0;
    if constexpr (K == 4)
        a0 = MlasLoadFloat16x4(reinterpret_cast<const _mlas_fp16_*>(A));
    else {
        a0 = MlasZeroFloat16x4();
        a0 = MlasLoadLaneFloat16x4<0>(A, a0);
        if constexpr (K >= 2) a0 = MlasLoadLaneFloat16x4<1>(A + 1, a0);
        if constexpr (K >= 3) a0 = MlasLoadLaneFloat16x4<2>(A + 2, a0);
    }

    float16x4_t b0, b1, b2, b3;
    b0 = MlasLoadFloat16x4(reinterpret_cast<const _mlas_fp16_*>(B));
    if constexpr (N > 1) b1 = MlasLoadFloat16x4(reinterpret_cast<const _mlas_fp16_*>(B + ldb));
    if constexpr (N > 2) b2 = MlasLoadFloat16x4(reinterpret_cast<const _mlas_fp16_*>(B + ldb * 2));
    if constexpr (N > 3) b3 = MlasLoadFloat16x4(reinterpret_cast<const _mlas_fp16_*>(B + ldb * 3));

    float16x4_t c00, c01, c02, c03;
    c00 = vmul_f16(b0, a0);
    if constexpr (N > 1)
        c01 = vmul_f16(b1, a0);
    else
        c01 = MlasZeroFloat16x4();
    if constexpr (N > 2)
        c02 = vmul_f16(b2, a0);
    else
        c02 = MlasZeroFloat16x4();
    if constexpr (N > 3)
        c03 = vmul_f16(b3, a0);
    else
        c03 = MlasZeroFloat16x4();

    Transpose4x4(c00, c01, c02, c03);

    float16x4_t c = vadd_f16(vadd_f16(c00, c01), vadd_f16(c02, c03));
    return vadd_f16(c, accumulator);
}

template <size_t StrideN, bool Initialize, bool UseBias, size_t N, size_t M, size_t K>
MLAS_FORCEINLINE
    typename std::enable_if_t<((N == 16 || N == 8) && (M == 2 || M == 1) && (K == 8)), void>
    SQ4BitGemmRegisterKernel(
        const MLAS_FP16* a,
        const MLAS_FP16* b,
        const MLAS_FP16* bias,
        MLAS_FP16* buffer,
        size_t CountK,
        size_t lda,
        size_t ldb
    )
{
    const size_t ldb_substep = ldb * 8;
    float16x8_t accu00, accu01, accu10, accu11;
    // if k == 0 load bias or zero, else load buffer, init register accumulator
    auto* bias_1 = UseBias ? bias + 8 : nullptr;
    accu00 = PrepareAccumulator<Initialize, UseBias, 8, 1>(bias, buffer);
    if constexpr (N == 16) {
        accu01 = PrepareAccumulator<Initialize, UseBias, 8, 1>(bias_1, buffer + 8);
    }
    if constexpr (M == 2) {
        accu10 = PrepareAccumulator<Initialize, UseBias, 8, 1>(bias, buffer + StrideN);
    }
    if constexpr (M == 2 && N == 16) {
        accu11 = PrepareAccumulator<Initialize, UseBias, 8, 1>(bias_1, buffer + StrideN + 8);
    }

    const MLAS_FP16* aa = a;
    const MLAS_FP16* bb = b;
    size_t kk = 0;
    for (; kk + 8 <= CountK; kk += 8, aa += 8, bb += 8) {  // 16N_2M_8K
        accu00 = SQ4BitGemmMicroKernel<8, 1, 8>(aa, bb, accu00);
        if constexpr (N == 16) {
            accu01 = SQ4BitGemmMicroKernel<8, 1, 8>(aa, bb + ldb_substep, accu01);
        }
        if constexpr (M == 2) {
            accu10 = SQ4BitGemmMicroKernel<8, 1, 8>(aa + lda, bb, accu10);
        }
        if constexpr (M == 2 && N == 16) {
            accu11 = SQ4BitGemmMicroKernel<8, 1, 8>(aa + lda, bb + ldb_substep, accu11);
        }
    }

    // remaining K
    if (kk + 4 <= CountK) {
        accu00 = SQ4BitGemmMicroKernel<8, 1, 4>(aa, bb, accu00);
        if constexpr (N == 16) {
            accu01 = SQ4BitGemmMicroKernel<8, 1, 4>(aa, bb + ldb_substep, accu01);
        }
        if constexpr (M == 2) {
            accu10 = SQ4BitGemmMicroKernel<8, 1, 4>(aa + lda, bb, accu10);
        }
        if constexpr (M == 2 && N == 16) {
            accu11 = SQ4BitGemmMicroKernel<8, 1, 4>(aa + lda, bb + ldb_substep, accu11);
        }
        kk += 4, aa += 4, bb += 4;
    }

    if (kk + 2 <= CountK) {
        accu00 = SQ4BitGemmMicroKernel<8, 1, 2>(aa, bb, accu00);
        if constexpr (N == 16) {
            accu01 = SQ4BitGemmMicroKernel<8, 1, 2>(aa, bb + ldb_substep, accu01);
        }
        if constexpr (M == 2) {
            accu10 = SQ4BitGemmMicroKernel<8, 1, 2>(aa + lda, bb, accu10);
        }
        if constexpr (M == 2 && N == 16) {
            accu11 = SQ4BitGemmMicroKernel<8, 1, 2>(aa + lda, bb + ldb_substep, accu11);
        }
        kk += 2, aa += 2, bb += 2;
    }

    if (kk < CountK) {
        accu00 = SQ4BitGemmMicroKernel<8, 1, 1>(aa, bb, accu00);
        if constexpr (N == 16) {
            accu01 = SQ4BitGemmMicroKernel<8, 1, 1>(aa, bb + ldb_substep, accu01);
        }
        if constexpr (M == 2) {
            accu10 = SQ4BitGemmMicroKernel<8, 1, 1>(aa + lda, bb, accu10);
        }
        if constexpr (M == 2 && N == 16) {
            accu11 = SQ4BitGemmMicroKernel<8, 1, 1>(aa + lda, bb + ldb_substep, accu11);
        }
    }

    // save register to buffer
    MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(buffer), accu00);
    if constexpr (N == 16) {
        MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(buffer + 8), accu01);
    }
    if constexpr (M == 2) {
        MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(buffer + StrideN), accu10);
    }
    if constexpr (M == 2 && N == 16) {
        MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(buffer + StrideN + 8), accu11);
    }
}

template <size_t StrideN, bool Initialize, bool UseBias, size_t N, size_t M, size_t K>
MLAS_FORCEINLINE
    typename std::enable_if_t<((N > 0 && N <= 4) && (M == 2 || M == 1) && (K == 8)), void>
    SQ4BitGemmRegisterKernel(
        const MLAS_FP16* a,
        const MLAS_FP16* b,
        const MLAS_FP16* bias,
        MLAS_FP16* buffer,
        size_t CountK,
        size_t lda,
        size_t ldb
    )
{
    float16x4_t accu0, accu1;
    // if k == 0 load bias or zero, else load buffer, init register accumulator
    accu0 = PrepareAccumulator<Initialize, UseBias, N, 1>(bias, buffer);
    if constexpr (M == 2) {
        accu1 = PrepareAccumulator<Initialize, UseBias, N, 1>(bias, buffer + StrideN);
    }

    const MLAS_FP16* aa = a;
    const MLAS_FP16* bb = b;
    size_t kk = 0;
    for (; kk + 8 <= CountK; kk += 8, aa += 8, bb += 8) {  // 4N_2M_8K
        accu0 = SQ4BitGemmMicroKernel<N, 1, 8>(aa, bb, ldb, accu0);
        if constexpr (M == 2) {
            accu1 = SQ4BitGemmMicroKernel<N, 1, 8>(aa + lda, bb, ldb, accu1);
        }
    }

    // remaining K
    if (kk + 4 <= CountK) {
        accu0 = SQ4BitGemmMicroKernel<N, 1, 4>(aa, bb, ldb, accu0);
        if constexpr (M == 2) {
            accu1 = SQ4BitGemmMicroKernel<N, 1, 4>(aa + lda, bb, ldb, accu1);
        }
        kk += 4, aa += 4, bb += 4;
    }

    if (kk + 2 <= CountK) {
        accu0 = SQ4BitGemmMicroKernel<N, 1, 2>(aa, bb, ldb, accu0);
        if constexpr (M == 2) {
            accu1 = SQ4BitGemmMicroKernel<N, 1, 2>(aa + lda, bb, ldb, accu1);
        }
        kk += 2, aa += 2, bb += 2;
    }

    if (kk < CountK) {
        accu0 = SQ4BitGemmMicroKernel<N, 1, 1>(aa, bb, ldb, accu0);
        if constexpr (M == 2) {
            accu1 = SQ4BitGemmMicroKernel<N, 1, 1>(aa + lda, bb, ldb, accu1);
        }
    }

    // save register to buffer
    if constexpr (M == 2) {
        if constexpr (N == 4) {
            MlasStoreFloat16x4(reinterpret_cast<_mlas_fp16_*>(buffer), accu0);
            MlasStoreFloat16x4(reinterpret_cast<_mlas_fp16_*>(buffer + StrideN), accu1);
        } else {
            MlasStoreLaneFloat16x4<0>(buffer, accu0);
            MlasStoreLaneFloat16x4<0>(buffer + StrideN, accu1);
            if constexpr (N > 1) {
                MlasStoreLaneFloat16x4<1>(buffer + 1, accu0);
                MlasStoreLaneFloat16x4<1>(buffer + StrideN + 1, accu1);
            }
            if constexpr (N > 2) {
                MlasStoreLaneFloat16x4<2>(buffer + 2, accu0);
                MlasStoreLaneFloat16x4<2>(buffer + StrideN + 2, accu1);
            }
        }
    } else {
        if constexpr (N == 4)
            MlasStoreFloat16x4(reinterpret_cast<_mlas_fp16_*>(buffer), accu0);
        else {
            MlasStoreLaneFloat16x4<0>(buffer, accu0);
            if constexpr (N > 1) MlasStoreLaneFloat16x4<1>(buffer + 1, accu0);
            if constexpr (N > 2) MlasStoreLaneFloat16x4<2>(buffer + 2, accu0);
        }
    }
}

template <size_t StrideM, size_t StrideN>
void
SQ4BitGemmKernel_CompFp16(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    const MLAS_FP16* Bias,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const size_t stride_M,
    const size_t stride_N
)
{
    assert(stride_M == StrideM);
    assert(stride_N == StrideN);
    assert(StrideM >= CountM);
    assert(StrideN >= CountN);
    MLAS_UNREFERENCED_PARAMETER(stride_M);
    MLAS_UNREFERENCED_PARAMETER(stride_N);

    constexpr size_t StrideK = 128;
    constexpr size_t m_step = 2;
    constexpr size_t n_step = 16;
    constexpr size_t k_step = 8;
    const size_t lda_step = lda * m_step;
    const size_t ldb_step = ldb * n_step;
    const size_t ldc_step = StrideN * m_step;

    MLAS_FP16 buffer[StrideM * StrideN];

    // First cache block K, directly write to buffer. Add bias if exists.
    size_t k = 0, CountK;
    if (k < K) {
        CountK = std::min(K, StrideK);
        if (Bias) {
            const MLAS_FP16 *b = B, *bias = Bias;
            size_t nn = 0, c = 0;
            for (; nn + n_step <= CountN; nn += n_step) {
                const MLAS_FP16* a = A;
                size_t mm = 0, cc = c;
                for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                    SQ4BitGemmRegisterKernel<StrideN, true, true, n_step, m_step, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                if (mm < CountM) {
                    SQ4BitGemmRegisterKernel<StrideN, true, true, n_step, 1, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                b += ldb_step;
                bias = bias ? bias + n_step : nullptr;
                c += n_step;
            }

            if (nn + 8 <= CountN) {
                const MLAS_FP16* a = A;
                size_t mm = 0, cc = c;
                for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                    SQ4BitGemmRegisterKernel<StrideN, true, true, 8, m_step, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                if (mm < CountM) {
                    SQ4BitGemmRegisterKernel<StrideN, true, true, 8, 1, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                nn += 8;
                b += ldb * 8;
                bias = bias ? bias + 8 : nullptr;
                c += 8;
            }

            if (nn + 4 <= CountN) {
                const MLAS_FP16* a = A;
                size_t mm = 0, cc = c;
                for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                    SQ4BitGemmRegisterKernel<StrideN, true, true, 4, m_step, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                if (mm < CountM) {
                    SQ4BitGemmRegisterKernel<StrideN, true, true, 4, 1, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                nn += 4;
                b += ldb * 4;
                bias = bias ? bias + 4 : nullptr;
                c += 4;
            }

            if (nn + 2 <= CountN) {
                const MLAS_FP16* a = A;
                size_t mm = 0, cc = c;
                for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                    SQ4BitGemmRegisterKernel<StrideN, true, true, 2, m_step, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                // remaining 1M
                if (mm < CountM) {
                    SQ4BitGemmRegisterKernel<StrideN, true, true, 2, 1, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                nn += 2;
                b += ldb * 2;
                bias = bias ? bias + 2 : nullptr;
                c += 2;
            }

            if (nn < CountN) {
                const MLAS_FP16* a = A;
                size_t mm = 0, cc = c;
                for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                    SQ4BitGemmRegisterKernel<StrideN, true, true, 1, m_step, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                if (mm < CountM) {
                    SQ4BitGemmRegisterKernel<StrideN, true, true, 1, 1, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }
            }
        } else {
            const MLAS_FP16 *b = B, *bias = nullptr;
            size_t nn = 0, c = 0;
            for (; nn + n_step <= CountN; nn += n_step, b += ldb_step, c += n_step) {
                const MLAS_FP16* a = A;
                size_t mm = 0, cc = c;
                for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                    SQ4BitGemmRegisterKernel<StrideN, true, false, n_step, m_step, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                if (mm < CountM) {
                    SQ4BitGemmRegisterKernel<StrideN, true, false, n_step, 1, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }
            }

            // remaining N
            if (nn + 8 <= CountN) {
                const MLAS_FP16* a = A;
                size_t mm = 0, cc = c;
                for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                    SQ4BitGemmRegisterKernel<StrideN, true, false, 8, m_step, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                if (mm < CountM) {
                    SQ4BitGemmRegisterKernel<StrideN, true, false, 8, 1, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                nn += 8;
                b += ldb * 8;
                c += 8;
            }

            if (nn + 4 <= CountN) {
                const MLAS_FP16* a = A;
                size_t mm = 0, cc = c;
                for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                    SQ4BitGemmRegisterKernel<StrideN, true, false, 4, m_step, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                // remaining 1M
                if (mm < CountM) {
                    SQ4BitGemmRegisterKernel<StrideN, true, false, 4, 1, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                nn += 4;
                b += ldb * 4;
                c += 4;
            }

            if (nn + 2 <= CountN) {
                const MLAS_FP16* a = A;
                size_t mm = 0, cc = c;
                for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                    SQ4BitGemmRegisterKernel<StrideN, true, false, 2, m_step, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                // remaining 1M
                if (mm < CountM) {
                    SQ4BitGemmRegisterKernel<StrideN, true, false, 2, 1, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                nn += 2;
                b += ldb * 2;
                c += 2;
            }

            if (nn < CountN) {
                const MLAS_FP16* a = A;
                size_t mm = 0, cc = c;
                for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                    SQ4BitGemmRegisterKernel<StrideN, true, false, 1, m_step, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }

                // remaining 1M
                if (mm < CountM) {
                    SQ4BitGemmRegisterKernel<StrideN, true, false, 1, 1, k_step>(
                        a, b, bias, buffer + cc, CountK, lda, ldb
                    );
                }
            }
        }

        k += CountK;
        A += CountK;
        B += CountK;
    }

    // 2nd+ cache block K, accumulate to buffer.
    for (; k < K; k += CountK, A += CountK, B += CountK) {
        CountK = std::min(K - k, StrideK);
        const MLAS_FP16* b = B;
        size_t nn = 0, c = 0;
        for (; nn + n_step <= CountN; nn += n_step) {
            const MLAS_FP16* a = A;
            size_t mm = 0, cc = c;
            for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                // load buffer, init register accumulator
                SQ4BitGemmRegisterKernel<StrideN, false, false, n_step, m_step, k_step>(
                    a, B, nullptr, buffer + cc, CountK, lda, ldb
                );
            }

            // remaining 1M
            if (mm < CountM) {
                SQ4BitGemmRegisterKernel<StrideN, false, false, n_step, 1, k_step>(
                    a, B, nullptr, buffer + cc, CountK, lda, ldb
                );
            }

            b += ldb_step;
            c += n_step;
        }

        // remaining N
        if (nn + 8 <= CountN) {
            const MLAS_FP16* a = A;
            size_t mm = 0, cc = c;
            for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                // load buffer, init register accumulator
                SQ4BitGemmRegisterKernel<StrideN, false, false, 8, 2, k_step>(
                    a, B, nullptr, buffer + cc, CountK, lda, ldb
                );
            }

            // remaining 1M
            if (mm < CountM) {
                SQ4BitGemmRegisterKernel<StrideN, false, false, 8, 1, k_step>(
                    a, B, nullptr, buffer + cc, CountK, lda, ldb
                );
            }

            nn += 8;
            b += ldb * 8;
            c += 8;
        }

        if (nn + 4 <= CountN) {
            const MLAS_FP16* a = A;
            size_t mm = 0, cc = c;
            for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                // load buffer, init register accumulator
                SQ4BitGemmRegisterKernel<StrideN, false, false, 4, 2, k_step>(
                    a, B, nullptr, buffer + cc, CountK, lda, ldb
                );
            }

            // remaining 1M
            if (mm < CountM) {
                SQ4BitGemmRegisterKernel<StrideN, false, false, 4, 1, k_step>(
                    a, B, nullptr, buffer + cc, CountK, lda, ldb
                );
            }

            nn += 4;
            b += ldb * 4;
            c += 4;
        }

        if (nn + 2 <= CountN) {
            const MLAS_FP16* a = A;
            size_t mm = 0, cc = c;
            for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                // load buffer, init register accumulator
                SQ4BitGemmRegisterKernel<StrideN, false, false, 2, 2, k_step>(
                    a, B, nullptr, buffer + cc, CountK, lda, ldb
                );
            }

            // remaining 1M
            if (mm < CountM) {
                SQ4BitGemmRegisterKernel<StrideN, false, false, 2, 1, k_step>(
                    a, B, nullptr, buffer + cc, CountK, lda, ldb
                );
            }

            nn += 2;
            b += ldb * 2;
            c += 2;
        }

        if (nn < CountN) {
            const MLAS_FP16* a = A;
            size_t mm = 0, cc = c;
            for (; mm + m_step <= CountM; mm += m_step, a += lda_step, cc += ldc_step) {
                // load buffer, init register accumulator
                SQ4BitGemmRegisterKernel<StrideN, false, false, 1, 2, k_step>(
                    a, B, nullptr, buffer + cc, CountK, lda, ldb
                );
            }

            // remaining 1M
            if (mm < CountM) {
                SQ4BitGemmRegisterKernel<StrideN, false, false, 1, 1, k_step>(
                    a, B, nullptr, buffer + cc, CountK, lda, ldb
                );
            }
        }
    }

    // save buffer to C
    if (CountN == StrideN) {
        size_t m = 0;
        MLAS_FP16* pbuffer = buffer;
        for (; m + 2 <= CountM; m += 2, C += ldc * 2, pbuffer += StrideN * 2) {
            float16x8_t c00 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C), c00);
            float16x8_t c01 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer + 8));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C + 8), c01);
            float16x8_t c02 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer + 16));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C + 16), c02);
            float16x8_t c03 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer + 24));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C + 24), c03);
            float16x8_t c10 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer + StrideN));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C + ldc), c10);
            float16x8_t c11 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer + StrideN + 8));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C + ldc + 8), c11);
            float16x8_t c12 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer + StrideN + 16));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C + ldc + 16), c12);
            float16x8_t c13 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer + StrideN + 24));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C + ldc + 24), c13);
        }

        if (m < CountM) {
            float16x8_t c00 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C), c00);
            float16x8_t c01 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer + 8));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C + 8), c01);
            float16x8_t c02 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer + 16));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C + 16), c02);
            float16x8_t c03 = MlasLoadFloat16x8(reinterpret_cast<const _mlas_fp16_*>(pbuffer + 24));
            MlasStoreFloat16x8(reinterpret_cast<_mlas_fp16_*>(C + 24), c03);
        }
    } else {
        size_t m = 0;
        MLAS_FP16* pbuffer = buffer;
        for (; m + 2 <= CountM; m += 2, C += ldc * 2, pbuffer += StrideN * 2) {
            std::memcpy(C, pbuffer, sizeof(MLAS_FP16) * CountN);
            std::memcpy(C + ldc, pbuffer + StrideN, sizeof(MLAS_FP16) * CountN);
        }

        if (m < CountM) {
            std::memcpy(C, pbuffer, sizeof(MLAS_FP16) * CountN);
        }
    }
}

template
void
SQ4BitGemmKernel_CompFp16<64, 32>(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    const MLAS_FP16* Bias,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const size_t stride_M,
    const size_t stride_N
);

}  // namespace sqnbitgemm_neon
