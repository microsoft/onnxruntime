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

            const size_t data_offset = n * ld + k_blk * k_blk_bytes;
            const uint8_t* src = reinterpret_cast<const uint8_t*>(QuantBDataBegin) + data_offset;
            uint8_t* dst = reinterpret_cast<uint8_t*>(PackedQuantBDataBegin) + data_offset;

            if (n + n_blk_dim <= N) {
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
                vst1_u8(dst + ld, v1);
                vst1_u8(dst + 2*ld, v2);
                vst1_u8(dst + 3*ld, v3);
                vst1_u8(dst + 4*ld, v4);
                vst1_u8(dst + 5*ld, v5);
                vst1_u8(dst + 6*ld, v6);
                vst1_u8(dst + 7*ld, v7);
            } else {
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
}  // namespace sqnbitgemm_neon
