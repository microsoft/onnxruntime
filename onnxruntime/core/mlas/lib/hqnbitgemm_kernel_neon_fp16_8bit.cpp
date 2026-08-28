/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    hqnbitgemm_kernel_neon_fp16_8bit.cpp

Abstract:

    This module implements the 8-bit quantized B data packing (8N interleaving)
    and dequantization to fp16 for ARM NEON, used with HQNBIT_CompFp16.

    The GEMM kernel itself (HQ4BitGemmKernel_CompFp16) is reused from
    hqnbitgemm_kernel_neon_fp16.cpp since it operates on dequantized fp16 data
    and is bit-width agnostic.

--*/

#include <arm_neon.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "fp16_common.h"
#include "qnbitgemm.h"
#include "qnbitgemm_kernel_neon.h"

namespace sqnbitgemm_neon
{

//
// 8x8 byte transpose using NEON. Same as in hqnbitgemm_kernel_neon_fp16.cpp.
//
MLAS_FORCEINLINE void
Transpose8x8_8bit(uint8x8_t& v0, uint8x8_t& v1, uint8x8_t& v2, uint8x8_t& v3,
                  uint8x8_t& v4, uint8x8_t& v5, uint8x8_t& v6, uint8x8_t& v7)
{
    uint8x8x2_t a0 = vtrn_u8(v0, v1);
    uint8x8x2_t a1 = vtrn_u8(v2, v3);
    uint8x8x2_t a2 = vtrn_u8(v4, v5);
    uint8x8x2_t a3 = vtrn_u8(v6, v7);

    uint16x4x2_t b0 = vtrn_u16(vreinterpret_u16_u8(a0.val[0]), vreinterpret_u16_u8(a1.val[0]));
    uint16x4x2_t b1 = vtrn_u16(vreinterpret_u16_u8(a0.val[1]), vreinterpret_u16_u8(a1.val[1]));
    uint16x4x2_t b2 = vtrn_u16(vreinterpret_u16_u8(a2.val[0]), vreinterpret_u16_u8(a3.val[0]));
    uint16x4x2_t b3 = vtrn_u16(vreinterpret_u16_u8(a2.val[1]), vreinterpret_u16_u8(a3.val[1]));

    uint32x2x2_t c0 = vtrn_u32(vreinterpret_u32_u16(b0.val[0]), vreinterpret_u32_u16(b2.val[0]));
    uint32x2x2_t c1 = vtrn_u32(vreinterpret_u32_u16(b0.val[1]), vreinterpret_u32_u16(b2.val[1]));
    uint32x2x2_t c2 = vtrn_u32(vreinterpret_u32_u16(b1.val[0]), vreinterpret_u32_u16(b3.val[0]));
    uint32x2x2_t c3 = vtrn_u32(vreinterpret_u32_u16(b1.val[1]), vreinterpret_u32_u16(b3.val[1]));

    v0 = vreinterpret_u8_u32(c0.val[0]);
    v1 = vreinterpret_u8_u32(c2.val[0]);
    v2 = vreinterpret_u8_u32(c1.val[0]);
    v3 = vreinterpret_u8_u32(c3.val[0]);
    v4 = vreinterpret_u8_u32(c0.val[1]);
    v5 = vreinterpret_u8_u32(c2.val[1]);
    v6 = vreinterpret_u8_u32(c1.val[1]);
    v7 = vreinterpret_u8_u32(c3.val[1]);
}

//
// Pack 8-bit quantized B data with 8N column interleaving.
//
// For full N=8 column blocks:
//   Two Transpose8x8 operations (since 8-bit has 16 bytes per column per k_blk
//   vs 4-bit's 8 bytes). After packing, each 8-byte load gives 1 value per
//   column for one K position, enabling vectorized per-column scale FMA.
//
// For remainder N<8 columns:
//   Data is copied as-is (no interleaving needed since N=1 dequant uses
//   broadcast scale).
//
void
HQ8BitGemmPackQuantBData_CompFp16(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    MLAS_UNREFERENCED_PARAMETER(ComputeType);
    constexpr size_t nbits = 8;
    constexpr size_t k_blk_dim = 16;
    constexpr size_t n_blk_dim = 8;
    assert(BlkLen > 0 && BlkLen % k_blk_dim == 0);

    const size_t k_blk_num = MlasDivRoundup(K, k_blk_dim);
    const size_t n_blk_num = MlasDivRoundup(N, n_blk_dim);
    constexpr size_t k_blk_bytes = MlasQNBitBlkDataSizeInBytes(nbits, k_blk_dim);  // = 16
    const size_t iterations = k_blk_num * n_blk_num;
    const size_t ld = MlasDivRoundup(K, BlkLen) * MlasQNBitBlkDataSizeInBytes(nbits, BlkLen);

    MlasTrySimpleParallel(
        ThreadPool, iterations,
        [&](ptrdiff_t tid) {
            const size_t n_blk = tid / k_blk_num;
            const size_t k_blk = tid % k_blk_num;
            size_t n = n_blk * n_blk_dim;
            const size_t src_offset = n * ld + k_blk * k_blk_bytes;

            if (n + n_blk_dim <= N) {
                // Full 8-column block: 8N interleave via two Transpose8x8
                // Output offset: each tile = k_blk_bytes * n_blk_dim = 128 bytes
                const size_t dst_offset = n * ld + k_blk * k_blk_bytes * n_blk_dim;
                const uint8_t* src = reinterpret_cast<const uint8_t*>(QuantBDataBegin) + src_offset;
                uint8_t* dst = reinterpret_cast<uint8_t*>(PackedQuantBDataBegin) + dst_offset;

                // First Transpose8x8: bytes [0..7] of each column (K positions [0..7])
                uint8x8_t v0 = vld1_u8(src);
                uint8x8_t v1 = vld1_u8(src + ld);
                uint8x8_t v2 = vld1_u8(src + 2 * ld);
                uint8x8_t v3 = vld1_u8(src + 3 * ld);
                uint8x8_t v4 = vld1_u8(src + 4 * ld);
                uint8x8_t v5 = vld1_u8(src + 5 * ld);
                uint8x8_t v6 = vld1_u8(src + 6 * ld);
                uint8x8_t v7 = vld1_u8(src + 7 * ld);

                Transpose8x8_8bit(v0, v1, v2, v3, v4, v5, v6, v7);

                vst1_u8(dst, v0);
                vst1_u8(dst + 8, v1);
                vst1_u8(dst + 16, v2);
                vst1_u8(dst + 24, v3);
                vst1_u8(dst + 32, v4);
                vst1_u8(dst + 40, v5);
                vst1_u8(dst + 48, v6);
                vst1_u8(dst + 56, v7);

                // Second Transpose8x8: bytes [8..15] of each column (K positions [8..15])
                uint8x8_t w0 = vld1_u8(src + 8);
                uint8x8_t w1 = vld1_u8(src + ld + 8);
                uint8x8_t w2 = vld1_u8(src + 2 * ld + 8);
                uint8x8_t w3 = vld1_u8(src + 3 * ld + 8);
                uint8x8_t w4 = vld1_u8(src + 4 * ld + 8);
                uint8x8_t w5 = vld1_u8(src + 5 * ld + 8);
                uint8x8_t w6 = vld1_u8(src + 6 * ld + 8);
                uint8x8_t w7 = vld1_u8(src + 7 * ld + 8);

                Transpose8x8_8bit(w0, w1, w2, w3, w4, w5, w6, w7);

                vst1_u8(dst + 64, w0);
                vst1_u8(dst + 72, w1);
                vst1_u8(dst + 80, w2);
                vst1_u8(dst + 88, w3);
                vst1_u8(dst + 96, w4);
                vst1_u8(dst + 104, w5);
                vst1_u8(dst + 112, w6);
                vst1_u8(dst + 120, w7);
            } else {
                // Remainder N < 8: copy as-is (no interleaving needed)
                const uint8_t* src = reinterpret_cast<const uint8_t*>(QuantBDataBegin) + src_offset;
                uint8_t* dst = reinterpret_cast<uint8_t*>(PackedQuantBDataBegin) + src_offset;

                for (; n < N; ++n, src += ld, dst += ld) {
                    std::memcpy(dst, src, k_blk_bytes);
                }
            }
        }
    );
}

//
// 8-bit dequant kernel for N=8 columns, 16 K positions.
// Input: 128 bytes of 8N-interleaved data (16 loads of 8 bytes each).
// Each 8-byte load has 1 value per column (from Transpose8x8 packing).
// Output: 16 x float16x8_t = 128 fp16 values (same layout as 4-bit dequant).
//
template <size_t N, size_t K>
MLAS_FORCEINLINE
typename std::enable_if_t<(N == 8 && K == 16), void>
HQ8BitBlkDequantBKernel(
    const std::uint8_t* src_ptr,
    const float16x8_t& scale,
    const float16x8_t& neg_scaled_zp,
    _mlas_fp16_* dst_ptr
)
{
    // Load 16 x uint8x8_t (128 bytes): each has 1 byte per column
    uint8x8_t b0 = vld1_u8(src_ptr);
    uint8x8_t b1 = vld1_u8(src_ptr + 8);
    uint8x8_t b2 = vld1_u8(src_ptr + 16);
    uint8x8_t b3 = vld1_u8(src_ptr + 24);
    uint8x8_t b4 = vld1_u8(src_ptr + 32);
    uint8x8_t b5 = vld1_u8(src_ptr + 40);
    uint8x8_t b6 = vld1_u8(src_ptr + 48);
    uint8x8_t b7 = vld1_u8(src_ptr + 56);
    uint8x8_t b8 = vld1_u8(src_ptr + 64);
    uint8x8_t b9 = vld1_u8(src_ptr + 72);
    uint8x8_t ba = vld1_u8(src_ptr + 80);
    uint8x8_t bb = vld1_u8(src_ptr + 88);
    uint8x8_t bc = vld1_u8(src_ptr + 96);
    uint8x8_t bd = vld1_u8(src_ptr + 104);
    uint8x8_t be = vld1_u8(src_ptr + 112);
    uint8x8_t bf = vld1_u8(src_ptr + 120);

    // Widen uint8x8 → uint16x8 → float16x8 and dequantize
    float16x8_t f0 = vcvtq_f16_u16(vshll_n_u8(b0, 0));
    float16x8_t f1 = vcvtq_f16_u16(vshll_n_u8(b1, 0));
    float16x8_t f2 = vcvtq_f16_u16(vshll_n_u8(b2, 0));
    float16x8_t f3 = vcvtq_f16_u16(vshll_n_u8(b3, 0));
    float16x8_t f4 = vcvtq_f16_u16(vshll_n_u8(b4, 0));
    float16x8_t f5 = vcvtq_f16_u16(vshll_n_u8(b5, 0));
    float16x8_t f6 = vcvtq_f16_u16(vshll_n_u8(b6, 0));
    float16x8_t f7 = vcvtq_f16_u16(vshll_n_u8(b7, 0));
    float16x8_t f8 = vcvtq_f16_u16(vshll_n_u8(b8, 0));
    float16x8_t f9 = vcvtq_f16_u16(vshll_n_u8(b9, 0));
    float16x8_t fa = vcvtq_f16_u16(vshll_n_u8(ba, 0));
    float16x8_t fb = vcvtq_f16_u16(vshll_n_u8(bb, 0));
    float16x8_t fc = vcvtq_f16_u16(vshll_n_u8(bc, 0));
    float16x8_t fd = vcvtq_f16_u16(vshll_n_u8(bd, 0));
    float16x8_t fe = vcvtq_f16_u16(vshll_n_u8(be, 0));
    float16x8_t ff = vcvtq_f16_u16(vshll_n_u8(bf, 0));

    float16x8_t c0 = vfmaq_f16(neg_scaled_zp, f0, scale);
    float16x8_t c1 = vfmaq_f16(neg_scaled_zp, f1, scale);
    float16x8_t c2 = vfmaq_f16(neg_scaled_zp, f2, scale);
    float16x8_t c3 = vfmaq_f16(neg_scaled_zp, f3, scale);
    float16x8_t c4 = vfmaq_f16(neg_scaled_zp, f4, scale);
    float16x8_t c5 = vfmaq_f16(neg_scaled_zp, f5, scale);
    float16x8_t c6 = vfmaq_f16(neg_scaled_zp, f6, scale);
    float16x8_t c7 = vfmaq_f16(neg_scaled_zp, f7, scale);
    float16x8_t c8 = vfmaq_f16(neg_scaled_zp, f8, scale);
    float16x8_t c9 = vfmaq_f16(neg_scaled_zp, f9, scale);
    float16x8_t ca = vfmaq_f16(neg_scaled_zp, fa, scale);
    float16x8_t cb = vfmaq_f16(neg_scaled_zp, fb, scale);
    float16x8_t cc = vfmaq_f16(neg_scaled_zp, fc, scale);
    float16x8_t cd = vfmaq_f16(neg_scaled_zp, fd, scale);
    float16x8_t ce = vfmaq_f16(neg_scaled_zp, fe, scale);
    float16x8_t cf = vfmaq_f16(neg_scaled_zp, ff, scale);

    MlasStoreFloat16x8(dst_ptr, c0);
    MlasStoreFloat16x8(dst_ptr + 8, c1);
    MlasStoreFloat16x8(dst_ptr + 16, c2);
    MlasStoreFloat16x8(dst_ptr + 24, c3);
    MlasStoreFloat16x8(dst_ptr + 32, c4);
    MlasStoreFloat16x8(dst_ptr + 40, c5);
    MlasStoreFloat16x8(dst_ptr + 48, c6);
    MlasStoreFloat16x8(dst_ptr + 56, c7);
    MlasStoreFloat16x8(dst_ptr + 64, c8);
    MlasStoreFloat16x8(dst_ptr + 72, c9);
    MlasStoreFloat16x8(dst_ptr + 80, ca);
    MlasStoreFloat16x8(dst_ptr + 88, cb);
    MlasStoreFloat16x8(dst_ptr + 96, cc);
    MlasStoreFloat16x8(dst_ptr + 104, cd);
    MlasStoreFloat16x8(dst_ptr + 112, ce);
    MlasStoreFloat16x8(dst_ptr + 120, cf);
}

//
// 8-bit dequant kernel for N=1 (remainder columns).
// Input: 16 bytes (sequential, not interleaved).
// Output: 2 x float16x8_t = 16 fp16 values.
//
template <size_t N, size_t K>
MLAS_FORCEINLINE
typename std::enable_if_t<(N == 1 && K == 16), void>
HQ8BitBlkDequantBKernel(
    const std::uint8_t* src_ptr,
    const float16x8_t& scale,
    const float16x8_t& neg_scaled_zp,
    _mlas_fp16_* dst_ptr
)
{
    uint8x16_t raw = vld1q_u8(src_ptr);

    float16x8_t f_lo = vcvtq_f16_u16(vmovl_u8(vget_low_u8(raw)));
    float16x8_t f_hi = vcvtq_f16_u16(vmovl_u8(vget_high_u8(raw)));

    float16x8_t c0 = vfmaq_f16(neg_scaled_zp, f_lo, scale);
    float16x8_t c1 = vfmaq_f16(neg_scaled_zp, f_hi, scale);

    MlasStoreFloat16x8(dst_ptr, c0);
    MlasStoreFloat16x8(dst_ptr + 8, c1);
}

//
// Dequantize 8-bit packed (8N-interleaved) quantized B data to fp16.
// For N=8: reads from 8N-interleaved packed data (128 bytes per 16K tile).
// For N<8: reads from sequential (unmodified) data (16 bytes per 16K tile).
//
void
HQ8BitBlkDequantBForHgemm_CompFp16(
    size_t BlkLen,
    MLAS_FP16* FpData,
    const std::byte* QuantBData,
    const MLAS_FP16* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t K,
    size_t BlockCountK
)
{
    MLAS_UNREFERENCED_PARAMETER(K);
    constexpr size_t nbits = 8;
    constexpr size_t kk_blk_dim = 16;
    constexpr size_t n_blk_dim = 8;
    assert(BlkLen > 0 && BlkLen % kk_blk_dim == 0);

    constexpr size_t kk_blk_bytes = MlasQNBitBlkDataSizeInBytes(nbits, kk_blk_dim);  // = 16
    const size_t kk_n_src_bytes = kk_blk_bytes * n_blk_dim;  // 128 bytes per 16K x 8N tile
    const size_t kk_n_dst_size = kk_blk_dim * n_blk_dim;     // 128 fp16 values per tile
    const size_t kk_blk_num = BlockCountK * BlkLen / kk_blk_dim;
    const size_t ld_blk_src = kk_blk_num * kk_n_src_bytes;
    const size_t ld_blk_dst = BlkLen * BlockCountK * n_blk_dim;
    const size_t ld_blk_scale = BlockCountK * n_blk_dim;
    // For 8-bit: ZP is 1 byte per block (not packed 2-per-byte like 4-bit)
    const size_t ld_zp = BlockCountK;
    const size_t ld_blk_zp = ld_zp * n_blk_dim;
    // Default zero point for 8-bit unsigned is 128
    const float16x8_t zp_mid_point_vec = MlasBroadcastFloat16x8(MLAS_FP16(128.0f).val);
    const bool has_zp = QuantBZeroPoint != nullptr;

    // Process full 8N column blocks (data is 8N-interleaved from packing)
    size_t n = 0;
    for (; n + n_blk_dim <= CountN; n += n_blk_dim) {
        const auto* scales_ptr = reinterpret_cast<const _mlas_fp16_*>(QuantBScale);
        const std::uint8_t* zero_points_ptr = reinterpret_cast<const uint8_t*>(QuantBZeroPoint);
        const std::uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(QuantBData);
        auto* dst_ptr = reinterpret_cast<_mlas_fp16_*>(FpData);

        for (size_t k_blk_i = 0; k_blk_i < BlockCountK; ++k_blk_i) {
            _mlas_fp16_ scales[n_blk_dim];
            float16x8_t scale_vec;
            float16x8_t neg_scaled_zp_vec;

            UnrolledLoop<n_blk_dim>([&](int nn) {
                scales[nn] = scales_ptr[nn * BlockCountK];
            });
            scale_vec = MlasLoadFloat16x8(scales);

            if (has_zp) {
                uint16_t zero_points[n_blk_dim];
                UnrolledLoop<n_blk_dim>([&](int nn) {
                    // 8-bit: 1 ZP per byte, no nibble packing
                    zero_points[nn] = static_cast<uint16_t>(zero_points_ptr[nn * ld_zp]);
                });
                uint16x8_t zp_u16_vec = vld1q_u16(zero_points);
                neg_scaled_zp_vec = vcvtq_f16_u16(zp_u16_vec);
            } else {
                neg_scaled_zp_vec = zp_mid_point_vec;
            }
            neg_scaled_zp_vec = vnegq_f16(vmulq_f16(scale_vec, neg_scaled_zp_vec));

            for (size_t kk = 0; kk < BlkLen; kk += kk_blk_dim) {
                HQ8BitBlkDequantBKernel<8, 16>(src_ptr, scale_vec, neg_scaled_zp_vec, dst_ptr);

                src_ptr += kk_n_src_bytes;
                dst_ptr += kk_n_dst_size;
            }

            ++scales_ptr;
            if (has_zp) {
                ++zero_points_ptr;
            }
        }

        QuantBData += ld_blk_src;
        FpData += ld_blk_dst;
        QuantBScale += ld_blk_scale;
        QuantBZeroPoint = has_zp ? QuantBZeroPoint + ld_blk_zp : nullptr;
    }

    // Process remaining columns one by one (data is sequential, not interleaved)
    for (; n < CountN; ++n) {
        const auto* scales_ptr = reinterpret_cast<const _mlas_fp16_*>(QuantBScale);
        const std::uint8_t* zero_points_ptr = reinterpret_cast<const uint8_t*>(QuantBZeroPoint);

        for (size_t k_blk_i = 0; k_blk_i < BlockCountK; ++k_blk_i) {
            const auto scale = scales_ptr[0];
            float16x8_t scale_vec = MlasBroadcastFloat16x8(scale);
            float16x8_t neg_scaled_zp_vec;

            if (has_zp) {
                uint8_t zero_point = zero_points_ptr[0];
                uint16x8_t zp_u16_vec = vdupq_n_u16(static_cast<uint16_t>(zero_point));
                neg_scaled_zp_vec = vcvtq_f16_u16(zp_u16_vec);
            } else {
                neg_scaled_zp_vec = zp_mid_point_vec;
            }
            neg_scaled_zp_vec = vnegq_f16(vmulq_f16(scale_vec, neg_scaled_zp_vec));

            for (size_t kk = 0; kk < BlkLen; kk += kk_blk_dim) {
                HQ8BitBlkDequantBKernel<1, 16>(
                    reinterpret_cast<const uint8_t*>(QuantBData), scale_vec, neg_scaled_zp_vec,
                    reinterpret_cast<_mlas_fp16_*>(FpData)
                );

                QuantBData += kk_blk_bytes;
                FpData += kk_blk_dim;
            }

            ++scales_ptr;
            if (has_zp) {
                ++zero_points_ptr;
            }
        }

        QuantBScale += BlockCountK;
        if (has_zp) {
            QuantBZeroPoint += ld_zp;
        }
    }
}

}  // namespace sqnbitgemm_neon
