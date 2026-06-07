/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512vnni_2bit_blklen64.h

Abstract:

    AVX-512 tiled kernel for the 2-bit weight CompInt8 GEMM path
    (BlkBitWidth=2, BlkLen=64). Header-only; the kernel is templated on a
    `bool kVnni` parameter so the same source supports both:

      * AVX-512-VNNI host: `kVnni == true`  -> single `_mm512_dpbusd_epi32`
        per block MAC. Included by sqnbitgemm_kernel_avx512vnni.cpp.
      * AVX-512 (no VNNI): `kVnni == false` -> three-instruction MAC chain
        `vpmaddubsw + vpmaddwd + vpaddd` (AVX-512BW only). Included by
        sqnbitgemm_kernel_avx512.cpp.

    The file name still carries `vnni` for git-history continuity; the
    kernel handles both ISA targets via the template parameter, mirroring
    W4's `dot_accumulate_2blk` / `dot_accumulate_2blkvnni` split in
    sqnbitgemm_kernel_avx512_int8_blklen64.h.

    Architecture mirrors W4's MlasQ4Int8GemmKernelBlkLen64Avx512<vnni>
    (in sqnbitgemm_kernel_avx512_int8_blklen64.h):

      * Per-(m,n) accumulator is a __m512 carrying lane-interleaved
        scaled partial sums. Final _mm512_reduce_add_ps happens ONCE per
        (m,n) tile, not per K-block.
      * Outer tile shape is R2 x C4 (2 M-rows x 4 N-cols) so each A vector
        load amortises across 4 N-cols and each B load across 2 M-rows.
        16 ZMM accumulators in flight.
      * Inner unroll is PerAccuBlk2 = 2 K-blocks per iteration. Pairs of
        dpbusd outputs are interleaved (unpacklo/hi + add) so a single
        FMA applies both blocks' scales in one shot.
      * Packed-B layout is 4-N-col-grouped + 2-K-block-paired (matches W4),
        so the 4 cols of a tile read as a contiguous 128-byte stream.
      * BlkLen=64 only.
      * Symmetric quantization only (QuantBZeroPoint must be null).
      * Tail tiles R2xC1, R1xC4, R1xC1 cover M % 2 != 0 / N % 4 != 0.

    Zero-point correction is performed OUTSIDE the int8 kernel via the
    platform float SGEMM kernel (GetMlasPlatform().GemmFloatKernel), exactly
    as W4 does. This requires QuantBBlkSum to be in the W4 "width-16 row-
    major chunked" layout, which is produced by
    SQ2BitGemmPackQuantBDataAndBlkSum_Scalar.

    Dequant prologue (per 64-element block):

        __m128i p   = _mm_loadu_si128(packed);         // 16 packed bytes
        __m512i p4  = _mm512_broadcast_i32x4(p);       // 4 lanes of those 16 bytes
        __m512i sh  = {0,0,0,0, 2,2,2,2, 4,4,4,4, 6,6,6,6};  // per-dword shifts
        __m512i v   = _mm512_srlv_epi32(p4, sh);
        __m512i b   = _mm512_and_si512(v, _mm512_set1_epi8(0x03));

--*/

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <immintrin.h>

#include "mlasi.h"
#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx512_2bit.h"

namespace onnxruntime {
namespace mlas {
namespace sq2bit_avx512 {

// kNCols4 (= 4) and kPerAccuBlk2 (= 2) come from the shared header so the
// pack layout and the kernel use the same constants. kNRows2 is the M-tile
// shape, kernel-only.
inline constexpr size_t kNRows2 = 2;

//
// Dequant one 64-element 2-bit weight block from the packed (broadcast +
// shift) layout into a ZMM of 64 unsigned bytes in [0, 3].
//
// Bytes 0..15  : weights[0..15]   (shift 0, & 0x03)
// Bytes 16..31 : weights[16..31]  (shift 2, & 0x03)
// Bytes 32..47 : weights[32..47]  (shift 4, & 0x03)
// Bytes 48..63 : weights[48..63]  (shift 6, & 0x03)
//
static MLAS_FORCEINLINE __m512i
unpack_w2_blk_to_zmm(__m128i p128)
{
    const __m512i p_dup = _mm512_broadcast_i32x4(p128);
    // Per-dword right shifts, in memory order (lane 0 first):
    //   [0,0,0,0, 2,2,2,2, 4,4,4,4, 6,6,6,6]
    // _mm512_set_epi32 takes args in reverse order (lane 15 first).
    const __m512i shifts = _mm512_set_epi32(
        6, 6, 6, 6,
        4, 4, 4, 4,
        2, 2, 2, 2,
        0, 0, 0, 0);
    const __m512i mask03 = _mm512_set1_epi8(0x03);
    return _mm512_and_si512(_mm512_srlv_epi32(p_dup, shifts), mask03);
}

//
// Load + dequant ONE block. Used by the single-block (tail) helpers.
//
static MLAS_FORCEINLINE __m512i
load_unpack_1blk_w2(const std::byte* packed)
{
    return unpack_w2_blk_to_zmm(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed)));
}

//
// Load + dequant TWO consecutive K-blocks via one 256-bit YMM load.
// 32 packed bytes -> 2 ZMMs of 64 weights each.
//
static MLAS_FORCEINLINE void
load_unpack_2blk_w2(const std::byte* packed, __m512i& bv0_64_epi8, __m512i& bv1_64_epi8)
{
    const __m256i p_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    bv0_64_epi8 = unpack_w2_blk_to_zmm(_mm256_castsi256_si128(p_ymm));
    bv1_64_epi8 = unpack_w2_blk_to_zmm(_mm256_extracti128_si256(p_ymm, 1));
}

//
// Lane-interleaved 2-K-block accumulator (single M-row, single N-col).
// Mirrors W4's dot_accumulate_2blk / dot_accumulate_2blkvnni split (in
// sqnbitgemm_kernel_avx512_int8_blklen64.h). Identical math for W2 because
// the dequanted block is already in the same uint8 [0,3] form W4 produces.
//
//   acc += sum_2blks( cvt(dot(bv, av)) * scale_a * scale_b )
//
// Two variants:
//   * dot_accumulate_2blk_w2: AVX-512BW only (vpmaddubsw + vpmaddwd + add).
//                             Reduces at epi16 granularity, then ONE vpmaddwd
//                             folds adjacent pairs to epi32 (one madd_epi16
//                             saved per K-block-pair vs an epi32-interleave
//                             approach).
//   * dot_accumulate_2blk_w2_vnni: VNNI variant using _mm512_dpbusd_epi32
//                                  for the inner MAC.
//
// `scale_a` and `scale_b` point to TWO consecutive floats (scales for blk0
// and blk1). The double-broadcast trick gives the 16-lane pattern
//   [s0,s1, s0,s1, s0,s1, s0,s1, s0,s1, s0,s1, s0,s1, s0,s1]
// which matches the post-interleave lane layout (blk0 in even lanes,
// blk1 in odd lanes).
//
static MLAS_FORCEINLINE __m512i
ones_32_epi16_w2()
{
    const __m512i zeros = _mm512_setzero_si512();
    return _mm512_srli_epi16(_mm512_ternarylogic_epi64(zeros, zeros, zeros, 1), 15);
}

static MLAS_FORCEINLINE void
dot_accumulate_2blk_w2(
    const __m512i& av0_64_epi8,
    const __m512i& av1_64_epi8,
    const float* scale_a,
    const __m512i& bv0_64_epi8,
    const __m512i& bv1_64_epi8,
    const __m512& scale_b_16_ps,
    __m512& acc)
{
    const __m512i dot0_32_epi16 = _mm512_maddubs_epi16(bv0_64_epi8, av0_64_epi8);
    const __m512i dot1_32_epi16 = _mm512_maddubs_epi16(bv1_64_epi8, av1_64_epi8);

    const __m512i t1 = _mm512_unpacklo_epi32(dot0_32_epi16, dot1_32_epi16);
    const __m512i t2 = _mm512_unpackhi_epi32(dot0_32_epi16, dot1_32_epi16);
    const __m512i sum_32_epi16 = _mm512_add_epi16(t1, t2);  // [b0 b0 b1 b1 ...] in epi16
    const __m512i ones = ones_32_epi16_w2();
    const __m512i sum_16_epi32 = _mm512_madd_epi16(ones, sum_32_epi16);  // [b0 b1 b0 b1 ...] in epi32
    const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);

    const __m256 scale_a_8_ps = _mm256_castpd_ps(_mm256_broadcast_sd(reinterpret_cast<const double*>(scale_a)));
    const __m512 scale_a_16_ps = _mm512_broadcast_f32x8(scale_a_8_ps);

    acc = _mm512_fmadd_ps(sum_16_ps, _mm512_mul_ps(scale_a_16_ps, scale_b_16_ps), acc);
}

static MLAS_FORCEINLINE void
dot_accumulate_2blk_w2_vnni(
    const __m512i& av0_64_epi8,
    const __m512i& av1_64_epi8,
    const float* scale_a,
    const __m512i& bv0_64_epi8,
    const __m512i& bv1_64_epi8,
    const __m512& scale_b_16_ps,
    __m512& acc)
{
    const __m512i dot0_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv0_64_epi8, av0_64_epi8);
    const __m512i dot1_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv1_64_epi8, av1_64_epi8);

    const __m512i t1 = _mm512_unpacklo_epi32(dot0_16_epi32, dot1_16_epi32);
    const __m512i t2 = _mm512_unpackhi_epi32(dot0_16_epi32, dot1_16_epi32);
    const __m512i sum_16_epi32 = _mm512_add_epi32(t1, t2);
    const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);

    const __m256 scale_a_8_ps = _mm256_castpd_ps(_mm256_broadcast_sd(reinterpret_cast<const double*>(scale_a)));
    const __m512 scale_a_16_ps = _mm512_broadcast_f32x8(scale_a_8_ps);

    acc = _mm512_fmadd_ps(sum_16_ps, _mm512_mul_ps(scale_a_16_ps, scale_b_16_ps), acc);
}

//
// Single-K-block accumulator. Uses uniform 16-lane scale broadcast since
// there's only one block's scale in play.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
dot_accumulate_1blk_w2(
    const __m512i& av_64_epi8,
    const float* scale_a,
    const __m512i& bv_64_epi8,
    const __m512& scale_b_16_ps,
    __m512& acc)
{
    __m512i dot_16_epi32;
    if constexpr (kVnni) {
        dot_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv_64_epi8, av_64_epi8);
    } else {
        const __m512i ones = ones_32_epi16_w2();
        const __m512i dot_32_epi16 = _mm512_maddubs_epi16(bv_64_epi8, av_64_epi8);
        dot_16_epi32 = _mm512_madd_epi16(dot_32_epi16, ones);
    }
    const __m512 sum_16_ps = _mm512_cvtepi32_ps(dot_16_epi32);

    const __m128 scale_a_ps = _mm_broadcast_ss(scale_a);
    const __m512 scale_a_16_ps = _mm512_broadcast_f32x2(scale_a_ps);

    acc = _mm512_fmadd_ps(sum_16_ps, _mm512_mul_ps(scale_a_16_ps, scale_b_16_ps), acc);
}

//
// 2 M-rows x 1 N-col x 2 K-blocks accumulator. The 2-block B load is shared
// across the 2 M-rows.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen64_r2c1blk2(
    const __m512i& av00_64_epi8, const __m512i& av01_64_epi8,
    const __m512i& av10_64_epi8, const __m512i& av11_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_a1,
    const float* scale_b,
    __m512& acc0,
    __m512& acc1)
{
    __m512i bv0, bv1;
    load_unpack_2blk_w2(QuantBDataPtr, bv0, bv1);

    const __m256 scale_b_8_ps = _mm256_castpd_ps(_mm256_broadcast_sd(reinterpret_cast<const double*>(scale_b)));
    const __m512 scale_b_16_ps = _mm512_broadcast_f32x8(scale_b_8_ps);

    if constexpr (kVnni) {
        dot_accumulate_2blk_w2_vnni(av00_64_epi8, av01_64_epi8, scale_a0, bv0, bv1, scale_b_16_ps, acc0);
        dot_accumulate_2blk_w2_vnni(av10_64_epi8, av11_64_epi8, scale_a1, bv0, bv1, scale_b_16_ps, acc1);
    } else {
        dot_accumulate_2blk_w2(av00_64_epi8, av01_64_epi8, scale_a0, bv0, bv1, scale_b_16_ps, acc0);
        dot_accumulate_2blk_w2(av10_64_epi8, av11_64_epi8, scale_a1, bv0, bv1, scale_b_16_ps, acc1);
    }
}

//
// 2 M-rows x 1 N-col x 1 K-block accumulator (K-tail).
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen64_r2c1blk1(
    const __m512i& av0_64_epi8,
    const __m512i& av1_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_a1,
    const float* scale_b,
    __m512& acc0,
    __m512& acc1)
{
    const __m512i bv = load_unpack_1blk_w2(QuantBDataPtr);

    const __m128 scale_b_ps = _mm_broadcast_ss(scale_b);
    const __m512 scale_b_16_ps = _mm512_broadcast_f32x2(scale_b_ps);

    dot_accumulate_1blk_w2<kVnni>(av0_64_epi8, scale_a0, bv, scale_b_16_ps, acc0);
    dot_accumulate_1blk_w2<kVnni>(av1_64_epi8, scale_a1, bv, scale_b_16_ps, acc1);
}

//
// 1 M-row x 1 N-col x 2 K-blocks accumulator.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen64_r1c1blk2(
    const __m512i& av0_64_epi8,
    const __m512i& av1_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    __m512& acc)
{
    __m512i bv0, bv1;
    load_unpack_2blk_w2(QuantBDataPtr, bv0, bv1);

    const __m256 scale_b_8_ps = _mm256_castpd_ps(_mm256_broadcast_sd(reinterpret_cast<const double*>(scale_b)));
    const __m512 scale_b_16_ps = _mm512_broadcast_f32x8(scale_b_8_ps);

    if constexpr (kVnni) {
        dot_accumulate_2blk_w2_vnni(av0_64_epi8, av1_64_epi8, scale_a, bv0, bv1, scale_b_16_ps, acc);
    } else {
        dot_accumulate_2blk_w2(av0_64_epi8, av1_64_epi8, scale_a, bv0, bv1, scale_b_16_ps, acc);
    }
}

//
// 1 M-row x 1 N-col x 1 K-block accumulator.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen64_r1c1blk1(
    const __m512i& av_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    __m512& acc)
{
    const __m512i bv = load_unpack_1blk_w2(QuantBDataPtr);

    const __m128 scale_b_ps = _mm_broadcast_ss(scale_b);
    const __m512 scale_b_16_ps = _mm512_broadcast_f32x2(scale_b_ps);

    dot_accumulate_1blk_w2<kVnni>(av_64_epi8, scale_a, bv, scale_b_16_ps, acc);
}

//
// R2 x C4 tile. Main hot path for the customer model.
//
// Layout assumptions (matches the W4 layout produced by W2's pack function):
//   * QuantBData : 4-N-col grouped, 2-K-block paired (W4-style). Within a
//     group the inner col stride is kPerAccuBlk2 * kBlkBytes (32 B) for pair
//     blocks and kBlkBytes (16 B) for the single-block trailing slot when
//     BlockCountK is odd. Per-group total = BlockCountK * kNCols4 * kBlkBytes.
//   * QuantBScale: same grouping; col stride kPerAccuBlk2 (2 floats) /
//     1 float in the single-block slot. Per-group total = BlockCountK * kNCols4.
//   * QuantA     : row-major int8, BlockCountK * kBlkLen bytes per row.
//   * QuantAScale: row-major float, BlockCountK floats per row.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR2xC4BlkLen64Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc)
{
    const size_t lda = BlockCountK * kBlkLen;
    // Per-group strides for the 4-N-col grouped layout.
    constexpr size_t PerColPairBytes = kPerAccuBlk2 * kBlkBytes;        // 32 B per col in a K-pair slot
    constexpr size_t PerColSingleBytes = kBlkBytes;                     // 16 B per col in a K-single slot
    constexpr size_t PerColPairScale = kPerAccuBlk2;                    // 2 floats per col in a K-pair slot
    constexpr size_t PerKPairAdvanceBytes = kNCols4 * PerColPairBytes;  // 128 B per K-pair iteration
    constexpr size_t PerKSingleAdvanceBytes = kNCols4 * PerColSingleBytes;
    constexpr size_t PerKPairAdvanceScale = kNCols4 * PerColPairScale;  // 8 floats per K-pair iteration
    constexpr size_t PerKSingleAdvanceScale = kNCols4;
    const size_t GroupStrideBytes = BlockCountK * kNCols4 * kBlkBytes;
    const size_t GroupStrideScale = BlockCountK * kNCols4;

    assert(CountM % kNRows2 == 0);
    assert(CountN % kNCols4 == 0);

    for (size_t m = 0; m < CountM; m += kNRows2) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        float* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += kNCols4) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m512 acc[kNCols4 * kNRows2] = {
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
            };

            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining > 1; k_blks_remaining -= kPerAccuBlk2) {
                const __m512i av_00 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr));
                const __m512i av_01 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + kBlkLen));
                const __m512i av_10 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + lda));
                const __m512i av_11 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + lda + kBlkLen));

                accumulate_w2_blklen64_r2c1blk2<kVnni>(
                    av_00, av_01, av_10, av_11,
                    QuantBDataPtr + 0 * PerColPairBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 0 * PerColPairScale,
                    acc[0], acc[kNCols4 + 0]);
                accumulate_w2_blklen64_r2c1blk2<kVnni>(
                    av_00, av_01, av_10, av_11,
                    QuantBDataPtr + 1 * PerColPairBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 1 * PerColPairScale,
                    acc[1], acc[kNCols4 + 1]);
                accumulate_w2_blklen64_r2c1blk2<kVnni>(
                    av_00, av_01, av_10, av_11,
                    QuantBDataPtr + 2 * PerColPairBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 2 * PerColPairScale,
                    acc[2], acc[kNCols4 + 2]);
                accumulate_w2_blklen64_r2c1blk2<kVnni>(
                    av_00, av_01, av_10, av_11,
                    QuantBDataPtr + 3 * PerColPairBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 3 * PerColPairScale,
                    acc[3], acc[kNCols4 + 3]);

                QuantAPtr += kBlkLen * kPerAccuBlk2;
                QuantAScalePtr += kPerAccuBlk2;
                QuantBDataPtr += PerKPairAdvanceBytes;
                QuantBScalePtr += PerKPairAdvanceScale;
            }

            while (k_blks_remaining-- > 0) {
                const __m512i av_00 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr));
                const __m512i av_10 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + lda));

                accumulate_w2_blklen64_r2c1blk1<kVnni>(
                    av_00, av_10,
                    QuantBDataPtr + 0 * PerColSingleBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 0,
                    acc[0], acc[kNCols4 + 0]);
                accumulate_w2_blklen64_r2c1blk1<kVnni>(
                    av_00, av_10,
                    QuantBDataPtr + 1 * PerColSingleBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 1,
                    acc[1], acc[kNCols4 + 1]);
                accumulate_w2_blklen64_r2c1blk1<kVnni>(
                    av_00, av_10,
                    QuantBDataPtr + 2 * PerColSingleBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 2,
                    acc[2], acc[kNCols4 + 2]);
                accumulate_w2_blklen64_r2c1blk1<kVnni>(
                    av_00, av_10,
                    QuantBDataPtr + 3 * PerColSingleBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 3,
                    acc[3], acc[kNCols4 + 3]);

                QuantAPtr += kBlkLen;
                QuantAScalePtr++;
                QuantBDataPtr += PerKSingleAdvanceBytes;
                QuantBScalePtr += PerKSingleAdvanceScale;
            }

            SumPtr[0] = _mm512_reduce_add_ps(acc[0]);
            SumPtr[1] = _mm512_reduce_add_ps(acc[1]);
            SumPtr[2] = _mm512_reduce_add_ps(acc[2]);
            SumPtr[3] = _mm512_reduce_add_ps(acc[3]);
            SumPtr[ldc + 0] = _mm512_reduce_add_ps(acc[kNCols4 + 0]);
            SumPtr[ldc + 1] = _mm512_reduce_add_ps(acc[kNCols4 + 1]);
            SumPtr[ldc + 2] = _mm512_reduce_add_ps(acc[kNCols4 + 2]);
            SumPtr[ldc + 3] = _mm512_reduce_add_ps(acc[kNCols4 + 3]);
            if (BiasPtr != nullptr) {
                SumPtr[0] += BiasPtr[0];
                SumPtr[1] += BiasPtr[1];
                SumPtr[2] += BiasPtr[2];
                SumPtr[3] += BiasPtr[3];
                SumPtr[ldc + 0] += BiasPtr[0];
                SumPtr[ldc + 1] += BiasPtr[1];
                SumPtr[ldc + 2] += BiasPtr[2];
                SumPtr[ldc + 3] += BiasPtr[3];
            }

            QuantBDataColPtr += GroupStrideBytes;
            QuantBScaleColPtr += GroupStrideScale;
            BiasPtr += BiasPtr != nullptr ? kNCols4 : 0;
            SumPtr += kNCols4;
        }
    }
}

//
// R2 x C1 tile (N-tail). Operates on the column-major tail region of the
// packed B buffer (cols NMain..N-1), which the dispatcher addresses with
// the same `multipleCols * (BlockCountK * kBlkBytes)` offset as the old
// pure column-major layout did (the grouped main region was sized to fit
// in exactly that many bytes by construction).
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR2xC1BlkLen64Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc)
{
    const size_t lda = BlockCountK * kBlkLen;
    const size_t ColStrideBytes = BlockCountK * kBlkBytes;
    const size_t ColStrideScale = BlockCountK;

    assert(CountM % kNRows2 == 0);

    for (size_t m = 0; m < CountM; m += kNRows2) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        float* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining > 1; k_blks_remaining -= kPerAccuBlk2) {
                const __m512i av_00 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr));
                const __m512i av_01 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + kBlkLen));
                const __m512i av_10 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + lda));
                const __m512i av_11 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + lda + kBlkLen));

                accumulate_w2_blklen64_r2c1blk2<kVnni>(
                    av_00, av_01, av_10, av_11,
                    QuantBDataPtr,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr,
                    acc0, acc1);

                QuantAPtr += kBlkLen * kPerAccuBlk2;
                QuantAScalePtr += kPerAccuBlk2;
                QuantBDataPtr += kPerAccuBlk2 * kBlkBytes;
                QuantBScalePtr += kPerAccuBlk2;
            }

            while (k_blks_remaining-- > 0) {
                const __m512i av_00 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr));
                const __m512i av_10 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + lda));

                accumulate_w2_blklen64_r2c1blk1<kVnni>(
                    av_00, av_10,
                    QuantBDataPtr,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr,
                    acc0, acc1);

                QuantAPtr += kBlkLen;
                QuantAScalePtr++;
                QuantBDataPtr += kBlkBytes;
                QuantBScalePtr++;
            }

            SumPtr[0] = _mm512_reduce_add_ps(acc0);
            SumPtr[ldc] = _mm512_reduce_add_ps(acc1);
            if (BiasPtr != nullptr) {
                SumPtr[0] += BiasPtr[0];
                SumPtr[ldc] += BiasPtr[0];
            }

            QuantBDataColPtr += ColStrideBytes;
            QuantBScaleColPtr += ColStrideScale;
            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;
        }
    }
}

//
// R1 x C4 tile (M-tail). Uses the same 4-N-col grouped layout as R2 x C4.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR1xC4BlkLen64Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc)
{
    const size_t lda = BlockCountK * kBlkLen;
    constexpr size_t PerColPairBytes = kPerAccuBlk2 * kBlkBytes;
    constexpr size_t PerColSingleBytes = kBlkBytes;
    constexpr size_t PerColPairScale = kPerAccuBlk2;
    constexpr size_t PerKPairAdvanceBytes = kNCols4 * PerColPairBytes;
    constexpr size_t PerKSingleAdvanceBytes = kNCols4 * PerColSingleBytes;
    constexpr size_t PerKPairAdvanceScale = kNCols4 * PerColPairScale;
    constexpr size_t PerKSingleAdvanceScale = kNCols4;
    const size_t GroupStrideBytes = BlockCountK * kNCols4 * kBlkBytes;
    const size_t GroupStrideScale = BlockCountK * kNCols4;

    assert(CountN % kNCols4 == 0);

    for (size_t m = 0; m < CountM; ++m) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        float* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += kNCols4) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m512 acc[kNCols4] = {
                _mm512_setzero_ps(), _mm512_setzero_ps(),
                _mm512_setzero_ps(), _mm512_setzero_ps()
            };

            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining > 1; k_blks_remaining -= kPerAccuBlk2) {
                const __m512i av_0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr));
                const __m512i av_1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + kBlkLen));

                accumulate_w2_blklen64_r1c1blk2<kVnni>(
                    av_0, av_1,
                    QuantBDataPtr + 0 * PerColPairBytes,
                    QuantAScalePtr, QuantBScalePtr + 0 * PerColPairScale, acc[0]);
                accumulate_w2_blklen64_r1c1blk2<kVnni>(
                    av_0, av_1,
                    QuantBDataPtr + 1 * PerColPairBytes,
                    QuantAScalePtr, QuantBScalePtr + 1 * PerColPairScale, acc[1]);
                accumulate_w2_blklen64_r1c1blk2<kVnni>(
                    av_0, av_1,
                    QuantBDataPtr + 2 * PerColPairBytes,
                    QuantAScalePtr, QuantBScalePtr + 2 * PerColPairScale, acc[2]);
                accumulate_w2_blklen64_r1c1blk2<kVnni>(
                    av_0, av_1,
                    QuantBDataPtr + 3 * PerColPairBytes,
                    QuantAScalePtr, QuantBScalePtr + 3 * PerColPairScale, acc[3]);

                QuantAPtr += kBlkLen * kPerAccuBlk2;
                QuantAScalePtr += kPerAccuBlk2;
                QuantBDataPtr += PerKPairAdvanceBytes;
                QuantBScalePtr += PerKPairAdvanceScale;
            }

            while (k_blks_remaining-- > 0) {
                const __m512i av = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr));

                accumulate_w2_blklen64_r1c1blk1<kVnni>(
                    av, QuantBDataPtr + 0 * PerColSingleBytes,
                    QuantAScalePtr, QuantBScalePtr + 0, acc[0]);
                accumulate_w2_blklen64_r1c1blk1<kVnni>(
                    av, QuantBDataPtr + 1 * PerColSingleBytes,
                    QuantAScalePtr, QuantBScalePtr + 1, acc[1]);
                accumulate_w2_blklen64_r1c1blk1<kVnni>(
                    av, QuantBDataPtr + 2 * PerColSingleBytes,
                    QuantAScalePtr, QuantBScalePtr + 2, acc[2]);
                accumulate_w2_blklen64_r1c1blk1<kVnni>(
                    av, QuantBDataPtr + 3 * PerColSingleBytes,
                    QuantAScalePtr, QuantBScalePtr + 3, acc[3]);

                QuantAPtr += kBlkLen;
                QuantAScalePtr++;
                QuantBDataPtr += PerKSingleAdvanceBytes;
                QuantBScalePtr += PerKSingleAdvanceScale;
            }

            SumPtr[0] = _mm512_reduce_add_ps(acc[0]);
            SumPtr[1] = _mm512_reduce_add_ps(acc[1]);
            SumPtr[2] = _mm512_reduce_add_ps(acc[2]);
            SumPtr[3] = _mm512_reduce_add_ps(acc[3]);
            if (BiasPtr != nullptr) {
                SumPtr[0] += BiasPtr[0];
                SumPtr[1] += BiasPtr[1];
                SumPtr[2] += BiasPtr[2];
                SumPtr[3] += BiasPtr[3];
            }

            QuantBDataColPtr += GroupStrideBytes;
            QuantBScaleColPtr += GroupStrideScale;
            BiasPtr += BiasPtr != nullptr ? kNCols4 : 0;
            SumPtr += kNCols4;
        }
    }
}

//
// R1 x C1 tile (corner). Same column-major tail-region addressing as R2 x C1.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR1xC1BlkLen64Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc)
{
    const size_t lda = BlockCountK * kBlkLen;
    const size_t ColStrideBytes = BlockCountK * kBlkBytes;
    const size_t ColStrideScale = BlockCountK;

    for (size_t m = 0; m < CountM; ++m) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        float* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m512 acc = _mm512_setzero_ps();

            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining > 1; k_blks_remaining -= kPerAccuBlk2) {
                const __m512i av_0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr));
                const __m512i av_1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + kBlkLen));

                accumulate_w2_blklen64_r1c1blk2<kVnni>(
                    av_0, av_1, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc);

                QuantAPtr += kBlkLen * kPerAccuBlk2;
                QuantAScalePtr += kPerAccuBlk2;
                QuantBDataPtr += kPerAccuBlk2 * kBlkBytes;
                QuantBScalePtr += kPerAccuBlk2;
            }

            while (k_blks_remaining-- > 0) {
                const __m512i av = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr));

                accumulate_w2_blklen64_r1c1blk1<kVnni>(
                    av, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc);

                QuantAPtr += kBlkLen;
                QuantAScalePtr++;
                QuantBDataPtr += kBlkBytes;
                QuantBScalePtr++;
            }

            SumPtr[0] = _mm512_reduce_add_ps(acc);
            if (BiasPtr != nullptr) {
                SumPtr[0] += BiasPtr[0];
            }

            QuantBDataColPtr += ColStrideBytes;
            QuantBScaleColPtr += ColStrideScale;
            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;
        }
    }
}

//
// Tile dispatcher. Mirrors W4's MlasQ4Int8GemmKernelBlkLen64Avx512.
//
template <bool kVnni>
MLAS_FORCEINLINE void
MlasQ2Int8GemmKernelBlkLen64Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc)
{
    const size_t lda = BlockCountK * kBlkLen;
    const size_t lda_scale = BlockCountK;
    const size_t ColStrideBytes = BlockCountK * kBlkBytes;
    const size_t ColStrideScale = BlockCountK;

    const size_t remainingRows = CountM % kNRows2;
    const size_t multipleRows = CountM - remainingRows;
    const size_t remainingCols = CountN % kNCols4;
    const size_t multipleCols = CountN - remainingCols;

    if (multipleRows > 0 && multipleCols > 0) {
        Q2Int8GemmR2xC4BlkLen64Avx512<kVnni>(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, multipleRows, multipleCols, BlockCountK, Bias, ldc);
    }
    if (remainingCols > 0 && multipleRows > 0) {
        Q2Int8GemmR2xC1BlkLen64Avx512<kVnni>(
            QuantA, QuantAScale,
            QuantBData + multipleCols * ColStrideBytes,
            QuantBScale + multipleCols * ColStrideScale,
            C + multipleCols,
            multipleRows, remainingCols, BlockCountK,
            Bias ? Bias + multipleCols : nullptr, ldc);
    }
    if (remainingRows > 0 && multipleCols > 0) {
        Q2Int8GemmR1xC4BlkLen64Avx512<kVnni>(
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData, QuantBScale,
            C + multipleRows * ldc,
            remainingRows, multipleCols, BlockCountK, Bias, ldc);
    }
    if (remainingRows > 0 && remainingCols > 0) {
        Q2Int8GemmR1xC1BlkLen64Avx512<kVnni>(
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData + multipleCols * ColStrideBytes,
            QuantBScale + multipleCols * ColStrideScale,
            C + multipleRows * ldc + multipleCols,
            remainingRows, remainingCols, BlockCountK,
            Bias ? Bias + multipleCols : nullptr, ldc);
    }
}

//
// Common dispatched-kernel body. Templated on <bool kVnni>; the two top-level
// wrappers below instantiate it.
//
// Steps:
// 1) Calls the tile dispatcher, which computes
//        C[m,n] = bias[n] + sum_blk(scale_a * scale_b * dot(b, a))
//    using either VNNI (`_mm512_dpbusd_epi32`) or AVX-512BW
//    (`vpmaddubsw + vpmaddwd + vpaddd`) depending on the template parameter.
// 2) Adds the symmetric zero-point correction
//        C[m,n] += sum_blk(ABlockSum[m,blk] * QuantBBlkSum[n,blk])
//    via the platform float SGEMM micro-kernel. QuantBBlkSum is in the
//    W4 width-16 row-major chunked layout (produced by the W2 pack
//    function), which is what GemmFloatKernel expects for its packed-B
//    operand. ZeroMode=false means SGEMM does `C += A @ B`, so the bias
//    and int8 contribution already in C are preserved.
//
template <bool kVnni>
static MLAS_FORCEINLINE size_t
SQ2BitGemmKernel_BlkSum_CompInt8_Impl(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* /* QuantBZeroPoint */,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t /* CountK */,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum)
{
    if (BlkLen != kBlkLen) {
        return 0;
    }

    MlasQ2Int8GemmKernelBlkLen64Avx512<kVnni>(
        QuantA, QuantAScale, QuantBData, QuantBScale,
        C, CountM, CountN, BlockCountK, Bias, ldc);

    // BlkSum correction: C += ABlockSum [M x BlockCountK] @ QuantBBlkSum [BlockCountK x N].
    //
    // TEMP DEBUG: scalar reference instead of GetMlasPlatform().GemmFloatKernel
    // to test whether the SGEMM-call shape/layout assumptions are wrong.
    // QuantBBlkSum is in the "width-16 chunked" layout:
    //   BlkSum[(n/16) * BlockCountK * 16 + blk * 16 + (n%16)]
    {
        for (size_t m = 0; m < CountM; ++m) {
            const float* a_row = ABlockSum + m * BlockCountK;
            float* c_row = C + m * ldc;
            for (size_t n = 0; n < CountN; ++n) {
                const size_t chunk = n / 16;
                const size_t lane  = n % 16;
                float acc = 0.0f;
                for (size_t blk = 0; blk < BlockCountK; ++blk) {
                    const float b = QuantBBlkSum[(chunk * BlockCountK + blk) * 16 + lane];
                    acc += a_row[blk] * b;
                }
                c_row[n] += acc;
            }
        }
    }
    // Original fast path (disabled for debug):
    // float* c_blk = C;
    // const float* b_blk_sum = QuantBBlkSum;
    // size_t RowsRemaining = CountM;
    // const float* a_blksum_row = ABlockSum;
    // while (RowsRemaining > 0) {
    //     const auto RowsHandled = GetMlasPlatform().GemmFloatKernel(
    //         a_blksum_row, b_blk_sum, c_blk,
    //         BlockCountK, RowsRemaining, CountN,
    //         BlockCountK, ldc, 1.0f, false);
    //
    //     c_blk += ldc * RowsHandled;
    //     a_blksum_row += BlockCountK * RowsHandled;
    //     RowsRemaining -= RowsHandled;
    // }

    return CountM;
}

//
// Top-level VNNI variant registered into MlasSQNBitGemmDispatchAvx512vnni.
//
static MLAS_FORCEINLINE size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum)
{
    return SQ2BitGemmKernel_BlkSum_CompInt8_Impl<true>(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale, QuantBZeroPoint,
        C, CountM, CountN, CountK, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

//
// Top-level non-VNNI variant registered into MlasSQNBitGemmDispatchAvx512.
// Uses the AVX-512BW MAC chain (`vpmaddubsw + vpmaddwd + vpaddd`) instead of
// `_mm512_dpbusd_epi32`. Same tile shapes, same pack layout, same numerical
// result.
//
static MLAS_FORCEINLINE size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx512(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum)
{
    return SQ2BitGemmKernel_BlkSum_CompInt8_Impl<false>(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale, QuantBZeroPoint,
        C, CountM, CountN, CountK, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

}  // namespace sq2bit_avx512
}  // namespace mlas
}  // namespace onnxruntime
