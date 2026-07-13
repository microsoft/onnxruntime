/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2_2bit_blklen64.h

Abstract:

    AVX2 (-VNNI) W2 kernel for BlkLen=64, consuming the block-group packed
    layout (sqnbitgemm_kernel_avx512_2bit.h). A 64-element K-block is one ZMM on
    AVX-512; on AVX2 it is two YMM halves, so each block's dot is two chained
    VPDPBUSD (one per 32-lane half) summed into the same int32 vector. Layout,
    stride math, scale handling, and tail logic match the AVX-512 sibling.

    Templated on `<bool kVnni>`. The un-suffixed
    SQ2BitGemmKernel_BlkSum_CompInt8_Avx2(Vnni) wrappers are the default entry
    the dispatcher routes to for BlkLen == 64.

--*/

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <immintrin.h>

#include "mlasi.h"
#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_kernel_avx512_2bit.h"

namespace onnxruntime {
namespace mlas {
namespace sq2bit_avx2 {

using namespace sq2bit_avx512;

inline constexpr size_t kNRows2 = 2;

//
// acc += sum_{k=0..3} scale_a[k]*scale_b[k] * dot64(av_k, bv_k). Each block's
// 64 unsigned weights come from the block-group halves glo/ghi via a fixed
// shift+mask; each block's 64 int8 activations are two YMM halves.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
dot_accumulate_4blk_w2_blklen64(
    const __m256i& glo, const __m256i& ghi,
    const __m256i& a0lo, const __m256i& a0hi, const __m256i& a1lo, const __m256i& a1hi,
    const __m256i& a2lo, const __m256i& a2hi, const __m256i& a3lo, const __m256i& a3hi,
    const float* scale_a, const float* scale_b, __m256& acc)
{
    const __m256i m03 = _mm256_set1_epi8(0x03);
    __m256i d0, d1, d2, d3;
    // GCC 11+ is needed for _mm256_dpbusds_avx_epi32; older toolchains fall
    // through to the vpmaddubsw+vpmaddwd path even on AVX-VNNI hardware.
#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (kVnni) {
        d0 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_and_si256(glo, m03), a0lo);
        d0 = _mm256_dpbusds_avx_epi32(d0, _mm256_and_si256(ghi, m03), a0hi);
        d1 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_and_si256(_mm256_srli_epi16(glo, 2), m03), a1lo);
        d1 = _mm256_dpbusds_avx_epi32(d1, _mm256_and_si256(_mm256_srli_epi16(ghi, 2), m03), a1hi);
        d2 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_and_si256(_mm256_srli_epi16(glo, 4), m03), a2lo);
        d2 = _mm256_dpbusds_avx_epi32(d2, _mm256_and_si256(_mm256_srli_epi16(ghi, 4), m03), a2hi);
        d3 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_and_si256(_mm256_srli_epi16(glo, 6), m03), a3lo);
        d3 = _mm256_dpbusds_avx_epi32(d3, _mm256_and_si256(_mm256_srli_epi16(ghi, 6), m03), a3hi);
    } else
#endif
    {
        const __m256i ones = _mm256_set1_epi16(1);
        auto dot2 = [&](const __m256i& blo, const __m256i& bhi, const __m256i& alo, const __m256i& ahi) -> __m256i {
            const __m256i t_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(blo, alo), ones);
            const __m256i t_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(bhi, ahi), ones);
            return _mm256_add_epi32(t_lo, t_hi);
        };
        d0 = dot2(_mm256_and_si256(glo, m03), _mm256_and_si256(ghi, m03), a0lo, a0hi);
        d1 = dot2(_mm256_and_si256(_mm256_srli_epi16(glo, 2), m03),
                  _mm256_and_si256(_mm256_srli_epi16(ghi, 2), m03), a1lo, a1hi);
        d2 = dot2(_mm256_and_si256(_mm256_srli_epi16(glo, 4), m03),
                  _mm256_and_si256(_mm256_srli_epi16(ghi, 4), m03), a2lo, a2hi);
        d3 = dot2(_mm256_and_si256(_mm256_srli_epi16(glo, 6), m03),
                  _mm256_and_si256(_mm256_srli_epi16(ghi, 6), m03), a3lo, a3hi);
    }

    const __m256 s0 = _mm256_set1_ps(scale_a[0] * scale_b[0]);
    const __m256 s1 = _mm256_set1_ps(scale_a[1] * scale_b[1]);
    const __m256 s2 = _mm256_set1_ps(scale_a[2] * scale_b[2]);
    const __m256 s3 = _mm256_set1_ps(scale_a[3] * scale_b[3]);

    __m256 acc_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(d0), s0);
    __m256 acc_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(d1), s1);
    acc_lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d2), s2, acc_lo);
    acc_hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d3), s3, acc_hi);
    acc = _mm256_add_ps(acc, _mm256_add_ps(acc_lo, acc_hi));
}

static MLAS_FORCEINLINE __m256i
load_a_half_blklen64(const std::byte* p)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen64_r1c1blk4(
    const __m256i& a0lo, const __m256i& a0hi, const __m256i& a1lo, const __m256i& a1hi,
    const __m256i& a2lo, const __m256i& a2hi, const __m256i& a3lo, const __m256i& a3hi,
    const std::byte* QuantBDataPtr,
    const float* scale_a, const float* scale_b, __m256& acc)
{
    const __m256i glo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    const __m256i ghi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));
    dot_accumulate_4blk_w2_blklen64<kVnni>(glo, ghi, a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi,
                                           scale_a, scale_b, acc);
}

//
// One block's contribution for both M-rows, sharing the unpacked bv pair.
// `Shift` selects the 2-bit field (0/2/4/6) for block {0,1,2,3}. A halves are
// loaded per call rather than preloaded: 16 A-half registers for an R2 group
// do not fit the 16-YMM file, and the reloads are L1-resident.
//
template <bool kVnni, int Shift>
static MLAS_FORCEINLINE void
dot_one_block_w2_blklen64_r2(
    const __m256i& glo, const __m256i& ghi, const __m256i& m03,
    const std::byte* a0_blk, const std::byte* a1_blk,
    float s0, float s1, __m256& acc0, __m256& acc1)
{
    const __m256i bvlo = _mm256_and_si256(_mm256_srli_epi16(glo, Shift), m03);
    const __m256i bvhi = _mm256_and_si256(_mm256_srli_epi16(ghi, Shift), m03);
    const __m256i a0lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a0_blk));
    const __m256i a0hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a0_blk + 32));
    const __m256i a1lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a1_blk));
    const __m256i a1hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a1_blk + 32));
    __m256i d0, d1;
#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (kVnni) {
        d0 = _mm256_dpbusds_avx_epi32(_mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bvlo, a0lo), bvhi, a0hi);
        d1 = _mm256_dpbusds_avx_epi32(_mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bvlo, a1lo), bvhi, a1hi);
    } else
#endif
    {
        const __m256i ones = _mm256_set1_epi16(1);
        d0 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(bvlo, a0lo), ones),
                              _mm256_madd_epi16(_mm256_maddubs_epi16(bvhi, a0hi), ones));
        d1 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(bvlo, a1lo), ones),
                              _mm256_madd_epi16(_mm256_maddubs_epi16(bvhi, a1hi), ones));
    }
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(s0), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_set1_ps(s1), acc1);
}

//
// 2 M-rows share one B-group load+unpack. nblk is 4 for a full group, 1..3 for
// the K-tail (guarded per-block calls; caller zero-pads trailing scale_a slots).
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen64_r2c1blk4(
    const std::byte* a0, const std::byte* a1, size_t nblk,
    const std::byte* QuantBDataPtr,
    const float* scale_a0, const float* scale_a1, const float* scale_b,
    __m256& acc0, __m256& acc1)
{
    const __m256i glo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    const __m256i ghi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));
    const __m256i m03 = _mm256_set1_epi8(0x03);

    dot_one_block_w2_blklen64_r2<kVnni, 0>(glo, ghi, m03, a0, a1,
        scale_a0[0] * scale_b[0], scale_a1[0] * scale_b[0], acc0, acc1);
    if (nblk > 1) {
        dot_one_block_w2_blklen64_r2<kVnni, 2>(glo, ghi, m03, a0 + kBlkLen, a1 + kBlkLen,
            scale_a0[1] * scale_b[1], scale_a1[1] * scale_b[1], acc0, acc1);
    }
    if (nblk > 2) {
        dot_one_block_w2_blklen64_r2<kVnni, 4>(glo, ghi, m03, a0 + 2 * kBlkLen, a1 + 2 * kBlkLen,
            scale_a0[2] * scale_b[2], scale_a1[2] * scale_b[2], acc0, acc1);
    }
    if (nblk > 3) {
        dot_one_block_w2_blklen64_r2<kVnni, 6>(glo, ghi, m03, a0 + 3 * kBlkLen, a1 + 3 * kBlkLen,
            scale_a0[3] * scale_b[3], scale_a1[3] * scale_b[3], acc0, acc1);
    }
}

template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR1xC4BlkLen64Avx2(
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
    constexpr size_t PerColGroupBytes = kBlockGroupBytes;
    constexpr size_t PerColGroupScale = kBlockGroupBlks;
    constexpr size_t PerKGroupAdvanceBytes = kNCols4 * PerColGroupBytes;
    constexpr size_t PerKGroupAdvanceScale = kNCols4 * PerColGroupScale;
    const size_t BlockGroupCountKPadded = MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    const size_t GroupStrideBytes = BlockCountKPadded * kNCols4 * kBlkBytes;
    const size_t GroupStrideScale = BlockCountKPadded * kNCols4;

    assert(CountN % kNCols4 == 0);
    const size_t FullGroups = BlockCountK / kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % kBlockGroupBlks;

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

            __m256 acc[kNCols4] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps()
            };

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                const __m256i a0lo = load_a_half_blklen64(QuantAPtr + 0 * kBlkLen);
                const __m256i a0hi = load_a_half_blklen64(QuantAPtr + 0 * kBlkLen + 32);
                const __m256i a1lo = load_a_half_blklen64(QuantAPtr + 1 * kBlkLen);
                const __m256i a1hi = load_a_half_blklen64(QuantAPtr + 1 * kBlkLen + 32);
                const __m256i a2lo = load_a_half_blklen64(QuantAPtr + 2 * kBlkLen);
                const __m256i a2hi = load_a_half_blklen64(QuantAPtr + 2 * kBlkLen + 32);
                const __m256i a3lo = load_a_half_blklen64(QuantAPtr + 3 * kBlkLen);
                const __m256i a3hi = load_a_half_blklen64(QuantAPtr + 3 * kBlkLen + 32);

                for (size_t c = 0; c < kNCols4; ++c) {
                    accumulate_w2_blklen64_r1c1blk4<kVnni>(
                        a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, a3lo, a3hi,
                        QuantBDataPtr + c * PerColGroupBytes,
                        QuantAScalePtr, QuantBScalePtr + c * PerColGroupScale, acc[c]);
                }

                QuantAPtr += kBlkLen * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += PerKGroupAdvanceBytes;
                QuantBScalePtr += PerKGroupAdvanceScale;
            }

            if (TailBlocks > 0) {
                const __m256i zero = _mm256_setzero_si256();
                const __m256i a0lo = load_a_half_blklen64(QuantAPtr + 0 * kBlkLen);
                const __m256i a0hi = load_a_half_blklen64(QuantAPtr + 0 * kBlkLen + 32);
                const __m256i a1lo = (TailBlocks > 1) ? load_a_half_blklen64(QuantAPtr + 1 * kBlkLen) : zero;
                const __m256i a1hi = (TailBlocks > 1) ? load_a_half_blklen64(QuantAPtr + 1 * kBlkLen + 32) : zero;
                const __m256i a2lo = (TailBlocks > 2) ? load_a_half_blklen64(QuantAPtr + 2 * kBlkLen) : zero;
                const __m256i a2hi = (TailBlocks > 2) ? load_a_half_blklen64(QuantAPtr + 2 * kBlkLen + 32) : zero;

                float scale_a_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a_safe[i] = QuantAScalePtr[i];
                }

                for (size_t c = 0; c < kNCols4; ++c) {
                    accumulate_w2_blklen64_r1c1blk4<kVnni>(
                        a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, zero, zero,
                        QuantBDataPtr + c * PerColGroupBytes,
                        scale_a_safe, QuantBScalePtr + c * PerColGroupScale, acc[c]);
                }
            }

            SumPtr[0] = hsum_float_8(acc[0]);
            SumPtr[1] = hsum_float_8(acc[1]);
            SumPtr[2] = hsum_float_8(acc[2]);
            SumPtr[3] = hsum_float_8(acc[3]);
            if (BiasPtr != nullptr) {
                SumPtr[0] += BiasPtr[0];
                SumPtr[1] += BiasPtr[1];
                SumPtr[2] += BiasPtr[2];
                SumPtr[3] += BiasPtr[3];
            }

            QuantBDataColPtr += GroupStrideBytes;
            QuantBScaleColPtr += GroupStrideScale;
            if (BiasPtr != nullptr) {
                BiasPtr += kNCols4;
            }
            SumPtr += kNCols4;
        }
    }
}

//
// R2 x C4 tile -- the prefill hot path (CountM >= 2 even; the caller routes the
// odd trailing row through R1xC4). Sharing each column's block-group load +
// unpack across 2 M-rows measured 11-15% faster than R1 at prefill shapes on
// AVX2-VNNI client silicon; M=1 decode always takes the R1 tile.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR2xC4BlkLen64Avx2(
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
    constexpr size_t PerColGroupBytes = kBlockGroupBytes;
    constexpr size_t PerColGroupScale = kBlockGroupBlks;
    constexpr size_t PerKGroupAdvanceBytes = kNCols4 * PerColGroupBytes;
    constexpr size_t PerKGroupAdvanceScale = kNCols4 * PerColGroupScale;
    const size_t BlockGroupCountKPadded = MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    const size_t GroupStrideBytes = BlockCountKPadded * kNCols4 * kBlkBytes;
    const size_t GroupStrideScale = BlockCountKPadded * kNCols4;

    assert(CountM % kNRows2 == 0);
    assert(CountN % kNCols4 == 0);
    const size_t FullGroups = BlockCountK / kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % kBlockGroupBlks;

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

            __m256 acc[kNCols4 * kNRows2] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()
            };

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                for (size_t c = 0; c < kNCols4; ++c) {
                    accumulate_w2_blklen64_r2c1blk4<kVnni>(
                        QuantAPtr, QuantAPtr + lda, kBlockGroupBlks,
                        QuantBDataPtr + c * PerColGroupBytes,
                        QuantAScalePtr, QuantAScalePtr + BlockCountK,
                        QuantBScalePtr + c * PerColGroupScale,
                        acc[c], acc[kNCols4 + c]);
                }
                QuantAPtr += kBlkLen * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += PerKGroupAdvanceBytes;
                QuantBScalePtr += PerKGroupAdvanceScale;
            }

            // K-tail: 1-3 trailing real K-blocks; guarded per-block calls plus
            // bounded scale_a copies for both M-rows.
            if (TailBlocks > 0) {
                float scale_a0_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                float scale_a1_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a0_safe[i] = QuantAScalePtr[i];
                    scale_a1_safe[i] = QuantAScalePtr[BlockCountK + i];
                }
                for (size_t c = 0; c < kNCols4; ++c) {
                    accumulate_w2_blklen64_r2c1blk4<kVnni>(
                        QuantAPtr, QuantAPtr + lda, TailBlocks,
                        QuantBDataPtr + c * PerColGroupBytes,
                        scale_a0_safe, scale_a1_safe,
                        QuantBScalePtr + c * PerColGroupScale,
                        acc[c], acc[kNCols4 + c]);
                }
            }

            for (size_t c = 0; c < kNCols4; ++c) {
                SumPtr[c] = hsum_float_8(acc[c]);
                SumPtr[ldc + c] = hsum_float_8(acc[kNCols4 + c]);
                if (BiasPtr != nullptr) {
                    SumPtr[c] += BiasPtr[c];
                    SumPtr[ldc + c] += BiasPtr[c];
                }
            }

            QuantBDataColPtr += GroupStrideBytes;
            QuantBScaleColPtr += GroupStrideScale;
            if (BiasPtr != nullptr) {
                BiasPtr += kNCols4;
            }
            SumPtr += kNCols4;
        }
    }
}

template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmRMxC_Tail_BlkLen64Avx2(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBDataTail,
    const float* QuantBScaleTail,
    float* C,
    size_t CountM,
    size_t TailN,
    size_t BlockCountK,
    const float* BiasTail,
    size_t ldc)
{
    assert(TailN >= 1 && TailN <= 3);
    const size_t lda = BlockCountK * kBlkLen;
    const size_t BlockGroupCountKPadded = MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t FullGroups = BlockCountK / kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % kBlockGroupBlks;
    const size_t ColStrideBytes = BlockGroupCountKPadded * kBlockGroupBytes;
    const size_t ColStrideScale = BlockGroupCountKPadded * kBlockGroupBlks;

    for (size_t m = 0; m < CountM; ++m) {
        for (size_t c = 0; c < TailN; ++c) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;
            const std::byte* QuantBDataPtr = QuantBDataTail + c * ColStrideBytes;
            const float* QuantBScalePtr = QuantBScaleTail + c * ColStrideScale;

            __m256 acc = _mm256_setzero_ps();

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                accumulate_w2_blklen64_r1c1blk4<kVnni>(
                    load_a_half_blklen64(QuantAPtr + 0 * kBlkLen), load_a_half_blklen64(QuantAPtr + 0 * kBlkLen + 32),
                    load_a_half_blklen64(QuantAPtr + 1 * kBlkLen), load_a_half_blklen64(QuantAPtr + 1 * kBlkLen + 32),
                    load_a_half_blklen64(QuantAPtr + 2 * kBlkLen), load_a_half_blklen64(QuantAPtr + 2 * kBlkLen + 32),
                    load_a_half_blklen64(QuantAPtr + 3 * kBlkLen), load_a_half_blklen64(QuantAPtr + 3 * kBlkLen + 32),
                    QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc);

                QuantAPtr += kBlkLen * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += kBlockGroupBytes;
                QuantBScalePtr += kBlockGroupBlks;
            }

            if (TailBlocks > 0) {
                const __m256i zero = _mm256_setzero_si256();
                const __m256i a0lo = load_a_half_blklen64(QuantAPtr + 0 * kBlkLen);
                const __m256i a0hi = load_a_half_blklen64(QuantAPtr + 0 * kBlkLen + 32);
                const __m256i a1lo = (TailBlocks > 1) ? load_a_half_blklen64(QuantAPtr + 1 * kBlkLen) : zero;
                const __m256i a1hi = (TailBlocks > 1) ? load_a_half_blklen64(QuantAPtr + 1 * kBlkLen + 32) : zero;
                const __m256i a2lo = (TailBlocks > 2) ? load_a_half_blklen64(QuantAPtr + 2 * kBlkLen) : zero;
                const __m256i a2hi = (TailBlocks > 2) ? load_a_half_blklen64(QuantAPtr + 2 * kBlkLen + 32) : zero;

                float scale_a_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a_safe[i] = QuantAScalePtr[i];
                }
                accumulate_w2_blklen64_r1c1blk4<kVnni>(
                    a0lo, a0hi, a1lo, a1hi, a2lo, a2hi, zero, zero,
                    QuantBDataPtr, scale_a_safe, QuantBScalePtr, acc);
            }

            float v = hsum_float_8(acc);
            if (BiasTail != nullptr) v += BiasTail[c];
            C[m * ldc + c] = v;
        }
    }
}

//
// Top-level BlkLen=64 kernel body (full signature; the dispatcher's default
// case). Returns 0 unless BlkLen == 64 so the caller can fall back.
//
template <bool kVnni>
static MLAS_FORCEINLINE size_t
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen64_Impl(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum)
{
    if (BlkLen != kBlkLen) {
        return 0;
    }
    if (BlockCountK == 0 || CountM == 0 || CountN == 0) {
        return 0;
    }

    const size_t NMain = (CountN / kNCols4) * kNCols4;
    const size_t NTail = CountN - NMain;

    const size_t M_pairs = CountM / kNRows2;
    const size_t M_main = M_pairs * kNRows2;
    const size_t M_tail = CountM - M_main;
    const size_t lda = BlockCountK * kBlkLen;

    if (NMain > 0) {
        if (M_main > 0) {
            Q2Int8GemmR2xC4BlkLen64Avx2<kVnni>(
                QuantA, QuantAScale, QuantBData, QuantBScale,
                C, M_main, NMain, BlockCountK, Bias, ldc);
        }
        if (M_tail > 0) {
            Q2Int8GemmR1xC4BlkLen64Avx2<kVnni>(
                QuantA + M_main * lda,
                QuantAScale + M_main * BlockCountK,
                QuantBData, QuantBScale,
                C + M_main * ldc,
                /*CountM=*/M_tail,
                NMain, BlockCountK, Bias, ldc);
        }
    }

    if (NTail > 0) {
        const size_t BlockGroupCountKPadded = MlasDivRoundup(BlockCountK, kBlockGroupBlks);
        const std::byte* QuantBDataTail =
            QuantBData + NMain * BlockGroupCountKPadded * kBlockGroupBytes;
        const float* QuantBScaleTail =
            QuantBScale + NMain * BlockGroupCountKPadded * kBlockGroupBlks;
        const float* BiasTail = (Bias != nullptr) ? Bias + NMain : nullptr;

        Q2Int8GemmRMxC_Tail_BlkLen64Avx2<kVnni>(
            QuantA, QuantAScale, QuantBDataTail, QuantBScaleTail,
            C + NMain, CountM, NTail, BlockCountK, BiasTail, ldc);
    }

    float* c_blk = C;
    const float* b_blk_sum = QuantBBlkSum;
    size_t RowsRemaining = CountM;
    const float* a_blksum_row = ABlockSum;
    while (RowsRemaining > 0) {
        const auto RowsHandled = GetMlasPlatform().GemmFloatKernel(
            a_blksum_row, b_blk_sum, c_blk,
            BlockCountK, RowsRemaining, CountN, BlockCountK, ldc,
            1.0f, /*ZeroMode=*/false);

        c_blk += ldc * RowsHandled;
        a_blksum_row += BlockCountK * RowsHandled;
        RowsRemaining -= RowsHandled;
    }
    return CountM;
}

static MLAS_FORCEINLINE size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni(
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
    return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen64_Impl<true>(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale,
        C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

static MLAS_FORCEINLINE size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx2(
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
    return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen64_Impl<false>(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale,
        C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

}  // namespace sq2bit_avx2
}  // namespace mlas
}  // namespace onnxruntime
