/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2_2bit_blklen128.h

Abstract:

    AVX2 (-VNNI) W2 kernel for BlkLen=128, consuming the block-group packed
    layout (sqnbitgemm_kernel_avx512_2bit.h). A 128-element K-block is four YMM
    quarters on AVX2, so each block's dot is four chained VPDPBUSD. Preloading
    all activations would exceed the 16-YMM file, so the accumulator loads
    activations from a base pointer guarded by a valid-block count, which also
    folds in the K-tail handling.

    Templated on `<bool kVnni>`.

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

// One block's dot: four 32-lane quarters summed into one int32 vector. `Shift`
// selects the 2-bit field (0/2/4/6) for block {0,1,2,3}.
template <bool kVnni, int Shift>
static MLAS_FORCEINLINE __m256i
dot_one_block_w2_blklen128(
    const __m256i& g0, const __m256i& g1, const __m256i& g2, const __m256i& g3,
    const __m256i& m03, const std::byte* a_blk)
{
    const __m256i b0 = _mm256_and_si256(_mm256_srli_epi16(g0, Shift), m03);
    const __m256i b1 = _mm256_and_si256(_mm256_srli_epi16(g1, Shift), m03);
    const __m256i b2 = _mm256_and_si256(_mm256_srli_epi16(g2, Shift), m03);
    const __m256i b3 = _mm256_and_si256(_mm256_srli_epi16(g3, Shift), m03);
    const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_blk + 0));
    const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_blk + 32));
    const __m256i a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_blk + 64));
    const __m256i a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_blk + 96));
    // GCC 11+ is needed for _mm256_dpbusds_avx_epi32; older toolchains fall
    // through to the vpmaddubsw+vpmaddwd path even on AVX-VNNI hardware.
#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (kVnni) {
        __m256i d = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), b0, a0);
        d = _mm256_dpbusds_avx_epi32(d, b1, a1);
        d = _mm256_dpbusds_avx_epi32(d, b2, a2);
        d = _mm256_dpbusds_avx_epi32(d, b3, a3);
        return d;
    } else
#endif
    {
        const __m256i ones = _mm256_set1_epi16(1);
        __m256i d = _mm256_madd_epi16(_mm256_maddubs_epi16(b0, a0), ones);
        d = _mm256_add_epi32(d, _mm256_madd_epi16(_mm256_maddubs_epi16(b1, a1), ones));
        d = _mm256_add_epi32(d, _mm256_madd_epi16(_mm256_maddubs_epi16(b2, a2), ones));
        d = _mm256_add_epi32(d, _mm256_madd_epi16(_mm256_maddubs_epi16(b3, a3), ones));
        return d;
    }
}

//
// acc += sum_{k<nblk} scale_a[k]*scale_b[k] * dot128(a_base + k*128, block k).
// nblk is 4 for a full group or 1..3 for the K-tail; a_base blocks at k >= nblk
// are never loaded and their scale_a is 0.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen128_r1c1blk4(
    const std::byte* a_base, size_t nblk,
    const std::byte* QuantBDataPtr,
    const float* scale_a, const float* scale_b, __m256& acc)
{
    const __m256i g0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 0));
    const __m256i g1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));
    const __m256i g2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 64));
    const __m256i g3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 96));
    const __m256i m03 = _mm256_set1_epi8(0x03);
    const __m256i zeroi = _mm256_setzero_si256();

    const __m256i d0 = dot_one_block_w2_blklen128<kVnni, 0>(g0, g1, g2, g3, m03, a_base + 0 * kBlkLen128);
    const __m256i d1 =
        (nblk > 1) ? dot_one_block_w2_blklen128<kVnni, 2>(g0, g1, g2, g3, m03, a_base + 1 * kBlkLen128) : zeroi;
    const __m256i d2 =
        (nblk > 2) ? dot_one_block_w2_blklen128<kVnni, 4>(g0, g1, g2, g3, m03, a_base + 2 * kBlkLen128) : zeroi;
    const __m256i d3 =
        (nblk > 3) ? dot_one_block_w2_blklen128<kVnni, 6>(g0, g1, g2, g3, m03, a_base + 3 * kBlkLen128) : zeroi;

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

template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR1xC4BlkLen128Avx2(
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
    const size_t lda = BlockCountK * kBlkLen128;
    constexpr size_t PerColGroupBytes = kBlockGroupBytes128;
    constexpr size_t PerColGroupScale = kBlockGroupBlks;
    constexpr size_t PerKGroupAdvanceBytes = kNCols4 * PerColGroupBytes;
    constexpr size_t PerKGroupAdvanceScale = kNCols4 * PerColGroupScale;
    const size_t BlockGroupCountKPadded = MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    const size_t GroupStrideBytes = BlockCountKPadded * kNCols4 * kBlkBytes128;
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
                for (size_t c = 0; c < kNCols4; ++c) {
                    accumulate_w2_blklen128_r1c1blk4<kVnni>(
                        QuantAPtr, kBlockGroupBlks,
                        QuantBDataPtr + c * PerColGroupBytes,
                        QuantAScalePtr, QuantBScalePtr + c * PerColGroupScale, acc[c]);
                }
                QuantAPtr += kBlkLen128 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += PerKGroupAdvanceBytes;
                QuantBScalePtr += PerKGroupAdvanceScale;
            }

            if (TailBlocks > 0) {
                float scale_a_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) scale_a_safe[i] = QuantAScalePtr[i];
                for (size_t c = 0; c < kNCols4; ++c) {
                    accumulate_w2_blklen128_r1c1blk4<kVnni>(
                        QuantAPtr, TailBlocks,
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

template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmRMxC_Tail_BlkLen128Avx2(
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
    const size_t lda = BlockCountK * kBlkLen128;
    const size_t BlockGroupCountKPadded = MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t FullGroups = BlockCountK / kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % kBlockGroupBlks;
    const size_t ColStrideBytes = BlockGroupCountKPadded * kBlockGroupBytes128;
    const size_t ColStrideScale = BlockGroupCountKPadded * kBlockGroupBlks;

    for (size_t m = 0; m < CountM; ++m) {
        for (size_t c = 0; c < TailN; ++c) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;
            const std::byte* QuantBDataPtr = QuantBDataTail + c * ColStrideBytes;
            const float* QuantBScalePtr = QuantBScaleTail + c * ColStrideScale;

            __m256 acc = _mm256_setzero_ps();

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    QuantAPtr, kBlockGroupBlks, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc);
                QuantAPtr += kBlkLen128 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += kBlockGroupBytes128;
                QuantBScalePtr += kBlockGroupBlks;
            }

            if (TailBlocks > 0) {
                float scale_a_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) scale_a_safe[i] = QuantAScalePtr[i];
                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    QuantAPtr, TailBlocks, QuantBDataPtr, scale_a_safe, QuantBScalePtr, acc);
            }

            float v = hsum_float_8(acc);
            if (BiasTail != nullptr) v += BiasTail[c];
            C[m * ldc + c] = v;
        }
    }
}

template <bool kVnni>
static MLAS_FORCEINLINE size_t
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Impl(
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
    if (BlockCountK == 0 || CountM == 0 || CountN == 0) {
        return 0;
    }

    const size_t NMain = (CountN / kNCols4) * kNCols4;
    const size_t NTail = CountN - NMain;

    if (NMain > 0) {
        Q2Int8GemmR1xC4BlkLen128Avx2<kVnni>(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, NMain, BlockCountK, Bias, ldc);
    }

    if (NTail > 0) {
        const size_t BlockGroupCountKPadded = MlasDivRoundup(BlockCountK, kBlockGroupBlks);
        const std::byte* QuantBDataTail =
            QuantBData + NMain * BlockGroupCountKPadded * kBlockGroupBytes128;
        const float* QuantBScaleTail =
            QuantBScale + NMain * BlockGroupCountKPadded * kBlockGroupBlks;
        const float* BiasTail = (Bias != nullptr) ? Bias + NMain : nullptr;

        Q2Int8GemmRMxC_Tail_BlkLen128Avx2<kVnni>(
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
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Avx2Vnni(
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
    return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Impl<true>(
        QuantA, QuantAScale, QuantBData, QuantBScale,
        C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

static MLAS_FORCEINLINE size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Avx2(
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
    return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Impl<false>(
        QuantA, QuantAScale, QuantBData, QuantBScale,
        C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

}  // namespace sq2bit_avx2
}  // namespace mlas
}  // namespace onnxruntime
