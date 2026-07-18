/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2_2bit_blklen32.h

Abstract:

    AVX2 (-VNNI) W2 kernel for BlkLen=32, consuming the block-group packed
    layout (sqnbitgemm_kernel_avx512_2bit.h). The AVX-512 BlkLen=32 kernel
    zero-extends a 32-byte YMM group into a ZMM and runs a half-width dpbusd;
    here the whole computation stays 256-bit, so a 32-byte block-group is one
    YMM load + four fixed shift/mask pairs and each dpbusd uses all 8 int32
    lanes.

    Templated on `<bool kVnni>`: the VNNI variant uses `_mm256_dpbusds_avx_epi32`
    for the integer MAC; non-VNNI uses the `vpmaddubsw + vpmaddwd` chain. Both
    produce bit-identical results.

    Constraints match the AVX-512 sibling: BlkLen == 32; no alignment
    requirement on CountM, CountN, or BlockCountK.

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

using namespace sq2bit_avx512;  // shared block-group W2 constants and pack helpers

inline constexpr size_t kNRows2_BlkLen32 = 2;

//
// One YMM load + four fixed shift/AND -> 4 YMMs of 32 unsigned weights [0,3].
// Bit layout of each byte b of `packed`:
//   bits[0..1] = block_0.weight[b], ..., bits[6..7] = block_3.weight[b].
//
static MLAS_FORCEINLINE void
load_unpack_w2_block_group_blklen32(
    const std::byte* packed,
    __m256i& bv0, __m256i& bv1, __m256i& bv2, __m256i& bv3)
{
    const __m256i group = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    const __m256i mask03 = _mm256_set1_epi8(0x03);
    bv0 = _mm256_and_si256(group, mask03);
    bv1 = _mm256_and_si256(_mm256_srli_epi16(group, 2), mask03);
    bv2 = _mm256_and_si256(_mm256_srli_epi16(group, 4), mask03);
    bv3 = _mm256_and_si256(_mm256_srli_epi16(group, 6), mask03);
}

static MLAS_FORCEINLINE __m256i
load_a_blklen32(const std::byte* a_block)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_block));
}

//
// Per single M-row, 4-K-block dot-and-accumulate. Two interleaved
// sub-accumulators (blocks {0,2} and {1,3}) keep the per-cell FMA chain two
// deep. Math per K-block: acc += scale_a[blk] * scale_b[blk] * dot(av, bv).
//
template <bool kVnni>
static MLAS_FORCEINLINE void
dot_accumulate_4blk_w2_blklen32(
    const __m256i& av0, const __m256i& av1, const __m256i& av2, const __m256i& av3,
    const __m256i& bv0, const __m256i& bv1, const __m256i& bv2, const __m256i& bv3,
    const float* scale_a, const float* scale_b, __m256& acc)
{
    __m256i d0, d1, d2, d3;
    // GCC 11+ is needed for _mm256_dpbusds_avx_epi32; older toolchains fall
    // through to the vpmaddubsw+vpmaddwd path even on AVX-VNNI hardware.
#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (kVnni) {
        // dpbusds: 2nd operand (bv=unsigned [0,3]) x 3rd operand (av=signed int8); consistent with maddubs(bv, av)
        d0 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv0, av0);
        d1 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv1, av1);
        d2 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv2, av2);
        d3 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv3, av3);
    } else
#endif
    {
        const __m256i ones = _mm256_set1_epi16(1);
        d0 = _mm256_madd_epi16(_mm256_maddubs_epi16(bv0, av0), ones);
        d1 = _mm256_madd_epi16(_mm256_maddubs_epi16(bv1, av1), ones);
        d2 = _mm256_madd_epi16(_mm256_maddubs_epi16(bv2, av2), ones);
        d3 = _mm256_madd_epi16(_mm256_maddubs_epi16(bv3, av3), ones);
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

template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen32_r1c1blk4(
    const __m256i& av0, const __m256i& av1, const __m256i& av2, const __m256i& av3,
    const std::byte* QuantBDataPtr,
    const float* scale_a, const float* scale_b, __m256& acc)
{
    __m256i bv0, bv1, bv2, bv3;
    load_unpack_w2_block_group_blklen32(QuantBDataPtr, bv0, bv1, bv2, bv3);
    dot_accumulate_4blk_w2_blklen32<kVnni>(av0, av1, av2, av3, bv0, bv1, bv2, bv3, scale_a, scale_b, acc);
}

//
// 2 M-rows x 1 N-col x 4 K-blocks (one block-group) accumulator. The
// block-group B load + unpack is shared across the 2 M-rows.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen32_r2c1blk4(
    const __m256i& av00, const __m256i& av01, const __m256i& av02, const __m256i& av03,
    const __m256i& av10, const __m256i& av11, const __m256i& av12, const __m256i& av13,
    const std::byte* QuantBDataPtr,
    const float* scale_a0, const float* scale_a1, const float* scale_b,
    __m256& acc0, __m256& acc1)
{
    __m256i bv0, bv1, bv2, bv3;
    load_unpack_w2_block_group_blklen32(QuantBDataPtr, bv0, bv1, bv2, bv3);
    dot_accumulate_4blk_w2_blklen32<kVnni>(av00, av01, av02, av03, bv0, bv1, bv2, bv3, scale_a0, scale_b, acc0);
    dot_accumulate_4blk_w2_blklen32<kVnni>(av10, av11, av12, av13, bv0, bv1, bv2, bv3, scale_a1, scale_b, acc1);
}

//
// R1 x C4 tile. Serves M=1 decode and, looped over CountM, every M-row.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR1xC4BlkLen32Avx2(
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
    const size_t lda = BlockCountK * kBlkLen32;
    constexpr size_t PerColGroupBytes = kBlockGroupBytes32;
    constexpr size_t PerColGroupScale = kBlockGroupBlks;
    constexpr size_t PerKGroupAdvanceBytes = kNCols4 * PerColGroupBytes;
    constexpr size_t PerKGroupAdvanceScale = kNCols4 * PerColGroupScale;
    const size_t BlockGroupCountKPadded = MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    const size_t GroupStrideBytes = BlockCountKPadded * kNCols4 * kBlkBytes32;
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
                const __m256i av0 = load_a_blklen32(QuantAPtr + 0 * kBlkLen32);
                const __m256i av1 = load_a_blklen32(QuantAPtr + 1 * kBlkLen32);
                const __m256i av2 = load_a_blklen32(QuantAPtr + 2 * kBlkLen32);
                const __m256i av3 = load_a_blklen32(QuantAPtr + 3 * kBlkLen32);

                accumulate_w2_blklen32_r1c1blk4<kVnni>(av0, av1, av2, av3,
                    QuantBDataPtr + 0 * PerColGroupBytes, QuantAScalePtr,
                    QuantBScalePtr + 0 * PerColGroupScale, acc[0]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(av0, av1, av2, av3,
                    QuantBDataPtr + 1 * PerColGroupBytes, QuantAScalePtr,
                    QuantBScalePtr + 1 * PerColGroupScale, acc[1]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(av0, av1, av2, av3,
                    QuantBDataPtr + 2 * PerColGroupBytes, QuantAScalePtr,
                    QuantBScalePtr + 2 * PerColGroupScale, acc[2]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(av0, av1, av2, av3,
                    QuantBDataPtr + 3 * PerColGroupBytes, QuantAScalePtr,
                    QuantBScalePtr + 3 * PerColGroupScale, acc[3]);

                QuantAPtr += kBlkLen32 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += PerKGroupAdvanceBytes;
                QuantBScalePtr += PerKGroupAdvanceScale;
            }

            // K-tail: 1-3 trailing real K-blocks. Zero YMM for missing A blocks
            // and a bounded scale_a copy to avoid reading uninitialised scales.
            if (TailBlocks > 0) {
                const __m256i zero = _mm256_setzero_si256();
                const __m256i av0 = load_a_blklen32(QuantAPtr + 0 * kBlkLen32);
                const __m256i av1 = (TailBlocks > 1) ? load_a_blklen32(QuantAPtr + 1 * kBlkLen32) : zero;
                const __m256i av2 = (TailBlocks > 2) ? load_a_blklen32(QuantAPtr + 2 * kBlkLen32) : zero;
                const __m256i av3 = zero;

                float scale_a_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a_safe[i] = QuantAScalePtr[i];
                }

                accumulate_w2_blklen32_r1c1blk4<kVnni>(av0, av1, av2, av3,
                    QuantBDataPtr + 0 * PerColGroupBytes, scale_a_safe,
                    QuantBScalePtr + 0 * PerColGroupScale, acc[0]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(av0, av1, av2, av3,
                    QuantBDataPtr + 1 * PerColGroupBytes, scale_a_safe,
                    QuantBScalePtr + 1 * PerColGroupScale, acc[1]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(av0, av1, av2, av3,
                    QuantBDataPtr + 2 * PerColGroupBytes, scale_a_safe,
                    QuantBScalePtr + 2 * PerColGroupScale, acc[2]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(av0, av1, av2, av3,
                    QuantBDataPtr + 3 * PerColGroupBytes, scale_a_safe,
                    QuantBScalePtr + 3 * PerColGroupScale, acc[3]);
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
// unpack across 2 M-rows measured ~5% faster than R1 at prefill shapes on
// AVX2-VNNI client silicon; M=1 decode always takes the R1 tile.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR2xC4BlkLen32Avx2(
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
    const size_t lda = BlockCountK * kBlkLen32;
    constexpr size_t PerColGroupBytes = kBlockGroupBytes32;
    constexpr size_t PerColGroupScale = kBlockGroupBlks;
    constexpr size_t PerKGroupAdvanceBytes = kNCols4 * PerColGroupBytes;
    constexpr size_t PerKGroupAdvanceScale = kNCols4 * PerColGroupScale;
    const size_t BlockGroupCountKPadded = MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    const size_t GroupStrideBytes = BlockCountKPadded * kNCols4 * kBlkBytes32;
    const size_t GroupStrideScale = BlockCountKPadded * kNCols4;

    assert(CountM % kNRows2_BlkLen32 == 0);
    assert(CountN % kNCols4 == 0);
    const size_t FullGroups = BlockCountK / kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % kBlockGroupBlks;

    for (size_t m = 0; m < CountM; m += kNRows2_BlkLen32) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        float* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += kNCols4) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;
            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m256 acc[kNCols4 * kNRows2_BlkLen32] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()
            };

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                const __m256i av00 = load_a_blklen32(QuantAPtr + 0 * kBlkLen32);
                const __m256i av01 = load_a_blklen32(QuantAPtr + 1 * kBlkLen32);
                const __m256i av02 = load_a_blklen32(QuantAPtr + 2 * kBlkLen32);
                const __m256i av03 = load_a_blklen32(QuantAPtr + 3 * kBlkLen32);
                const __m256i av10 = load_a_blklen32(QuantAPtr + lda + 0 * kBlkLen32);
                const __m256i av11 = load_a_blklen32(QuantAPtr + lda + 1 * kBlkLen32);
                const __m256i av12 = load_a_blklen32(QuantAPtr + lda + 2 * kBlkLen32);
                const __m256i av13 = load_a_blklen32(QuantAPtr + lda + 3 * kBlkLen32);

                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 0 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 0 * PerColGroupScale, acc[0], acc[kNCols4 + 0]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 1 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 1 * PerColGroupScale, acc[1], acc[kNCols4 + 1]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 2 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 2 * PerColGroupScale, acc[2], acc[kNCols4 + 2]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 3 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 3 * PerColGroupScale, acc[3], acc[kNCols4 + 3]);

                QuantAPtr += kBlkLen32 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += PerKGroupAdvanceBytes;
                QuantBScalePtr += PerKGroupAdvanceScale;
            }

            // K-tail: 1-3 trailing real K-blocks; zero YMM for missing A blocks
            // and bounded scale_a copies for both M-rows.
            if (TailBlocks > 0) {
                const __m256i zero = _mm256_setzero_si256();
                const __m256i av00 = load_a_blklen32(QuantAPtr + 0 * kBlkLen32);
                const __m256i av01 = (TailBlocks > 1) ? load_a_blklen32(QuantAPtr + 1 * kBlkLen32) : zero;
                const __m256i av02 = (TailBlocks > 2) ? load_a_blklen32(QuantAPtr + 2 * kBlkLen32) : zero;
                const __m256i av03 = zero;
                const __m256i av10 = load_a_blklen32(QuantAPtr + lda + 0 * kBlkLen32);
                const __m256i av11 = (TailBlocks > 1) ? load_a_blklen32(QuantAPtr + lda + 1 * kBlkLen32) : zero;
                const __m256i av12 = (TailBlocks > 2) ? load_a_blklen32(QuantAPtr + lda + 2 * kBlkLen32) : zero;
                const __m256i av13 = zero;

                float scale_a0_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                float scale_a1_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a0_safe[i] = QuantAScalePtr[i];
                    scale_a1_safe[i] = QuantAScalePtr[BlockCountK + i];
                }

                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 0 * PerColGroupBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 0 * PerColGroupScale, acc[0], acc[kNCols4 + 0]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 1 * PerColGroupBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 1 * PerColGroupScale, acc[1], acc[kNCols4 + 1]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 2 * PerColGroupBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 2 * PerColGroupScale, acc[2], acc[kNCols4 + 2]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 3 * PerColGroupBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 3 * PerColGroupScale, acc[3], acc[kNCols4 + 3]);
            }

            SumPtr[0] = hsum_float_8(acc[0]);
            SumPtr[1] = hsum_float_8(acc[1]);
            SumPtr[2] = hsum_float_8(acc[2]);
            SumPtr[3] = hsum_float_8(acc[3]);
            SumPtr[ldc + 0] = hsum_float_8(acc[kNCols4 + 0]);
            SumPtr[ldc + 1] = hsum_float_8(acc[kNCols4 + 1]);
            SumPtr[ldc + 2] = hsum_float_8(acc[kNCols4 + 2]);
            SumPtr[ldc + 3] = hsum_float_8(acc[kNCols4 + 3]);
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
            if (BiasPtr != nullptr) {
                BiasPtr += kNCols4;
            }
            SumPtr += kNCols4;
        }
    }
}

//
// 1 M-row x 1 N-col N-tail tile for the trailing 1-3 N-cols. The tail region of
// the packed B buffer is column-major (see PackedQuantBOffsetBytes_W2 for
// n >= NMain), so this walks one column at a time.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmRMxC_Tail_BlkLen32Avx2(
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
    const size_t lda = BlockCountK * kBlkLen32;
    const size_t BlockGroupCountKPadded = MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t FullGroups = BlockCountK / kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % kBlockGroupBlks;
    const size_t ColStrideBytes = BlockGroupCountKPadded * kBlockGroupBytes32;
    const size_t ColStrideScale = BlockGroupCountKPadded * kBlockGroupBlks;

    for (size_t m = 0; m < CountM; ++m) {
        for (size_t c = 0; c < TailN; ++c) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;
            const std::byte* QuantBDataPtr = QuantBDataTail + c * ColStrideBytes;
            const float* QuantBScalePtr = QuantBScaleTail + c * ColStrideScale;

            __m256 acc = _mm256_setzero_ps();

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                const __m256i av0 = load_a_blklen32(QuantAPtr + 0 * kBlkLen32);
                const __m256i av1 = load_a_blklen32(QuantAPtr + 1 * kBlkLen32);
                const __m256i av2 = load_a_blklen32(QuantAPtr + 2 * kBlkLen32);
                const __m256i av3 = load_a_blklen32(QuantAPtr + 3 * kBlkLen32);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(av0, av1, av2, av3,
                    QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc);

                QuantAPtr += kBlkLen32 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += kBlockGroupBytes32;
                QuantBScalePtr += kBlockGroupBlks;
            }

            if (TailBlocks > 0) {
                const __m256i zero = _mm256_setzero_si256();
                const __m256i av0 = load_a_blklen32(QuantAPtr + 0 * kBlkLen32);
                const __m256i av1 = (TailBlocks > 1) ? load_a_blklen32(QuantAPtr + 1 * kBlkLen32) : zero;
                const __m256i av2 = (TailBlocks > 2) ? load_a_blklen32(QuantAPtr + 2 * kBlkLen32) : zero;
                const __m256i av3 = zero;

                float scale_a_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a_safe[i] = QuantAScalePtr[i];
                }
                accumulate_w2_blklen32_r1c1blk4<kVnni>(av0, av1, av2, av3,
                    QuantBDataPtr, scale_a_safe, QuantBScalePtr, acc);
            }

            float v = hsum_float_8(acc);
            if (BiasTail != nullptr) v += BiasTail[c];
            C[m * ldc + c] = v;
        }
    }
}

//
// Top-level BlkLen=32 kernel body. Runs the integer tiles over NMain (R2xC4
// head + R1xC4 odd-row tail) and the 1-3 col N-tail, then the shared SGEMM
// BlkSum correction.
//
template <bool kVnni>
static MLAS_FORCEINLINE size_t
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Impl(
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

    const size_t M_pairs = CountM / kNRows2_BlkLen32;
    const size_t M_main = M_pairs * kNRows2_BlkLen32;
    const size_t M_tail = CountM - M_main;
    const size_t lda = BlockCountK * kBlkLen32;

    if (NMain > 0) {
        if (M_main > 0) {
            Q2Int8GemmR2xC4BlkLen32Avx2<kVnni>(
                QuantA, QuantAScale, QuantBData, QuantBScale,
                C, M_main, NMain, BlockCountK, Bias, ldc);
        }
        if (M_tail > 0) {
            Q2Int8GemmR1xC4BlkLen32Avx2<kVnni>(
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
            QuantBData + NMain * BlockGroupCountKPadded * kBlockGroupBytes32;
        const float* QuantBScaleTail =
            QuantBScale + NMain * BlockGroupCountKPadded * kBlockGroupBlks;
        const float* BiasTail = (Bias != nullptr) ? Bias + NMain : nullptr;

        Q2Int8GemmRMxC_Tail_BlkLen32Avx2<kVnni>(
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
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Avx2Vnni(
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
    return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Impl<true>(
        QuantA, QuantAScale, QuantBData, QuantBScale,
        C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

static MLAS_FORCEINLINE size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Avx2(
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
    return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Impl<false>(
        QuantA, QuantAScale, QuantBData, QuantBScale,
        C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

}  // namespace sq2bit_avx2
}  // namespace mlas
}  // namespace onnxruntime
