/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512_2bit_blklen32.h

Abstract:

    AVX-512 (-VNNI) W2 kernel for BlkLen=32. Sibling of
    sqnbitgemm_kernel_avx512_2bit_blklen64.h: same R2xC4 tile, same SGEMM
    correction step, same N-tail handling, same K-tail handling. The only
    differences are:
      * Each K-block is 32 weights = 8 bytes packed (vs 64 = 16 for BlkLen=64).
      * Each block-group is 4 K-blocks = 32 bytes = 1 YMM (vs 64 bytes = 1 ZMM).
      * Unpack: 1 YMM load + 4 fixed shift/AND -> 4 YMMs of 32 unpacked
        weights each. We zero-extend those YMMs into ZMM lower halves so the
        downstream MAC can stay full-width `dpbusd` (which is what AVX-512
        provides). Wasted ZMM upper half is paid for by avoiding two separate
        narrow-MAC paths -- a wash in instruction count and simpler to verify.
      * MAC per K-block: 1 dpbusd (operating on 32 of 64 lanes; upper half
        zero-padded). Same 2-sub-accumulator FMA chain as BlkLen=64.

    Templated on `<bool kVnni>` like the BlkLen=64/128 siblings.

    Constraints:
      * BlkLen == 32 only.
      * BlockCountK has no alignment requirement: the main K-loop iterates
        full block-groups; a K-tail handler picks up 1-3 trailing K-blocks.

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

inline constexpr size_t kNRows2_BlkLen32 = 2;

//
// Cheap block-group unpack for BlkLen=32: 1 YMM load + 4 fixed shift/AND ->
// 4 ZMMs holding 32 active bytes each in the LOW half (upper half zero).
//
// Bit layout of each byte b of the 32-byte packed block-group (b in [0, 31]):
//   bits[0..1] = block_0.weight[b]
//   bits[2..3] = block_1.weight[b]
//   bits[4..5] = block_2.weight[b]
//   bits[6..7] = block_3.weight[b]
//
// Each output ZMM contains 32 unpacked weights in lanes [0, 31] and zeros in
// lanes [32, 63]. The downstream dpbusd works correctly on this layout because
// the A vectors we feed it are similarly zero-padded in the upper half.
//
static MLAS_FORCEINLINE void
load_unpack_w2_block_group_blklen32(
    const std::byte* packed,
    __m512i& bv0_zext, __m512i& bv1_zext, __m512i& bv2_zext, __m512i& bv3_zext)
{
    const __m256i group_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    const __m256i mask03 = _mm256_set1_epi8(0x03);

    const __m256i bv0_ymm = _mm256_and_si256(group_ymm, mask03);
    const __m256i bv1_ymm = _mm256_and_si256(_mm256_srli_epi16(group_ymm, 2), mask03);
    const __m256i bv2_ymm = _mm256_and_si256(_mm256_srli_epi16(group_ymm, 4), mask03);
    const __m256i bv3_ymm = _mm256_and_si256(_mm256_srli_epi16(group_ymm, 6), mask03);

    // Zero-extend each YMM into the lower half of a ZMM; upper half = 0.
    bv0_zext = _mm512_zextsi256_si512(bv0_ymm);
    bv1_zext = _mm512_zextsi256_si512(bv1_ymm);
    bv2_zext = _mm512_zextsi256_si512(bv2_ymm);
    bv3_zext = _mm512_zextsi256_si512(bv3_ymm);
}

//
// Load one BlkLen=32 A block (32 int8 bytes) zero-extended into a ZMM.
// The lower half holds the 32 active bytes; upper half is zero. This pairs
// with the zero-extended B vectors produced by load_unpack_w2_block_group_blklen32
// so dpbusd produces the correct int32 partial sums in the low half (and
// 0 in the high half, harmless to the subsequent FP reduction).
//
static MLAS_FORCEINLINE __m512i
load_a_blklen32_zext(const std::byte* a_block)
{
    const __m256i a_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_block));
    return _mm512_zextsi256_si512(a_ymm);
}

//
// Per single M-row, 4-K-block dot-and-accumulate for BlkLen=32. Each K-block
// is 32 lanes wide; dpbusd produces 16 int32 partial sums in the low half
// (upper half is zero, contributes nothing). Same 2-sub-accumulator FMA
// strategy as BlkLen=64: blocks {0,2} chain into acc_lo, blocks {1,3} chain
// into acc_hi.
//
// Math per K-block: acc += scale_a[blk] * scale_b[blk] * dot(av[blk], bv[blk])
//
template <bool kVnni>
static MLAS_FORCEINLINE void
dot_accumulate_4blk_w2_blklen32(
    const __m512i& av0, const __m512i& av1, const __m512i& av2, const __m512i& av3,
    const __m512i& bv0, const __m512i& bv1, const __m512i& bv2, const __m512i& bv3,
    const float* scale_a,
    const float* scale_b,
    __m512& acc)
{
    __m512i d0, d1, d2, d3;
    if constexpr (kVnni) {
        // dpbusd: 2nd operand (bv=unsigned [0,3]) × 3rd operand (av=signed int8); consistent with maddubs(bv, av) below
        d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv0, av0);
        d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv1, av1);
        d2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv2, av2);
        d3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv3, av3);
    } else {
        const __m512i ones = _mm512_set1_epi16(1);
        const __m512i t0 = _mm512_maddubs_epi16(bv0, av0);
        const __m512i t1 = _mm512_maddubs_epi16(bv1, av1);
        const __m512i t2 = _mm512_maddubs_epi16(bv2, av2);
        const __m512i t3 = _mm512_maddubs_epi16(bv3, av3);
        d0 = _mm512_madd_epi16(t0, ones);
        d1 = _mm512_madd_epi16(t1, ones);
        d2 = _mm512_madd_epi16(t2, ones);
        d3 = _mm512_madd_epi16(t3, ones);
    }

    const __m512 s0 = _mm512_set1_ps(scale_a[0] * scale_b[0]);
    const __m512 s1 = _mm512_set1_ps(scale_a[1] * scale_b[1]);
    const __m512 s2 = _mm512_set1_ps(scale_a[2] * scale_b[2]);
    const __m512 s3 = _mm512_set1_ps(scale_a[3] * scale_b[3]);

    __m512 acc_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(d0), s0);
    __m512 acc_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(d1), s1);
    acc_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d2), s2, acc_lo);
    acc_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d3), s3, acc_hi);
    acc = _mm512_add_ps(acc, _mm512_add_ps(acc_lo, acc_hi));
}

template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen32_r2c1blk4(
    const __m512i& av00, const __m512i& av01, const __m512i& av02, const __m512i& av03,
    const __m512i& av10, const __m512i& av11, const __m512i& av12, const __m512i& av13,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_a1,
    const float* scale_b,
    __m512& acc0,
    __m512& acc1)
{
    __m512i bv0, bv1, bv2, bv3;
    load_unpack_w2_block_group_blklen32(QuantBDataPtr, bv0, bv1, bv2, bv3);

    dot_accumulate_4blk_w2_blklen32<kVnni>(
        av00, av01, av02, av03, bv0, bv1, bv2, bv3, scale_a0, scale_b, acc0);
    dot_accumulate_4blk_w2_blklen32<kVnni>(
        av10, av11, av12, av13, bv0, bv1, bv2, bv3, scale_a1, scale_b, acc1);
}

template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen32_r1c1blk4(
    const __m512i& av00, const __m512i& av01, const __m512i& av02, const __m512i& av03,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_b,
    __m512& acc0)
{
    __m512i bv0, bv1, bv2, bv3;
    load_unpack_w2_block_group_blklen32(QuantBDataPtr, bv0, bv1, bv2, bv3);

    dot_accumulate_4blk_w2_blklen32<kVnni>(
        av00, av01, av02, av03, bv0, bv1, bv2, bv3, scale_a0, scale_b, acc0);
}

//
// R1 x C4 tile (M=1 decode or trailing odd row of R2xC4).
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR1xC4BlkLen32Avx512(
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
    constexpr size_t PerColGroupBytes = kBlockGroupBytes32;                  // 32 B per col per group
    constexpr size_t PerColGroupScale = kBlockGroupBlks;                     // 4 scales per col per group
    constexpr size_t PerKGroupAdvanceBytes = kNCols4 * PerColGroupBytes;     // 128 B per K-group iter
    constexpr size_t PerKGroupAdvanceScale = kNCols4 * PerColGroupScale;     // 16 scales per K-group iter
    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
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

            __m512 acc[kNCols4] = {
                _mm512_setzero_ps(), _mm512_setzero_ps(),
                _mm512_setzero_ps(), _mm512_setzero_ps()
            };

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                const __m512i av00 = load_a_blklen32_zext(QuantAPtr + 0 * kBlkLen32);
                const __m512i av01 = load_a_blklen32_zext(QuantAPtr + 1 * kBlkLen32);
                const __m512i av02 = load_a_blklen32_zext(QuantAPtr + 2 * kBlkLen32);
                const __m512i av03 = load_a_blklen32_zext(QuantAPtr + 3 * kBlkLen32);

                accumulate_w2_blklen32_r1c1blk4<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 0 * PerColGroupBytes,
                    QuantAScalePtr, QuantBScalePtr + 0 * PerColGroupScale, acc[0]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 1 * PerColGroupBytes,
                    QuantAScalePtr, QuantBScalePtr + 1 * PerColGroupScale, acc[1]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 2 * PerColGroupBytes,
                    QuantAScalePtr, QuantBScalePtr + 2 * PerColGroupScale, acc[2]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 3 * PerColGroupBytes,
                    QuantAScalePtr, QuantBScalePtr + 3 * PerColGroupScale, acc[3]);

                QuantAPtr += kBlkLen32 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += PerKGroupAdvanceBytes;
                QuantBScalePtr += PerKGroupAdvanceScale;
            }

            if (TailBlocks > 0) {
                const __m512i zero = _mm512_setzero_si512();
                auto load_a = [&](size_t k) -> __m512i {
                    return (k < TailBlocks)
                        ? load_a_blklen32_zext(QuantAPtr + k * kBlkLen32)
                        : zero;
                };
                const __m512i av00 = load_a(0);
                const __m512i av01 = load_a(1);
                const __m512i av02 = load_a(2);
                // TailBlocks = BlockCountK % kBlockGroupBlks ∈ [1,3], so block 3 is never a real tail
                // block — hardcode av03 = zero. The unpacked bv3 is still non-zero, but its
                // contribution is zeroed twice: by av03==0 here and scale_a0_safe[3]==0 below.
                const __m512i av03 = zero;

                float scale_a0_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a0_safe[i] = QuantAScalePtr[i];
                }

                accumulate_w2_blklen32_r1c1blk4<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 0 * PerColGroupBytes,
                    scale_a0_safe, QuantBScalePtr + 0 * PerColGroupScale, acc[0]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 1 * PerColGroupBytes,
                    scale_a0_safe, QuantBScalePtr + 1 * PerColGroupScale, acc[1]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 2 * PerColGroupBytes,
                    scale_a0_safe, QuantBScalePtr + 2 * PerColGroupScale, acc[2]);
                accumulate_w2_blklen32_r1c1blk4<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 3 * PerColGroupBytes,
                    scale_a0_safe, QuantBScalePtr + 3 * PerColGroupScale, acc[3]);
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
// R2 x C4 tile (CountM >= 2 even).
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR2xC4BlkLen32Avx512(
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
    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
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

            __m512 acc[kNCols4 * kNRows2_BlkLen32] = {
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
            };

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                const __m512i av00 = load_a_blklen32_zext(QuantAPtr + 0 * kBlkLen32);
                const __m512i av01 = load_a_blklen32_zext(QuantAPtr + 1 * kBlkLen32);
                const __m512i av02 = load_a_blklen32_zext(QuantAPtr + 2 * kBlkLen32);
                const __m512i av03 = load_a_blklen32_zext(QuantAPtr + 3 * kBlkLen32);
                const __m512i av10 = load_a_blklen32_zext(QuantAPtr + lda + 0 * kBlkLen32);
                const __m512i av11 = load_a_blklen32_zext(QuantAPtr + lda + 1 * kBlkLen32);
                const __m512i av12 = load_a_blklen32_zext(QuantAPtr + lda + 2 * kBlkLen32);
                const __m512i av13 = load_a_blklen32_zext(QuantAPtr + lda + 3 * kBlkLen32);

                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 0 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 0 * PerColGroupScale,
                    acc[0], acc[kNCols4 + 0]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 1 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 1 * PerColGroupScale,
                    acc[1], acc[kNCols4 + 1]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 2 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 2 * PerColGroupScale,
                    acc[2], acc[kNCols4 + 2]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 3 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 3 * PerColGroupScale,
                    acc[3], acc[kNCols4 + 3]);

                QuantAPtr += kBlkLen32 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += PerKGroupAdvanceBytes;
                QuantBScalePtr += PerKGroupAdvanceScale;
            }

            if (TailBlocks > 0) {
                const __m512i zero = _mm512_setzero_si512();
                auto load_a = [&](size_t row_off, size_t k) -> __m512i {
                    return (k < TailBlocks)
                        ? load_a_blklen32_zext(QuantAPtr + row_off + k * kBlkLen32)
                        : zero;
                };
                const __m512i av00 = load_a(0, 0);
                const __m512i av01 = load_a(0, 1);
                const __m512i av02 = load_a(0, 2);
                // TailBlocks = BlockCountK % kBlockGroupBlks ∈ [1,3], so block 3 is never a real tail
                // block — hardcode av03 = zero. The unpacked bv3 is still non-zero, but its
                // contribution is zeroed twice: by av03==0 here and scale_a0_safe[3]==0 below.
                const __m512i av03 = zero;
                const __m512i av10 = load_a(lda, 0);
                const __m512i av11 = load_a(lda, 1);
                const __m512i av12 = load_a(lda, 2);
                const __m512i av13 = zero;

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
                    QuantBScalePtr + 0 * PerColGroupScale,
                    acc[0], acc[kNCols4 + 0]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 1 * PerColGroupBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 1 * PerColGroupScale,
                    acc[1], acc[kNCols4 + 1]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 2 * PerColGroupBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 2 * PerColGroupScale,
                    acc[2], acc[kNCols4 + 2]);
                accumulate_w2_blklen32_r2c1blk4<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 3 * PerColGroupBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 3 * PerColGroupScale,
                    acc[3], acc[kNCols4 + 3]);
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
// N-tail tile (1-3 trailing N-cols when CountN is not a multiple of kNCols4).
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmRMxC_Tail_BlkLen32Avx512(
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
    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    const size_t FullGroups = BlockCountK / kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % kBlockGroupBlks;

    for (size_t m = 0; m < CountM; ++m) {
        const std::byte* a_row = QuantA + m * lda;
        const float* a_scale_row = QuantAScale + m * BlockCountK;
        float* c_row = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            __m512 acc = _mm512_setzero_ps();

            const std::byte* b_col = QuantBData + n * BlockCountKPadded * kBlkBytes32;
            const float* b_scale_col = QuantBScale + n * BlockCountKPadded;
            const std::byte* QuantAPtr = a_row;
            const float* QuantAScalePtr = a_scale_row;
            const std::byte* QuantBDataPtr = b_col;
            const float* QuantBScalePtr = b_scale_col;

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                const __m512i av00 = load_a_blklen32_zext(QuantAPtr + 0 * kBlkLen32);
                const __m512i av01 = load_a_blklen32_zext(QuantAPtr + 1 * kBlkLen32);
                const __m512i av02 = load_a_blklen32_zext(QuantAPtr + 2 * kBlkLen32);
                const __m512i av03 = load_a_blklen32_zext(QuantAPtr + 3 * kBlkLen32);

                accumulate_w2_blklen32_r1c1blk4<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc);

                QuantAPtr += kBlkLen32 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += kBlockGroupBytes32;
                QuantBScalePtr += kBlockGroupBlks;
            }

            if (TailBlocks > 0) {
                const __m512i zero = _mm512_setzero_si512();
                auto load_a = [&](size_t k) -> __m512i {
                    return (k < TailBlocks)
                        ? load_a_blklen32_zext(QuantAPtr + k * kBlkLen32)
                        : zero;
                };
                const __m512i av00 = load_a(0);
                const __m512i av01 = load_a(1);
                const __m512i av02 = load_a(2);
                // TailBlocks = BlockCountK % kBlockGroupBlks ∈ [1,3], so block 3 is never a real tail
                // block — hardcode av03 = zero. The unpacked bv3 is still non-zero, but its
                // contribution is zeroed twice: by av03==0 here and scale_a0_safe[3]==0 below.
                const __m512i av03 = zero;

                float scale_a0_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a0_safe[i] = QuantAScalePtr[i];
                }

                accumulate_w2_blklen32_r1c1blk4<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr, scale_a0_safe, QuantBScalePtr, acc);
            }

            const float sum = _mm512_reduce_add_ps(acc);
            c_row[n] = (Bias != nullptr) ? (sum + Bias[n]) : sum;
        }
    }
}

//
// Top-level BlkLen=32 kernel templated on <bool kVnni>. Same SGEMM BlkSum
// correction step as the BlkLen=64/128 kernels.
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
            Q2Int8GemmR2xC4BlkLen32Avx512<kVnni>(
                QuantA, QuantAScale, QuantBData, QuantBScale,
                C, M_main, NMain, BlockCountK, Bias, ldc);
        }
        if (M_tail > 0) {
            Q2Int8GemmR1xC4BlkLen32Avx512<kVnni>(
                QuantA + M_main * lda,
                QuantAScale + M_main * BlockCountK,
                QuantBData, QuantBScale,
                C + M_main * ldc,
                /*CountM=*/M_tail,
                NMain, BlockCountK, Bias, ldc);
        }
    }

    if (NTail > 0) {
        const size_t BlockGroupCountKPadded =
            MlasDivRoundup(BlockCountK, kBlockGroupBlks);
        const std::byte* QuantBDataTail =
            QuantBData + NMain * BlockGroupCountKPadded * kBlockGroupBytes32;
        const float* QuantBScaleTail =
            QuantBScale + NMain * BlockGroupCountKPadded * kBlockGroupBlks;
        const float* BiasTail = (Bias != nullptr) ? Bias + NMain : nullptr;

        Q2Int8GemmRMxC_Tail_BlkLen32Avx512<kVnni>(
            QuantA, QuantAScale,
            QuantBDataTail, QuantBScaleTail,
            C + NMain,
            CountM, NTail, BlockCountK, BiasTail, ldc);
    }

    // BlkSum correction (width-16 chunked layout, same as other BlkLens).
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
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Avx512Vnni(
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
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Avx512(
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

}  // namespace sq2bit_avx512
}  // namespace mlas
}  // namespace onnxruntime
