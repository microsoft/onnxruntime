/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512_2bit_blklen128.h

Abstract:

    AVX-512 (-VNNI) W2 kernel for BlkLen=128. Sibling of
    sqnbitgemm_kernel_avx512_2bit_blklen64.h: same R2xC4 tile, same SGEMM
    correction step, same N-tail handling, same K-tail handling. The only
    differences are:
      * Each K-block is 128 weights = 32 bytes packed instead of 64 = 16.
      * Each block-group is 4 K-blocks = 128 bytes = 2 ZMMs (vs 1 ZMM for
        BlkLen=64).
      * Unpack: 2 ZMM loads + 4 fixed shift/AND per load -> 4 PAIRS of ZMMs,
        each pair holding 128 weights of one K-block (low half + high half).
      * MAC per K-block: 2 dpbusds (low+high halves) instead of 1.

    Templated on `<bool kVnni>` like the BlkLen=64 sibling.

    Constraints:
      * BlkLen == 128 only (other BlkLens go through their own SIMD headers).
      * BlockCountK has no alignment requirement: the main loop iterates
        full block-groups; a K-tail handler picks up 1-3 trailing K-blocks.
      * Production: CountM and CountN handled via the same R2/R1 + N-tail
        split as the BlkLen=64 kernel.

    Register pressure note:
      * R2xC4 needs 8 acc + 16 A halves (preloaded per block-group iter) +
        8 B halves (loaded once per N-col within the iter) = 32 ZMMs at
        peak, exactly at the AVX-512 register count. We do NOT preload all
        16 A halves at once; instead we hold 16 A halves at the top of the
        block-group iter (one per K-block per M-row, low+high), let the
        compiler spill the FMA temporaries if needed, and rely on the L1
        being warm for any spilled A halves. This matches the strategy the
        BlkLen=64 R2xC4 tile uses (it preloads 8 A vecs and the compiler
        manages the rest).

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

inline constexpr size_t kNRows2_BlkLen128 = 2;  // R2 tile shape, same as BlkLen=64

//
// Cheap block-group unpack for BlkLen=128: 2x ZMM loads + 4x (fixed-shift + AND)
// per load = 8 unpacked half-ZMMs.
//
// Bit layout of each byte b of the 128-byte packed block-group (b in [0, 127]):
//   bits[0..1] = block_0.weight[b]
//   bits[2..3] = block_1.weight[b]
//   bits[4..5] = block_2.weight[b]
//   bits[6..7] = block_3.weight[b]
//
// For each K-block k in [0,3], the 128 unpacked weights are split into two
// 64-byte ZMMs: bv_k_lo holds weights[0..63], bv_k_hi holds weights[64..127].
//
static MLAS_FORCEINLINE void
load_unpack_w2_block_group_blklen128(
    const std::byte* packed,
    __m512i& bv0_lo, __m512i& bv1_lo, __m512i& bv2_lo, __m512i& bv3_lo,
    __m512i& bv0_hi, __m512i& bv1_hi, __m512i& bv2_hi, __m512i& bv3_hi)
{
    const __m512i group_lo = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(packed));
    const __m512i group_hi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(packed + 64));
    const __m512i mask03 = _mm512_set1_epi8(0x03);

    bv0_lo = _mm512_and_si512(group_lo, mask03);
    bv1_lo = _mm512_and_si512(_mm512_srli_epi16(group_lo, 2), mask03);
    bv2_lo = _mm512_and_si512(_mm512_srli_epi16(group_lo, 4), mask03);
    bv3_lo = _mm512_and_si512(_mm512_srli_epi16(group_lo, 6), mask03);

    bv0_hi = _mm512_and_si512(group_hi, mask03);
    bv1_hi = _mm512_and_si512(_mm512_srli_epi16(group_hi, 2), mask03);
    bv2_hi = _mm512_and_si512(_mm512_srli_epi16(group_hi, 4), mask03);
    bv3_hi = _mm512_and_si512(_mm512_srli_epi16(group_hi, 6), mask03);
}

//
// Per single M-row, 4-K-block dot-and-accumulate for BlkLen=128.
// Each K-block is 128 weights = 2 dpbusds (low half + high half) summed
// before being scaled and added to acc.
//
// Math per K-block:
//   int32 d = dpbusd(0, bv_lo, av_lo); d = dpbusd(d, bv_hi, av_hi);
//   acc += scale_a[blk] * scale_b[blk] * cvtepi32_ps(d)
//
// scale_a and scale_b each point to 4 consecutive floats (one per K-block).
//
// We use the same two-sub-accumulator trick as the BlkLen=64 kernel: lo
// accumulator gets blocks {0,2}, hi accumulator gets {1,3}, summed at end.
// Critical path is ~2 FMA latencies per chain (vs 4 if single-chained).
//
template <bool kVnni>
static MLAS_FORCEINLINE void
dot_accumulate_4blk_w2_blklen128(
    const __m512i& av0_lo, const __m512i& av0_hi,
    const __m512i& av1_lo, const __m512i& av1_hi,
    const __m512i& av2_lo, const __m512i& av2_hi,
    const __m512i& av3_lo, const __m512i& av3_hi,
    const __m512i& bv0_lo, const __m512i& bv0_hi,
    const __m512i& bv1_lo, const __m512i& bv1_hi,
    const __m512i& bv2_lo, const __m512i& bv2_hi,
    const __m512i& bv3_lo, const __m512i& bv3_hi,
    const float* scale_a,
    const float* scale_b,
    __m512& acc)
{
    __m512i d0, d1, d2, d3;
    if constexpr (kVnni) {
        // dpbusd: 2nd operand (bv=unsigned [0,3]) x 3rd operand (av=signed int8); consistent with maddubs(bv, av) below
        d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv0_lo, av0_lo);
        d0 = _mm512_dpbusd_epi32(d0, bv0_hi, av0_hi);
        d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv1_lo, av1_lo);
        d1 = _mm512_dpbusd_epi32(d1, bv1_hi, av1_hi);
        d2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv2_lo, av2_lo);
        d2 = _mm512_dpbusd_epi32(d2, bv2_hi, av2_hi);
        d3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv3_lo, av3_lo);
        d3 = _mm512_dpbusd_epi32(d3, bv3_hi, av3_hi);
    } else {
        // Non-VNNI: vpmaddubsw -> vpmaddwd chain, low and high halves summed
        // before adding into the K-block accumulator.
        const __m512i ones = _mm512_set1_epi16(1);
        const __m512i t0_lo = _mm512_maddubs_epi16(bv0_lo, av0_lo);
        const __m512i t0_hi = _mm512_maddubs_epi16(bv0_hi, av0_hi);
        const __m512i t1_lo = _mm512_maddubs_epi16(bv1_lo, av1_lo);
        const __m512i t1_hi = _mm512_maddubs_epi16(bv1_hi, av1_hi);
        const __m512i t2_lo = _mm512_maddubs_epi16(bv2_lo, av2_lo);
        const __m512i t2_hi = _mm512_maddubs_epi16(bv2_hi, av2_hi);
        const __m512i t3_lo = _mm512_maddubs_epi16(bv3_lo, av3_lo);
        const __m512i t3_hi = _mm512_maddubs_epi16(bv3_hi, av3_hi);
        d0 = _mm512_add_epi32(_mm512_madd_epi16(t0_lo, ones),
                              _mm512_madd_epi16(t0_hi, ones));
        d1 = _mm512_add_epi32(_mm512_madd_epi16(t1_lo, ones),
                              _mm512_madd_epi16(t1_hi, ones));
        d2 = _mm512_add_epi32(_mm512_madd_epi16(t2_lo, ones),
                              _mm512_madd_epi16(t2_hi, ones));
        d3 = _mm512_add_epi32(_mm512_madd_epi16(t3_lo, ones),
                              _mm512_madd_epi16(t3_hi, ones));
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

//
// 2 M-rows x 1 N-col x 4 K-blocks (one block-group) accumulator for BlkLen=128.
// The block-group B load + unpack is shared across the 2 M-rows.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen128_r2c1blk4(
    const __m512i& av00_lo, const __m512i& av00_hi,
    const __m512i& av01_lo, const __m512i& av01_hi,
    const __m512i& av02_lo, const __m512i& av02_hi,
    const __m512i& av03_lo, const __m512i& av03_hi,
    const __m512i& av10_lo, const __m512i& av10_hi,
    const __m512i& av11_lo, const __m512i& av11_hi,
    const __m512i& av12_lo, const __m512i& av12_hi,
    const __m512i& av13_lo, const __m512i& av13_hi,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_a1,
    const float* scale_b,
    __m512& acc0,
    __m512& acc1)
{
    __m512i bv0_lo, bv1_lo, bv2_lo, bv3_lo;
    __m512i bv0_hi, bv1_hi, bv2_hi, bv3_hi;
    load_unpack_w2_block_group_blklen128(QuantBDataPtr,
                                         bv0_lo, bv1_lo, bv2_lo, bv3_lo,
                                         bv0_hi, bv1_hi, bv2_hi, bv3_hi);

    dot_accumulate_4blk_w2_blklen128<kVnni>(
        av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
        bv0_lo, bv0_hi, bv1_lo, bv1_hi, bv2_lo, bv2_hi, bv3_lo, bv3_hi,
        scale_a0, scale_b, acc0);
    dot_accumulate_4blk_w2_blklen128<kVnni>(
        av10_lo, av10_hi, av11_lo, av11_hi, av12_lo, av12_hi, av13_lo, av13_hi,
        bv0_lo, bv0_hi, bv1_lo, bv1_hi, bv2_lo, bv2_hi, bv3_lo, bv3_hi,
        scale_a1, scale_b, acc1);
}

//
// 1 M-row x 1 N-col x 4 K-blocks (one block-group) accumulator for BlkLen=128.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen128_r1c1blk4(
    const __m512i& av00_lo, const __m512i& av00_hi,
    const __m512i& av01_lo, const __m512i& av01_hi,
    const __m512i& av02_lo, const __m512i& av02_hi,
    const __m512i& av03_lo, const __m512i& av03_hi,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_b,
    __m512& acc0)
{
    __m512i bv0_lo, bv1_lo, bv2_lo, bv3_lo;
    __m512i bv0_hi, bv1_hi, bv2_hi, bv3_hi;
    load_unpack_w2_block_group_blklen128(QuantBDataPtr,
                                         bv0_lo, bv1_lo, bv2_lo, bv3_lo,
                                         bv0_hi, bv1_hi, bv2_hi, bv3_hi);

    dot_accumulate_4blk_w2_blklen128<kVnni>(
        av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
        bv0_lo, bv0_hi, bv1_lo, bv1_hi, bv2_lo, bv2_hi, bv3_lo, bv3_hi,
        scale_a0, scale_b, acc0);
}

//
// R1 x C4 tile (M=1 decode or trailing odd row of R2xC4).
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR1xC4BlkLen128Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,                  // expected to be 1 (caller-enforced)
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc)
{
    const size_t lda = BlockCountK * kBlkLen128;
    constexpr size_t PerColGroupBytes = kBlockGroupBytes128;                  // 128 B per col per group
    constexpr size_t PerColGroupScale = kBlockGroupBlks;                      // 4 scales per col per group
    constexpr size_t PerKGroupAdvanceBytes = kNCols4 * PerColGroupBytes;      // 512 B per K-group iter
    constexpr size_t PerKGroupAdvanceScale = kNCols4 * PerColGroupScale;      // 16 scales per K-group iter
    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
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

            __m512 acc[kNCols4] = {
                _mm512_setzero_ps(), _mm512_setzero_ps(),
                _mm512_setzero_ps(), _mm512_setzero_ps()
            };

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                // Load 4 K-blocks of A, each split into low+high 64-byte halves.
                const __m512i av00_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 0 * kBlkLen128));
                const __m512i av00_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 0 * kBlkLen128 + 64));
                const __m512i av01_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 1 * kBlkLen128));
                const __m512i av01_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 1 * kBlkLen128 + 64));
                const __m512i av02_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen128));
                const __m512i av02_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen128 + 64));
                const __m512i av03_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 3 * kBlkLen128));
                const __m512i av03_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 3 * kBlkLen128 + 64));

                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    QuantBDataPtr + 0 * PerColGroupBytes,
                    QuantAScalePtr, QuantBScalePtr + 0 * PerColGroupScale, acc[0]);
                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    QuantBDataPtr + 1 * PerColGroupBytes,
                    QuantAScalePtr, QuantBScalePtr + 1 * PerColGroupScale, acc[1]);
                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    QuantBDataPtr + 2 * PerColGroupBytes,
                    QuantAScalePtr, QuantBScalePtr + 2 * PerColGroupScale, acc[2]);
                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    QuantBDataPtr + 3 * PerColGroupBytes,
                    QuantAScalePtr, QuantBScalePtr + 3 * PerColGroupScale, acc[3]);

                QuantAPtr += kBlkLen128 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += PerKGroupAdvanceBytes;
                QuantBScalePtr += PerKGroupAdvanceScale;
            }

            // K-tail: 1-3 trailing real K-blocks. Zero-fill missing A halves.
            if (TailBlocks > 0) {
                const __m512i zero = _mm512_setzero_si512();
                auto load_lo = [&](size_t k) -> __m512i {
                    return (k < TailBlocks)
                        ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(
                              QuantAPtr + k * kBlkLen128))
                        : zero;
                };
                auto load_hi = [&](size_t k) -> __m512i {
                    return (k < TailBlocks)
                        ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(
                              QuantAPtr + k * kBlkLen128 + 64))
                        : zero;
                };
                const __m512i av00_lo = load_lo(0);
                const __m512i av00_hi = load_hi(0);
                const __m512i av01_lo = load_lo(1);
                const __m512i av01_hi = load_hi(1);
                const __m512i av02_lo = load_lo(2);
                const __m512i av02_hi = load_hi(2);
                // TailBlocks = BlockCountK % kBlockGroupBlks in [1,3], so block 3 is never a real tail
                // block. Keep av03 hardcoded to zero. The unpacked bv3 can still be non-zero,
                // but its contribution is zeroed twice: by av03==0 and scale_a0_safe[3]==0.
                const __m512i av03_lo = zero;
                const __m512i av03_hi = zero;

                float scale_a0_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a0_safe[i] = QuantAScalePtr[i];
                }

                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    QuantBDataPtr + 0 * PerColGroupBytes,
                    scale_a0_safe, QuantBScalePtr + 0 * PerColGroupScale, acc[0]);
                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    QuantBDataPtr + 1 * PerColGroupBytes,
                    scale_a0_safe, QuantBScalePtr + 1 * PerColGroupScale, acc[1]);
                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    QuantBDataPtr + 2 * PerColGroupBytes,
                    scale_a0_safe, QuantBScalePtr + 2 * PerColGroupScale, acc[2]);
                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
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
// R2 x C4 tile (CountM >= 2 even). Mirrors the BlkLen=64 R2xC4 tile but with
// BlkLen=128-specific A loads (split into low+high halves) and the
// BlkLen=128 accumulator.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR2xC4BlkLen128Avx512(
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
    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    const size_t GroupStrideBytes = BlockCountKPadded * kNCols4 * kBlkBytes128;
    const size_t GroupStrideScale = BlockCountKPadded * kNCols4;

    assert(CountM % kNRows2_BlkLen128 == 0);
    assert(CountN % kNCols4 == 0);
    const size_t FullGroups = BlockCountK / kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % kBlockGroupBlks;

    for (size_t m = 0; m < CountM; m += kNRows2_BlkLen128) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        float* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += kNCols4) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m512 acc[kNCols4 * kNRows2_BlkLen128] = {
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
            };

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                // M-row 0: 4 K-blocks * 2 halves = 8 A vecs
                const __m512i av00_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 0 * kBlkLen128));
                const __m512i av00_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 0 * kBlkLen128 + 64));
                const __m512i av01_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 1 * kBlkLen128));
                const __m512i av01_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 1 * kBlkLen128 + 64));
                const __m512i av02_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen128));
                const __m512i av02_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen128 + 64));
                const __m512i av03_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 3 * kBlkLen128));
                const __m512i av03_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 3 * kBlkLen128 + 64));

                // M-row 1: 4 K-blocks * 2 halves = 8 A vecs
                const __m512i av10_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 0 * kBlkLen128));
                const __m512i av10_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 0 * kBlkLen128 + 64));
                const __m512i av11_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 1 * kBlkLen128));
                const __m512i av11_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 1 * kBlkLen128 + 64));
                const __m512i av12_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 2 * kBlkLen128));
                const __m512i av12_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 2 * kBlkLen128 + 64));
                const __m512i av13_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 3 * kBlkLen128));
                const __m512i av13_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 3 * kBlkLen128 + 64));

                accumulate_w2_blklen128_r2c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    av10_lo, av10_hi, av11_lo, av11_hi, av12_lo, av12_hi, av13_lo, av13_hi,
                    QuantBDataPtr + 0 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 0 * PerColGroupScale,
                    acc[0], acc[kNCols4 + 0]);
                accumulate_w2_blklen128_r2c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    av10_lo, av10_hi, av11_lo, av11_hi, av12_lo, av12_hi, av13_lo, av13_hi,
                    QuantBDataPtr + 1 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 1 * PerColGroupScale,
                    acc[1], acc[kNCols4 + 1]);
                accumulate_w2_blklen128_r2c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    av10_lo, av10_hi, av11_lo, av11_hi, av12_lo, av12_hi, av13_lo, av13_hi,
                    QuantBDataPtr + 2 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 2 * PerColGroupScale,
                    acc[2], acc[kNCols4 + 2]);
                accumulate_w2_blklen128_r2c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    av10_lo, av10_hi, av11_lo, av11_hi, av12_lo, av12_hi, av13_lo, av13_hi,
                    QuantBDataPtr + 3 * PerColGroupBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 3 * PerColGroupScale,
                    acc[3], acc[kNCols4 + 3]);

                QuantAPtr += kBlkLen128 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += PerKGroupAdvanceBytes;
                QuantBScalePtr += PerKGroupAdvanceScale;
            }

            // K-tail
            if (TailBlocks > 0) {
                const __m512i zero = _mm512_setzero_si512();
                auto load_a_lo = [&](size_t row_off, size_t k) -> __m512i {
                    return (k < TailBlocks)
                        ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(
                              QuantAPtr + row_off + k * kBlkLen128))
                        : zero;
                };
                auto load_a_hi = [&](size_t row_off, size_t k) -> __m512i {
                    return (k < TailBlocks)
                        ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(
                              QuantAPtr + row_off + k * kBlkLen128 + 64))
                        : zero;
                };
                const __m512i av00_lo = load_a_lo(0, 0);
                const __m512i av00_hi = load_a_hi(0, 0);
                const __m512i av01_lo = load_a_lo(0, 1);
                const __m512i av01_hi = load_a_hi(0, 1);
                const __m512i av02_lo = load_a_lo(0, 2);
                const __m512i av02_hi = load_a_hi(0, 2);
                // TailBlocks = BlockCountK % kBlockGroupBlks in [1,3], so block 3 is never a real tail
                // block. Keep av03 hardcoded to zero. The unpacked bv3 can still be non-zero,
                // but its contribution is zeroed twice: by av03==0 and scale_a0_safe[3]==0.
                const __m512i av03_lo = zero;
                const __m512i av03_hi = zero;

                const __m512i av10_lo = load_a_lo(lda, 0);
                const __m512i av10_hi = load_a_hi(lda, 0);
                const __m512i av11_lo = load_a_lo(lda, 1);
                const __m512i av11_hi = load_a_hi(lda, 1);
                const __m512i av12_lo = load_a_lo(lda, 2);
                const __m512i av12_hi = load_a_hi(lda, 2);
                const __m512i av13_lo = zero;
                const __m512i av13_hi = zero;

                float scale_a0_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                float scale_a1_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a0_safe[i] = QuantAScalePtr[i];
                    scale_a1_safe[i] = QuantAScalePtr[BlockCountK + i];
                }

                accumulate_w2_blklen128_r2c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    av10_lo, av10_hi, av11_lo, av11_hi, av12_lo, av12_hi, av13_lo, av13_hi,
                    QuantBDataPtr + 0 * PerColGroupBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 0 * PerColGroupScale,
                    acc[0], acc[kNCols4 + 0]);
                accumulate_w2_blklen128_r2c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    av10_lo, av10_hi, av11_lo, av11_hi, av12_lo, av12_hi, av13_lo, av13_hi,
                    QuantBDataPtr + 1 * PerColGroupBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 1 * PerColGroupScale,
                    acc[1], acc[kNCols4 + 1]);
                accumulate_w2_blklen128_r2c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    av10_lo, av10_hi, av11_lo, av11_hi, av12_lo, av12_hi, av13_lo, av13_hi,
                    QuantBDataPtr + 2 * PerColGroupBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 2 * PerColGroupScale,
                    acc[2], acc[kNCols4 + 2]);
                accumulate_w2_blklen128_r2c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    av10_lo, av10_hi, av11_lo, av11_hi, av12_lo, av12_hi, av13_lo, av13_hi,
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
// Column-major packed B; walks one column at a time using the R1xC4 helper
// but with CountN = 1 per iteration. Slower than the main tile but bounded
// to at most 3 N-cols per call.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmRMxC_Tail_BlkLen128Avx512(
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
    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    // Column-major in this tail region: each N-col is BlockCountKPadded *
    // kBlkBytes128 bytes for B and BlockCountKPadded floats for scales.
    const size_t FullGroups = BlockCountK / kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % kBlockGroupBlks;

    for (size_t m = 0; m < CountM; ++m) {
        const std::byte* a_row = QuantA + m * lda;
        const float* a_scale_row = QuantAScale + m * BlockCountK;
        float* c_row = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            __m512 acc = _mm512_setzero_ps();

            const std::byte* b_col = QuantBData + n * BlockCountKPadded * kBlkBytes128;
            const float* b_scale_col = QuantBScale + n * BlockCountKPadded;
            const std::byte* QuantAPtr = a_row;
            const float* QuantAScalePtr = a_scale_row;
            const std::byte* QuantBDataPtr = b_col;
            const float* QuantBScalePtr = b_scale_col;

            for (size_t sb = 0; sb < FullGroups; ++sb) {
                const __m512i av00_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 0 * kBlkLen128));
                const __m512i av00_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 0 * kBlkLen128 + 64));
                const __m512i av01_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 1 * kBlkLen128));
                const __m512i av01_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 1 * kBlkLen128 + 64));
                const __m512i av02_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen128));
                const __m512i av02_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen128 + 64));
                const __m512i av03_lo = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 3 * kBlkLen128));
                const __m512i av03_hi = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 3 * kBlkLen128 + 64));

                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc);

                QuantAPtr += kBlkLen128 * kBlockGroupBlks;
                QuantAScalePtr += kBlockGroupBlks;
                QuantBDataPtr += kBlockGroupBytes128;
                QuantBScalePtr += kBlockGroupBlks;
            }

            if (TailBlocks > 0) {
                const __m512i zero = _mm512_setzero_si512();
                auto load_lo = [&](size_t k) -> __m512i {
                    return (k < TailBlocks)
                        ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(
                              QuantAPtr + k * kBlkLen128))
                        : zero;
                };
                auto load_hi = [&](size_t k) -> __m512i {
                    return (k < TailBlocks)
                        ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(
                              QuantAPtr + k * kBlkLen128 + 64))
                        : zero;
                };
                const __m512i av00_lo = load_lo(0);
                const __m512i av00_hi = load_hi(0);
                const __m512i av01_lo = load_lo(1);
                const __m512i av01_hi = load_hi(1);
                const __m512i av02_lo = load_lo(2);
                const __m512i av02_hi = load_hi(2);
                // TailBlocks = BlockCountK % kBlockGroupBlks in [1,3], so block 3 is never a real tail
                // block. Keep av03 hardcoded to zero. The unpacked bv3 can still be non-zero,
                // but its contribution is zeroed twice: by av03==0 and scale_a0_safe[3]==0.
                const __m512i av03_lo = zero;
                const __m512i av03_hi = zero;

                float scale_a0_safe[kBlockGroupBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a0_safe[i] = QuantAScalePtr[i];
                }

                accumulate_w2_blklen128_r1c1blk4<kVnni>(
                    av00_lo, av00_hi, av01_lo, av01_hi, av02_lo, av02_hi, av03_lo, av03_hi,
                    QuantBDataPtr, scale_a0_safe, QuantBScalePtr, acc);
            }

            const float sum = _mm512_reduce_add_ps(acc);
            c_row[n] = (Bias != nullptr) ? (sum + Bias[n]) : sum;
        }
    }
}

//
// Top-level BlkLen=128 kernel templated on <bool kVnni>. Same SGEMM BlkSum
// correction step as the BlkLen=64 kernel. Same R2/R1/N-tail split.
//
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

    const size_t M_pairs = CountM / kNRows2_BlkLen128;
    const size_t M_main = M_pairs * kNRows2_BlkLen128;
    const size_t M_tail = CountM - M_main;
    const size_t lda = BlockCountK * kBlkLen128;

    if (NMain > 0) {
        if (M_main > 0) {
            Q2Int8GemmR2xC4BlkLen128Avx512<kVnni>(
                QuantA, QuantAScale, QuantBData, QuantBScale,
                C, M_main, NMain, BlockCountK, Bias, ldc);
        }
        if (M_tail > 0) {
            Q2Int8GemmR1xC4BlkLen128Avx512<kVnni>(
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
            QuantBData + NMain * BlockGroupCountKPadded * kBlockGroupBytes128;
        const float* QuantBScaleTail =
            QuantBScale + NMain * BlockGroupCountKPadded * kBlockGroupBlks;
        const float* BiasTail = (Bias != nullptr) ? Bias + NMain : nullptr;

        Q2Int8GemmRMxC_Tail_BlkLen128Avx512<kVnni>(
            QuantA, QuantAScale,
            QuantBDataTail, QuantBScaleTail,
            C + NMain,
            CountM, NTail, BlockCountK, BiasTail, ldc);
    }

    // BlkSum correction (width-16 chunked layout, same as BlkLen=64).
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

//
// Top-level VNNI variant. Compiled into AVX-512-VNNI sources only.
//
static MLAS_FORCEINLINE size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Avx512Vnni(
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

//
// Top-level non-VNNI variant.
//
static MLAS_FORCEINLINE size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Avx512(
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

}  // namespace sq2bit_avx512
}  // namespace mlas
}  // namespace onnxruntime
