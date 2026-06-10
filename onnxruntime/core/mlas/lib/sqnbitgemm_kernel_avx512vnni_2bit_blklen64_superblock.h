/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512vnni_2bit_blklen64_superblock.h

Abstract:

    EXPERIMENTAL AVX-512 (-VNNI) W2 kernel that consumes the super-block packed
    layout (sqnbitgemm_kernel_avx512_2bit_superblock.h). Replaces the existing
    per-K-block broadcast + variable-shift unpack with one ZMM load and four
    fixed-shift+mask pairs, halving the inner-loop unpack cost.

    Templated on `<bool kVnni>` like its sibling header
    (sqnbitgemm_kernel_avx512vnni_2bit_blklen64.h): VNNI variant uses
    `_mm512_dpbusd_epi32` for the integer MAC; non-VNNI uses the
    `vpmaddubsw + vpmaddwd` chain. Both produce bit-identical results.

    Constraints (Phase 3):
      * BlkLen == 64 only.
      * BlockCountK must be a multiple of kSuperBlockBlks (= 4).
      * CountM must be a multiple of 2 (R2 tile); CountN must be a multiple
        of 4 (C4 tile). Tail handling is provided by the same R2xC1/R1xC4/
        R1xC1 helpers as the existing W2 kernel via a downgrade path the
        dispatcher will pick when these constraints don't hold.

    Layout reference:
      * 64-byte super-block: byte b holds 2-bit weight b from each of 4
        consecutive K-blocks at bit positions {0..1, 2..3, 4..5, 6..7}.
      * In a tile slot, 4 N-cols of super-block live consecutively: each
        N-col's super starts at offset c * kSuperBlockBytes within the slot.

--*/

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <immintrin.h>

#include "mlasi.h"
#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx512_2bit.h"
#include "sqnbitgemm_kernel_avx512_2bit_superblock.h"

namespace onnxruntime {
namespace mlas {
namespace sq2bit_avx512_super {

inline constexpr size_t kNRows2 = 2;  // matches the existing W2 R2 tile shape

//
// Cheap super-block unpack: 1x ZMM load + 4x (fixed-shift + AND).
// Critical path ~4c (load + and / load + srli + and parallel chains).
//
// Bit layout of each byte b of `packed`:
//   bits[0..1] = block_0.weight[b]
//   bits[2..3] = block_1.weight[b]
//   bits[4..5] = block_2.weight[b]
//   bits[6..7] = block_3.weight[b]
//
// `_mm512_srli_epi16` shifts each 16-bit lane by N. For the LOW byte of a
// 16-bit lane, the shift pulls in bits from the adjacent HIGH byte; the
// subsequent AND with 0x03 discards those leaked bits. For the HIGH byte,
// zeros are shifted in from the top -- exactly what we want.
//
static MLAS_FORCEINLINE void
load_unpack_super_w2(const std::byte* packed,
                     __m512i& bv0_64_epi8,
                     __m512i& bv1_64_epi8,
                     __m512i& bv2_64_epi8,
                     __m512i& bv3_64_epi8)
{
    const __m512i super = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(packed));
    const __m512i mask03 = _mm512_set1_epi8(0x03);
    bv0_64_epi8 = _mm512_and_si512(super, mask03);
    bv1_64_epi8 = _mm512_and_si512(_mm512_srli_epi16(super, 2), mask03);
    bv2_64_epi8 = _mm512_and_si512(_mm512_srli_epi16(super, 4), mask03);
    bv3_64_epi8 = _mm512_and_si512(_mm512_srli_epi16(super, 6), mask03);
}

//
// Per single M-row, 4-K-block dot-and-accumulate. Each block produces its own
// uniformly-scaled FMA into a sub-accumulator; we keep two sub-accumulators
// (alternating per K-block) so the per-cell FMA dependency chain is two FMAs
// deep instead of four. The two sub-accumulators are summed into `acc` at the
// end with one extra vector add per super-block per cell.
//
// Math per K-block: acc += scale_a[blk] * scale_b[blk] * dot(av[blk], bv[blk])
//
// scale_a and scale_b each point to 4 consecutive floats in their packed
// buffers (one per K-block of the super).
//
// Critical path analysis (Zen5 / SKX):
//   * Single-chain (prior version):
//       FMA latency * 4 = ~16 cycles per super-block per cell.
//   * Two sub-accumulators (current):
//       FMA latency * 2 = ~8 cycles per super-block per cell + one vaddps.
//   This roughly halves the FP critical path; the integer dpbusd chain
//   (one per block) is independent and runs in parallel with the FMAs.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
dot_accumulate_4blk_w2_super(const __m512i& av0_64_epi8, const __m512i& av1_64_epi8,
                             const __m512i& av2_64_epi8, const __m512i& av3_64_epi8,
                             const __m512i& bv0_64_epi8, const __m512i& bv1_64_epi8,
                             const __m512i& bv2_64_epi8, const __m512i& bv3_64_epi8,
                             const float* scale_a,
                             const float* scale_b,
                             __m512& acc)
{
    __m512i d0, d1, d2, d3;
    if constexpr (kVnni) {
        d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv0_64_epi8, av0_64_epi8);
        d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv1_64_epi8, av1_64_epi8);
        d2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv2_64_epi8, av2_64_epi8);
        d3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bv3_64_epi8, av3_64_epi8);
    } else {
        // Non-VNNI: vpmaddubsw producing 32 lanes of int16, then vpmaddwd against
        // a ones-vector reduces pairs to 16 lanes of int32. Same final layout as
        // dpbusd, so the downstream cvt + FMA is identical.
        const __m512i ones = _mm512_set1_epi16(1);
        const __m512i t0 = _mm512_maddubs_epi16(bv0_64_epi8, av0_64_epi8);
        const __m512i t1 = _mm512_maddubs_epi16(bv1_64_epi8, av1_64_epi8);
        const __m512i t2 = _mm512_maddubs_epi16(bv2_64_epi8, av2_64_epi8);
        const __m512i t3 = _mm512_maddubs_epi16(bv3_64_epi8, av3_64_epi8);
        d0 = _mm512_madd_epi16(t0, ones);
        d1 = _mm512_madd_epi16(t1, ones);
        d2 = _mm512_madd_epi16(t2, ones);
        d3 = _mm512_madd_epi16(t3, ones);
    }

    // Pre-multiplied per-block scales (scalar broadcast, uniform across 16 lanes).
    const __m512 s0 = _mm512_set1_ps(scale_a[0] * scale_b[0]);
    const __m512 s1 = _mm512_set1_ps(scale_a[1] * scale_b[1]);
    const __m512 s2 = _mm512_set1_ps(scale_a[2] * scale_b[2]);
    const __m512 s3 = _mm512_set1_ps(scale_a[3] * scale_b[3]);

    // Two interleaved sub-accumulators: lo gets blocks {0, 2}, hi gets {1, 3}.
    // Each sub-accumulator chain is 2 FMAs deep (~8c) vs the 4-FMA single chain.
    __m512 acc_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0), s0, _mm512_setzero_ps());
    __m512 acc_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1), s1, _mm512_setzero_ps());
    acc_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d2), s2, acc_lo);
    acc_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d3), s3, acc_hi);
    acc = _mm512_add_ps(acc, _mm512_add_ps(acc_lo, acc_hi));
}

//
// 2 M-rows x 1 N-col x 4 K-blocks (one super-block) accumulator. The
// super-block B load + unpack is shared across the 2 M-rows.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen64_r2c1blk4_super(
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
    load_unpack_super_w2(QuantBDataPtr, bv0, bv1, bv2, bv3);

    dot_accumulate_4blk_w2_super<kVnni>(
        av00, av01, av02, av03, bv0, bv1, bv2, bv3, scale_a0, scale_b, acc0);
    dot_accumulate_4blk_w2_super<kVnni>(
        av10, av11, av12, av13, bv0, bv1, bv2, bv3, scale_a1, scale_b, acc1);
}

//
// 1 M-row x 1 N-col x 4 K-blocks (one super-block) accumulator. Used by the
// R1xC4 tile for M=1 decode and as the trailing odd-row handler of R2xC4
// when CountM is odd.
//
template <bool kVnni>
static MLAS_FORCEINLINE void
accumulate_w2_blklen64_r1c1blk4_super(
    const __m512i& av00, const __m512i& av01, const __m512i& av02, const __m512i& av03,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_b,
    __m512& acc0)
{
    __m512i bv0, bv1, bv2, bv3;
    load_unpack_super_w2(QuantBDataPtr, bv0, bv1, bv2, bv3);

    dot_accumulate_4blk_w2_super<kVnni>(
        av00, av01, av02, av03, bv0, bv1, bv2, bv3, scale_a0, scale_b, acc0);
}

//
// R1 x C4 tile -- the M=1 decode path and the trailing odd-row handler for
// the R2xC4 tile when CountM is odd. Identical N-tile structure as R2xC4
// (4 N-cols, super-block K stride) but processes a single M-row, so it uses
// half the registers (4 accumulators instead of 8) and half the MAC count
// per super-block iteration.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR1xC4BlkLen64Avx512_Super(
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
    const size_t lda = BlockCountK * kBlkLen;
    constexpr size_t PerColSuperBytes = kSuperBlockBytes;
    constexpr size_t PerColSuperScale = kSuperBlockBlks;
    constexpr size_t PerKSuperAdvanceBytes = kNCols4 * PerColSuperBytes;
    constexpr size_t PerKSuperAdvanceScale = kNCols4 * PerColSuperScale;
    // GroupStride uses the PADDED BlockCountK because the packed B layout
    // walks N-groups at intervals of `SuperBlockCountKPadded * kNCols4 *
    // kSuperBlockBytes` (see PackedQuantBOffsetBytes_W2_SuperBlock). When
    // BlockCountK is a multiple of kSuperBlockBlks (== 4) the padded and
    // logical strides are identical; when it isn't, the kernel must step
    // past the padded slots to land on the next N-group correctly.
    const size_t SuperBlockCountKPadded =
        MlasDivRoundup(BlockCountK, kSuperBlockBlks);
    const size_t BlockCountKPadded = SuperBlockCountKPadded * kSuperBlockBlks;
    const size_t GroupStrideBytes = BlockCountKPadded * kNCols4 * kBlkBytes;
    const size_t GroupStrideScale = BlockCountKPadded * kNCols4;

    assert(CountN % kNCols4 == 0);
    // BlockCountK no longer required to be a multiple of kSuperBlockBlks:
    // the main K-loop iterates full supers; an optional tail handler picks up
    // the trailing 1-3 K-blocks (padded weights and scales contribute 0).
    const size_t FullSupers = BlockCountK / kSuperBlockBlks;
    const size_t TailBlocks = BlockCountK % kSuperBlockBlks;  // 0, 1, 2, or 3

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

            for (size_t sb = 0; sb < FullSupers; ++sb) {
                const __m512i av00 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr));
                const __m512i av01 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + kBlkLen));
                const __m512i av02 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen));
                const __m512i av03 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 3 * kBlkLen));

                accumulate_w2_blklen64_r1c1blk4_super<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 0 * PerColSuperBytes,
                    QuantAScalePtr,
                    QuantBScalePtr + 0 * PerColSuperScale,
                    acc[0]);
                accumulate_w2_blklen64_r1c1blk4_super<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 1 * PerColSuperBytes,
                    QuantAScalePtr,
                    QuantBScalePtr + 1 * PerColSuperScale,
                    acc[1]);
                accumulate_w2_blklen64_r1c1blk4_super<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 2 * PerColSuperBytes,
                    QuantAScalePtr,
                    QuantBScalePtr + 2 * PerColSuperScale,
                    acc[2]);
                accumulate_w2_blklen64_r1c1blk4_super<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 3 * PerColSuperBytes,
                    QuantAScalePtr,
                    QuantBScalePtr + 3 * PerColSuperScale,
                    acc[3]);

                QuantAPtr += kBlkLen * kSuperBlockBlks;
                QuantAScalePtr += kSuperBlockBlks;
                QuantBDataPtr += PerKSuperAdvanceBytes;
                QuantBScalePtr += PerKSuperAdvanceScale;
            }

            // K-tail: 1-3 trailing real K-blocks. Pack helper zero-padded the
            // missing K-block slots in B and the corresponding scale slots,
            // so the 4-block accumulator can run safely on the packed buffer.
            //
            // Two safety concerns:
            //   * A bytes: don't load past row end -- use zero ZMM for missing slots.
            //   * A scales: don't read past the row's logical BlockCountK scales --
            //     uninitialised memory there can contain NaN, which propagates
            //     through `0 * NaN = NaN` in the scale fmadd. We materialise a
            //     local 4-float scale_a buffer with real scales in [0, TailBlocks)
            //     and 0.0 in the trailing slots.
            if (TailBlocks > 0) {
                const __m512i zero = _mm512_setzero_si512();
                const __m512i av00 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 0 * kBlkLen));
                const __m512i av01 = (TailBlocks > 1)
                    ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + 1 * kBlkLen))
                    : zero;
                const __m512i av02 = (TailBlocks > 2)
                    ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen))
                    : zero;
                const __m512i av03 = zero;  // TailBlocks at most 3

                // Bounded scale_a copy.
                float scale_a0_safe[kSuperBlockBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a0_safe[i] = QuantAScalePtr[i];
                }

                accumulate_w2_blklen64_r1c1blk4_super<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 0 * PerColSuperBytes,
                    scale_a0_safe,
                    QuantBScalePtr + 0 * PerColSuperScale,
                    acc[0]);
                accumulate_w2_blklen64_r1c1blk4_super<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 1 * PerColSuperBytes,
                    scale_a0_safe,
                    QuantBScalePtr + 1 * PerColSuperScale,
                    acc[1]);
                accumulate_w2_blklen64_r1c1blk4_super<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 2 * PerColSuperBytes,
                    scale_a0_safe,
                    QuantBScalePtr + 2 * PerColSuperScale,
                    acc[2]);
                accumulate_w2_blklen64_r1c1blk4_super<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr + 3 * PerColSuperBytes,
                    scale_a0_safe,
                    QuantBScalePtr + 3 * PerColSuperScale,
                    acc[3]);
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
// R2 x C4 tile -- the main hot path for prefill (M >= 2). Iterates the K
// dimension in super-block strides of kSuperBlockBlks (= 4) K-blocks at a
// time. Assumes BlockCountK is a multiple of kSuperBlockBlks; the dispatcher
// must verify this before selecting the super-block kernel.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmR2xC4BlkLen64Avx512_Super(
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
    constexpr size_t PerColSuperBytes = kSuperBlockBytes;                       // 64 B per col per super
    constexpr size_t PerColSuperScale = kSuperBlockBlks;                        // 4 scales per col per super
    constexpr size_t PerKSuperAdvanceBytes = kNCols4 * PerColSuperBytes;        // 256 B per K-super iter
    constexpr size_t PerKSuperAdvanceScale = kNCols4 * PerColSuperScale;        // 16 scales per K-super iter
    // GroupStride uses the PADDED BlockCountK because the packed B layout
    // walks N-groups at intervals of `SuperBlockCountKPadded * kNCols4 *
    // kSuperBlockBytes` (see PackedQuantBOffsetBytes_W2_SuperBlock). When
    // BlockCountK is a multiple of kSuperBlockBlks (== 4) the padded and
    // logical strides are identical; when it isn't, the kernel must step
    // past the padded slots to land on the next N-group correctly.
    const size_t SuperBlockCountKPadded =
        MlasDivRoundup(BlockCountK, kSuperBlockBlks);
    const size_t BlockCountKPadded = SuperBlockCountKPadded * kSuperBlockBlks;
    const size_t GroupStrideBytes = BlockCountKPadded * kNCols4 * kBlkBytes;
    const size_t GroupStrideScale = BlockCountKPadded * kNCols4;

    assert(CountM % kNRows2 == 0);
    assert(CountN % kNCols4 == 0);
    // BlockCountK no longer required to be a multiple of kSuperBlockBlks:
    // the main K-loop iterates full supers; an optional tail handler picks up
    // the trailing 1-3 K-blocks (padded weights and scales contribute 0).
    const size_t FullSupers = BlockCountK / kSuperBlockBlks;
    const size_t TailBlocks = BlockCountK % kSuperBlockBlks;  // 0, 1, 2, or 3

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

            for (size_t sb = 0; sb < FullSupers; ++sb) {
                const __m512i av00 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr));
                const __m512i av01 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + kBlkLen));
                const __m512i av02 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen));
                const __m512i av03 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 3 * kBlkLen));
                const __m512i av10 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda));
                const __m512i av11 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + kBlkLen));
                const __m512i av12 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 2 * kBlkLen));
                const __m512i av13 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 3 * kBlkLen));

                accumulate_w2_blklen64_r2c1blk4_super<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 0 * PerColSuperBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 0 * PerColSuperScale,
                    acc[0], acc[kNCols4 + 0]);
                accumulate_w2_blklen64_r2c1blk4_super<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 1 * PerColSuperBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 1 * PerColSuperScale,
                    acc[1], acc[kNCols4 + 1]);
                accumulate_w2_blklen64_r2c1blk4_super<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 2 * PerColSuperBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 2 * PerColSuperScale,
                    acc[2], acc[kNCols4 + 2]);
                accumulate_w2_blklen64_r2c1blk4_super<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 3 * PerColSuperBytes,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK,
                    QuantBScalePtr + 3 * PerColSuperScale,
                    acc[3], acc[kNCols4 + 3]);

                QuantAPtr += kBlkLen * kSuperBlockBlks;
                QuantAScalePtr += kSuperBlockBlks;
                QuantBDataPtr += PerKSuperAdvanceBytes;
                QuantBScalePtr += PerKSuperAdvanceScale;
            }

            // K-tail: 1-3 trailing real K-blocks. See R1 tile comment above.
            if (TailBlocks > 0) {
                const __m512i zero = _mm512_setzero_si512();
                const __m512i av00 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 0 * kBlkLen));
                const __m512i av01 = (TailBlocks > 1)
                    ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + 1 * kBlkLen))
                    : zero;
                const __m512i av02 = (TailBlocks > 2)
                    ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen))
                    : zero;
                const __m512i av03 = zero;
                const __m512i av10 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + lda + 0 * kBlkLen));
                const __m512i av11 = (TailBlocks > 1)
                    ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + lda + 1 * kBlkLen))
                    : zero;
                const __m512i av12 = (TailBlocks > 2)
                    ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + lda + 2 * kBlkLen))
                    : zero;
                const __m512i av13 = zero;

                // Bounded scale_a copies for both M-rows (see R1 K-tail comment).
                float scale_a0_safe[kSuperBlockBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                float scale_a1_safe[kSuperBlockBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a0_safe[i] = QuantAScalePtr[i];
                    scale_a1_safe[i] = QuantAScalePtr[BlockCountK + i];
                }

                accumulate_w2_blklen64_r2c1blk4_super<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 0 * PerColSuperBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 0 * PerColSuperScale,
                    acc[0], acc[kNCols4 + 0]);
                accumulate_w2_blklen64_r2c1blk4_super<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 1 * PerColSuperBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 1 * PerColSuperScale,
                    acc[1], acc[kNCols4 + 1]);
                accumulate_w2_blklen64_r2c1blk4_super<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 2 * PerColSuperBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 2 * PerColSuperScale,
                    acc[2], acc[kNCols4 + 2]);
                accumulate_w2_blklen64_r2c1blk4_super<kVnni>(
                    av00, av01, av02, av03, av10, av11, av12, av13,
                    QuantBDataPtr + 3 * PerColSuperBytes,
                    scale_a0_safe, scale_a1_safe,
                    QuantBScalePtr + 3 * PerColSuperScale,
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
// 1 M-row x 1 N-col N-tail tile. Handles the 1-3 trailing N-cols when
// CountN is not a multiple of kNCols4. The tail region of the packed B
// buffer is column-major (one super-block per K-super per N-col, see
// PackedQuantBOffsetBytes_W2_SuperBlock for n >= NMain), so this tile
// walks one column at a time and reuses the same accumulator helper used
// by R1xC4 (`accumulate_w2_blklen64_r1c1blk4_super`). Slower than the
// R2xC4 main tile, but it processes at most 3 N-cols per call -- a
// trivial fraction of total work even on the worst-case shape.
//
// Pointer convention (caller-supplied bases):
//   QuantBDataTail  : start of the tail region in packed B -- exactly
//                     NMain * SuperBlockCountKPadded * kSuperBlockBytes
//                     bytes past PackedQuantBData (see callsite below).
//   QuantBScaleTail : same convention for the scale buffer (NMain *
//                     SuperBlockCountKPadded * kSuperBlockBlks floats).
//
// K-tail handling: identical to R1xC4 -- conditional A loads for the 1-3
// trailing real K-blocks and a bounded scale_a copy to avoid NaN from
// uninitialised QuantAScale slots.
//
template <bool kVnni>
MLAS_FORCEINLINE void
Q2Int8GemmRMxC_Tail_BlkLen64Avx512_Super(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBDataTail,
    const float* QuantBScaleTail,
    float* C,
    size_t CountM,
    size_t TailN,           // 1, 2, or 3
    size_t BlockCountK,
    const float* BiasTail,  // null OR points at the first tail N-col bias
    size_t ldc)
{
    assert(TailN >= 1 && TailN <= 3);
    constexpr size_t PerColSuperBytes = kSuperBlockBytes;     // 64 B per col per super
    constexpr size_t PerColSuperScale = kSuperBlockBlks;      // 4 scales per col per super

    const size_t lda = BlockCountK * kBlkLen;
    const size_t SuperBlockCountKPadded =
        MlasDivRoundup(BlockCountK, kSuperBlockBlks);
    const size_t FullSupers = BlockCountK / kSuperBlockBlks;
    const size_t TailBlocks = BlockCountK % kSuperBlockBlks;  // 0, 1, 2, or 3
    // In the tail region each N-col occupies SuperBlockCountKPadded
    // super-blocks back-to-back (column-major).
    const size_t ColStrideBytes = SuperBlockCountKPadded * PerColSuperBytes;
    const size_t ColStrideScale = SuperBlockCountKPadded * PerColSuperScale;

    for (size_t m = 0; m < CountM; ++m) {
        for (size_t c = 0; c < TailN; ++c) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataTail + c * ColStrideBytes;
            const float* QuantBScalePtr = QuantBScaleTail + c * ColStrideScale;

            __m512 acc = _mm512_setzero_ps();

            for (size_t sb = 0; sb < FullSupers; ++sb) {
                const __m512i av00 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr));
                const __m512i av01 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + kBlkLen));
                const __m512i av02 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen));
                const __m512i av03 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 3 * kBlkLen));

                accumulate_w2_blklen64_r1c1blk4_super<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr,
                    QuantAScalePtr,
                    QuantBScalePtr,
                    acc);

                QuantAPtr += kBlkLen * kSuperBlockBlks;
                QuantAScalePtr += kSuperBlockBlks;
                QuantBDataPtr += PerColSuperBytes;
                QuantBScalePtr += PerColSuperScale;
            }

            if (TailBlocks > 0) {
                const __m512i zero = _mm512_setzero_si512();
                const __m512i av00 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(QuantAPtr + 0 * kBlkLen));
                const __m512i av01 = (TailBlocks > 1)
                    ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + 1 * kBlkLen))
                    : zero;
                const __m512i av02 = (TailBlocks > 2)
                    ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantAPtr + 2 * kBlkLen))
                    : zero;
                const __m512i av03 = zero;

                float scale_a0_safe[kSuperBlockBlks] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    scale_a0_safe[i] = QuantAScalePtr[i];
                }

                accumulate_w2_blklen64_r1c1blk4_super<kVnni>(
                    av00, av01, av02, av03,
                    QuantBDataPtr,
                    scale_a0_safe,
                    QuantBScalePtr,
                    acc);
            }

            float* SumPtr = C + m * ldc + c;
            float v = _mm512_reduce_add_ps(acc);
            if (BiasTail != nullptr) v += BiasTail[c];
            *SumPtr = v;
        }
    }
}

//
// Top-level dispatched-kernel body. Mirrors the production W2 kernel's
// `SQ2BitGemmKernel_BlkSum_CompInt8_Impl` and the helper-mediated SGEMM
// correction step. Templated on `<bool kVnni>`; the AVX-512 and AVX-512-VNNI
// .cpp files instantiate it via test-entry forwarders.
//
// Restrictions enforced here:
//   * BlkLen must equal kBlkLen (64). Otherwise returns 0 to signal "did not
//     handle these rows" (the dispatcher will fall back).
//   * CountM has no alignment requirement: the R2xC4 tile handles the
//     M-aligned head and a single R1xC4 invocation picks up any trailing
//     odd row. CountM == 1 (decode) lands directly on the R1 path.
//   * CountN has no alignment requirement: the R2/R1 tiles process
//     NMain = floor(CountN/4)*4 cols against the 4-N-col-grouped packed
//     layout; a per-1-col tail tile picks up the trailing 1-3 cols against
//     the column-major tail region of the same packed buffer.
//   * BlockCountK has no alignment requirement: the R2/R1 tiles and the
//     N-tail tile each run a partial-super K-tail handler that loads only
//     the real trailing K-blocks (zero ZMM for missing slots, bounded
//     scale_a copy) and lets the pre-zeroed packed-B / scale slots
//     contribute 0 to the dot product.
//
template <bool kVnni>
static MLAS_FORCEINLINE size_t
SQ2BitGemmKernel_BlkSum_CompInt8_Super_Impl(
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
    if (BlockCountK == 0 || CountM == 0 || CountN == 0) {
        return 0;
    }

    // Split CountN into a 4-aligned head (NMain) handled by the R2xC4/R1xC4
    // tiles and a 1-3-col tail handled by Q2Int8GemmRMxC_Tail.
    const size_t NMain = (CountN / kNCols4) * kNCols4;
    const size_t NTail = CountN - NMain;

    // Split CountM into an R2 head and an optional R1 tail row.
    const size_t M_pairs = CountM / kNRows2;
    const size_t M_main = M_pairs * kNRows2;
    const size_t M_tail = CountM - M_main;
    const size_t lda = BlockCountK * kBlkLen;

    if (NMain > 0) {
        if (M_main > 0) {
            Q2Int8GemmR2xC4BlkLen64Avx512_Super<kVnni>(
                QuantA, QuantAScale, QuantBData, QuantBScale,
                C, M_main, NMain, BlockCountK, Bias, ldc);
        }
        if (M_tail > 0) {
            // R1 picks up the single trailing row. Pointers advance past the M_main
            // rows the R2 tile already consumed: A advances M_main*lda bytes, A-scale
            // advances M_main*BlockCountK floats, C advances M_main*ldc floats. The
            // packed-B buffer is reused (column-major over N).
            Q2Int8GemmR1xC4BlkLen64Avx512_Super<kVnni>(
                QuantA + M_main * lda,
                QuantAScale + M_main * BlockCountK,
                QuantBData, QuantBScale,
                C + M_main * ldc,
                /*CountM=*/M_tail,
                NMain, BlockCountK, Bias, ldc);
        }
    }

    if (NTail > 0) {
        // The tail region of the packed B buffer is column-major and starts
        // immediately after the NMain-cols grouped region:
        //   tail_base_bytes  = NMain * SuperBlockCountKPadded * kSuperBlockBytes
        //   tail_base_scales = NMain * SuperBlockCountKPadded * kSuperBlockBlks
        const size_t SuperBlockCountKPadded =
            MlasDivRoundup(BlockCountK, kSuperBlockBlks);
        const std::byte* QuantBDataTail =
            QuantBData + NMain * SuperBlockCountKPadded * kSuperBlockBytes;
        const float* QuantBScaleTail =
            QuantBScale + NMain * SuperBlockCountKPadded * kSuperBlockBlks;
        const float* BiasTail = (Bias != nullptr) ? Bias + NMain : nullptr;

        Q2Int8GemmRMxC_Tail_BlkLen64Avx512_Super<kVnni>(
            QuantA, QuantAScale,
            QuantBDataTail, QuantBScaleTail,
            C + NMain,
            CountM, NTail, BlockCountK, BiasTail, ldc);
    }

    // BlkSum correction: same width-16 chunked layout as the existing W2
    // kernel, so we reuse the production SGEMM micro-kernel. The BlkSum
    // buffer covers ALL N (including the tail), so this runs once over
    // the full CountN.
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
SQ2BitGemmKernel_BlkSum_CompInt8_Super_Avx512Vnni(
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
    return SQ2BitGemmKernel_BlkSum_CompInt8_Super_Impl<true>(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale, QuantBZeroPoint,
        C, CountM, CountN, CountK, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

//
// Top-level non-VNNI variant. Same tile + layout; integer MAC uses the
// vpmaddubsw + vpmaddwd chain instead of dpbusd.
//
static MLAS_FORCEINLINE size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Super_Avx512(
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
    return SQ2BitGemmKernel_BlkSum_CompInt8_Super_Impl<false>(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale, QuantBZeroPoint,
        C, CountM, CountN, CountK, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

}  // namespace sq2bit_avx512_super
}  // namespace mlas
}  // namespace onnxruntime
