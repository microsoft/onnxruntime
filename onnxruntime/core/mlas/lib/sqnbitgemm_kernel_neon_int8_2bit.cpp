/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon_int8_2bit.cpp

Abstract:

    ARM NEON (FEAT_DotProd) implementation of the W2 SQNBIT_CompInt8 inner
    kernel.

    This TU consumes the block-group packed B-data layout produced by the
    portable W2 pack helpers in `sqnbitgemm_kernel_avx512_2bit.{h,cpp}`. The
    file naming "avx512_2bit" is historical: those pack helpers are pure
    C++ and serve as the cross-arch layout authority for W2.

    Scope of this checkpoint:
      * `BlkLen == 32`, `BlkLen == 64`, `BlkLen == 128`: native NEON dotprod
        (SDOT) inner loops. The packed B layout is the same 4-K-block group
        with bit positions {0..1, 2..3, 4..5, 6..7} per byte for all three
        BlkLens (only the bytes-per-K-block scales with BlkLen). One block-
        group is loaded as 2 / 4 / 8 int8x16 chunks respectively, the in-
        block-group selector is applied with one shift + mask per 16-byte
        chunk, and the per-block int8 dot is computed with 2 / 4 / 8 SDOTs.

    Tile shape: R1xC1 to start (single-row, single-column inner loop).
    Wider tiles (R2xC4 main + R1xC4 tail, etc.) are a follow-up
    optimization checkpoint.

--*/

#include <arm_neon.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "qnbitgemm.h"
#include "qnbitgemm_kernel_neon.h"
#include "sqnbitgemm_kernel_avx512_2bit.h"

namespace sqnbitgemm_neon
{

namespace
{

namespace sq2 = onnxruntime::mlas::sq2bit_avx512;

//
// Sum 4 SDOTs over 64 int8 activation values against 64 int8 weights in
// [0, 3] and reduce to a single int32 dot product.
//
MLAS_FORCEINLINE int32_t
DotInt8_BlkLen64_DotProd(const int8_t* a_blk,
                          const uint8x16_t bw_0_15,
                          const uint8x16_t bw_16_31,
                          const uint8x16_t bw_32_47,
                          const uint8x16_t bw_48_63)
{
    const int8x16_t a0 = vld1q_s8(a_blk + 0);
    const int8x16_t a1 = vld1q_s8(a_blk + 16);
    const int8x16_t a2 = vld1q_s8(a_blk + 32);
    const int8x16_t a3 = vld1q_s8(a_blk + 48);

    const int8x16_t b0 = vreinterpretq_s8_u8(bw_0_15);
    const int8x16_t b1 = vreinterpretq_s8_u8(bw_16_31);
    const int8x16_t b2 = vreinterpretq_s8_u8(bw_32_47);
    const int8x16_t b3 = vreinterpretq_s8_u8(bw_48_63);

    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, a0, b0);
    acc = vdotq_s32(acc, a1, b1);
    acc = vdotq_s32(acc, a2, b2);
    acc = vdotq_s32(acc, a3, b3);
    return vaddvq_s32(acc);
}

//
// Unpack one block_in_group (0..3) slice of a 64-byte block-group into 4
// uint8x16 vectors of weights in [0, 3].
//
// Layout (matches sqnbitgemm_kernel_avx512_2bit.h): byte b of the block-group
// holds the 2-bit weights of all 4 blocks at in-block position b at bit
// positions {0..1, 2..3, 4..5, 6..7}.
//
template <size_t BlkInGroup>
MLAS_FORCEINLINE void
UnpackBlockGroupSliceBlkLen64_DotProd(const std::byte* group,
                                       uint8x16_t& out0_15,
                                       uint8x16_t& out16_31,
                                       uint8x16_t& out32_47,
                                       uint8x16_t& out48_63)
{
    static_assert(BlkInGroup < sq2::kBlockGroupBlks, "BlkInGroup must be in [0, 4)");

    const uint8x16_t mask03 = vdupq_n_u8(0x03);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(group);
    const uint8x16_t v0 = vld1q_u8(p + 0);
    const uint8x16_t v1 = vld1q_u8(p + 16);
    const uint8x16_t v2 = vld1q_u8(p + 32);
    const uint8x16_t v3 = vld1q_u8(p + 48);

    if constexpr (BlkInGroup == 0) {
        // No shift -- vshrq_n_u8 requires immediate shift count in [1, 8].
        out0_15  = vandq_u8(v0, mask03);
        out16_31 = vandq_u8(v1, mask03);
        out32_47 = vandq_u8(v2, mask03);
        out48_63 = vandq_u8(v3, mask03);
    } else {
        constexpr int kShift = 2 * static_cast<int>(BlkInGroup);
        out0_15  = vandq_u8(vshrq_n_u8(v0, kShift), mask03);
        out16_31 = vandq_u8(vshrq_n_u8(v1, kShift), mask03);
        out32_47 = vandq_u8(vshrq_n_u8(v2, kShift), mask03);
        out48_63 = vandq_u8(vshrq_n_u8(v3, kShift), mask03);
    }
}

//
// One (m, n) cell's contribution from a single block-group (4 K-blocks):
// adds 4 * (a_scale * b_scale * int_dot + a_blksum * b_blksum) into acc.
//
// `blk0` is the logical K-block index of the first block in this group.
// Caller guarantees `blk0 + 4 <= BlockCountK` (this is the main inner-loop
// path); the K-tail handler covers groups where blk0 + 3 may exceed
// BlockCountK.
//
MLAS_FORCEINLINE void
AccumOneBlockGroup_BlkLen64_DotProd(float& acc,
                                     const std::byte* group,
                                     const int8_t* a_blk_base,
                                     const float* a_scale_row,
                                     const float* a_blksum_row,
                                     const float* b_scale_for_group,
                                     const float b_blksum_for_4_blks[sq2::kBlockGroupBlks],
                                     size_t blk0)
{
    uint8x16_t bw_0_15, bw_16_31, bw_32_47, bw_48_63;

    // Block 0
    UnpackBlockGroupSliceBlkLen64_DotProd<0>(group, bw_0_15, bw_16_31, bw_32_47, bw_48_63);
    {
        const int32_t dot = DotInt8_BlkLen64_DotProd(a_blk_base + 0 * sq2::kBlkLen,
                                                     bw_0_15, bw_16_31, bw_32_47, bw_48_63);
        acc += a_scale_row[blk0 + 0] * b_scale_for_group[0] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 0] * b_blksum_for_4_blks[0];
    }
    // Block 1
    UnpackBlockGroupSliceBlkLen64_DotProd<1>(group, bw_0_15, bw_16_31, bw_32_47, bw_48_63);
    {
        const int32_t dot = DotInt8_BlkLen64_DotProd(a_blk_base + 1 * sq2::kBlkLen,
                                                     bw_0_15, bw_16_31, bw_32_47, bw_48_63);
        acc += a_scale_row[blk0 + 1] * b_scale_for_group[1] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 1] * b_blksum_for_4_blks[1];
    }
    // Block 2
    UnpackBlockGroupSliceBlkLen64_DotProd<2>(group, bw_0_15, bw_16_31, bw_32_47, bw_48_63);
    {
        const int32_t dot = DotInt8_BlkLen64_DotProd(a_blk_base + 2 * sq2::kBlkLen,
                                                     bw_0_15, bw_16_31, bw_32_47, bw_48_63);
        acc += a_scale_row[blk0 + 2] * b_scale_for_group[2] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 2] * b_blksum_for_4_blks[2];
    }
    // Block 3
    UnpackBlockGroupSliceBlkLen64_DotProd<3>(group, bw_0_15, bw_16_31, bw_32_47, bw_48_63);
    {
        const int32_t dot = DotInt8_BlkLen64_DotProd(a_blk_base + 3 * sq2::kBlkLen,
                                                     bw_0_15, bw_16_31, bw_32_47, bw_48_63);
        acc += a_scale_row[blk0 + 3] * b_scale_for_group[3] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 3] * b_blksum_for_4_blks[3];
    }
}

//
// Tail handler for the last (possibly partial) block-group at the K boundary
// when BlockCountK % kBlockGroupBlks != 0. Only `blocks_in_tail` (1..3) of
// the 4 blocks in the group correspond to real K-blocks; the rest are
// zero-padded by the pack helper (both weights and scales) and the kernel
// must NOT read A past blk = BlockCountK.
//
MLAS_FORCEINLINE void
AccumLastBlockGroup_BlkLen64_DotProd(float& acc,
                                      const std::byte* group,
                                      const int8_t* a_blk_base,
                                      const float* a_scale_row,
                                      const float* a_blksum_row,
                                      const float* b_scale_for_group,
                                      const float b_blksum_for_4_blks[sq2::kBlockGroupBlks],
                                      size_t blk0,
                                      size_t blocks_in_tail)
{
    assert(blocks_in_tail >= 1 && blocks_in_tail <= sq2::kBlockGroupBlks);

    uint8x16_t bw_0_15, bw_16_31, bw_32_47, bw_48_63;

    if (blocks_in_tail >= 1) {
        UnpackBlockGroupSliceBlkLen64_DotProd<0>(group, bw_0_15, bw_16_31, bw_32_47, bw_48_63);
        const int32_t dot = DotInt8_BlkLen64_DotProd(a_blk_base + 0 * sq2::kBlkLen,
                                                     bw_0_15, bw_16_31, bw_32_47, bw_48_63);
        acc += a_scale_row[blk0 + 0] * b_scale_for_group[0] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 0] * b_blksum_for_4_blks[0];
    }
    if (blocks_in_tail >= 2) {
        UnpackBlockGroupSliceBlkLen64_DotProd<1>(group, bw_0_15, bw_16_31, bw_32_47, bw_48_63);
        const int32_t dot = DotInt8_BlkLen64_DotProd(a_blk_base + 1 * sq2::kBlkLen,
                                                     bw_0_15, bw_16_31, bw_32_47, bw_48_63);
        acc += a_scale_row[blk0 + 1] * b_scale_for_group[1] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 1] * b_blksum_for_4_blks[1];
    }
    if (blocks_in_tail >= 3) {
        UnpackBlockGroupSliceBlkLen64_DotProd<2>(group, bw_0_15, bw_16_31, bw_32_47, bw_48_63);
        const int32_t dot = DotInt8_BlkLen64_DotProd(a_blk_base + 2 * sq2::kBlkLen,
                                                     bw_0_15, bw_16_31, bw_32_47, bw_48_63);
        acc += a_scale_row[blk0 + 2] * b_scale_for_group[2] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 2] * b_blksum_for_4_blks[2];
    }
    // blocks_in_tail == 4 is the main-loop case; the tail handler is only
    // invoked when BlockCountK % 4 != 0, so we never need to process block 3
    // here.
}

//
// Native NEON dotprod inner kernel for BlkLen=64. Same external math as
// SQ2BitGemmKernel_BlkSum_CompInt8_Scalar (sqnbitgemm_kernel_avx512_2bit.cpp).
// R1xC1 iteration order; the caller (qnbitgemm.cpp) walks the (m, n) grid.
//
size_t
SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd_BlkLen64(
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
    if (BlockCountK == 0) {
        return 0;
    }

    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, sq2::kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * sq2::kBlockGroupBlks;
    const size_t NMainLocal = (CountN / sq2::kNCols4) * sq2::kNCols4;

    const size_t MainBlockGroups = BlockCountK / sq2::kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % sq2::kBlockGroupBlks;  // 0..3

    const size_t lda = BlockCountK * sq2::kBlkLen;  // bytes per A row
    const size_t lda_scale = BlockCountK;

    for (size_t m = 0; m < CountM; ++m) {
        const int8_t* a_row = reinterpret_cast<const int8_t*>(QuantA + m * lda);
        const float* a_scale_row = QuantAScale + m * lda_scale;
        const float* a_blksum_row = ABlockSum + m * lda_scale;
        float* c_row = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            float acc = (Bias != nullptr) ? Bias[n] : 0.0f;

            // Main K-loop: full block-groups.
            for (size_t g = 0; g < MainBlockGroups; ++g) {
                const size_t blk0 = g * sq2::kBlockGroupBlks;

                const size_t b_offset = sq2::PackedQuantBOffsetBytes_W2(
                    n, g, BlockGroupCountKPadded, NMainLocal);
                const std::byte* group = QuantBData + b_offset;

                // Gather the 4 per-block scales for this (n, group). The
                // scale layout matches the W2 contract -- one float per
                // (n, blk) in PackedQuantBScale, addressed via
                // PackedQuantBScaleOffset_W2.
                float b_scale_for_group[sq2::kBlockGroupBlks];
                for (size_t i = 0; i < sq2::kBlockGroupBlks; ++i) {
                    b_scale_for_group[i] = QuantBScale[sq2::PackedQuantBScaleOffset_W2(
                        n, blk0 + i, BlockCountKPadded, NMainLocal)];
                }

                // Same for the 4 per-block QuantBBlkSums (width-16
                // SGEMM-style layout).
                float b_blksum_for_4_blks[sq2::kBlockGroupBlks];
                for (size_t i = 0; i < sq2::kBlockGroupBlks; ++i) {
                    const size_t off = ((n / 16) * BlockCountK + (blk0 + i)) * 16 + (n % 16);
                    b_blksum_for_4_blks[i] = QuantBBlkSum[off];
                }

                AccumOneBlockGroup_BlkLen64_DotProd(
                    acc, group,
                    a_row + blk0 * sq2::kBlkLen,
                    a_scale_row, a_blksum_row,
                    b_scale_for_group, b_blksum_for_4_blks, blk0);
            }

            // K-tail: process the trailing partial group (1..3 real blocks).
            if (TailBlocks != 0) {
                const size_t g = MainBlockGroups;
                const size_t blk0 = g * sq2::kBlockGroupBlks;

                const size_t b_offset = sq2::PackedQuantBOffsetBytes_W2(
                    n, g, BlockGroupCountKPadded, NMainLocal);
                const std::byte* group = QuantBData + b_offset;

                float b_scale_for_group[sq2::kBlockGroupBlks] = {};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    b_scale_for_group[i] = QuantBScale[sq2::PackedQuantBScaleOffset_W2(
                        n, blk0 + i, BlockCountKPadded, NMainLocal)];
                }

                float b_blksum_for_4_blks[sq2::kBlockGroupBlks] = {};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    const size_t off = ((n / 16) * BlockCountK + (blk0 + i)) * 16 + (n % 16);
                    b_blksum_for_4_blks[i] = QuantBBlkSum[off];
                }

                AccumLastBlockGroup_BlkLen64_DotProd(
                    acc, group,
                    a_row + blk0 * sq2::kBlkLen,
                    a_scale_row, a_blksum_row,
                    b_scale_for_group, b_blksum_for_4_blks, blk0, TailBlocks);
            }

            c_row[n] = acc;
        }
    }

    return CountM;
}

// -----------------------------------------------------------------------------
// BlkLen=128 helpers + inner kernel.
//
// One K-block holds 128 weights = 8 int8x16 chunks. One block-group holds 4
// K-blocks = 128 packed bytes. Layout (bit-positions of the in-block-group
// selector) is identical to BlkLen=64; only the bytes-per-K-block scales.
// -----------------------------------------------------------------------------

constexpr size_t kBlkLen128VecsPerBlk = 8;  // 128 / 16

MLAS_FORCEINLINE int32_t
DotInt8_BlkLen128_DotProd(const int8_t* a_blk, const uint8x16_t bw[kBlkLen128VecsPerBlk])
{
    int32x4_t acc = vdupq_n_s32(0);
    for (size_t i = 0; i < kBlkLen128VecsPerBlk; ++i) {
        const int8x16_t a = vld1q_s8(a_blk + i * 16);
        const int8x16_t b = vreinterpretq_s8_u8(bw[i]);
        acc = vdotq_s32(acc, a, b);
    }
    return vaddvq_s32(acc);
}

template <size_t BlkInGroup>
MLAS_FORCEINLINE void
UnpackBlockGroupSliceBlkLen128_DotProd(const std::byte* group, uint8x16_t out[kBlkLen128VecsPerBlk])
{
    static_assert(BlkInGroup < sq2::kBlockGroupBlks, "BlkInGroup must be in [0, 4)");

    const uint8x16_t mask03 = vdupq_n_u8(0x03);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(group);
    for (size_t i = 0; i < kBlkLen128VecsPerBlk; ++i) {
        const uint8x16_t v = vld1q_u8(p + i * 16);
        if constexpr (BlkInGroup == 0) {
            out[i] = vandq_u8(v, mask03);
        } else {
            constexpr int kShift = 2 * static_cast<int>(BlkInGroup);
            out[i] = vandq_u8(vshrq_n_u8(v, kShift), mask03);
        }
    }
}

MLAS_FORCEINLINE void
AccumOneBlockGroup_BlkLen128_DotProd(float& acc,
                                      const std::byte* group,
                                      const int8_t* a_blk_base,
                                      const float* a_scale_row,
                                      const float* a_blksum_row,
                                      const float* b_scale_for_group,
                                      const float b_blksum_for_4_blks[sq2::kBlockGroupBlks],
                                      size_t blk0)
{
    uint8x16_t bw[kBlkLen128VecsPerBlk];

    UnpackBlockGroupSliceBlkLen128_DotProd<0>(group, bw);
    {
        const int32_t dot = DotInt8_BlkLen128_DotProd(a_blk_base + 0 * sq2::kBlkLen128, bw);
        acc += a_scale_row[blk0 + 0] * b_scale_for_group[0] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 0] * b_blksum_for_4_blks[0];
    }
    UnpackBlockGroupSliceBlkLen128_DotProd<1>(group, bw);
    {
        const int32_t dot = DotInt8_BlkLen128_DotProd(a_blk_base + 1 * sq2::kBlkLen128, bw);
        acc += a_scale_row[blk0 + 1] * b_scale_for_group[1] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 1] * b_blksum_for_4_blks[1];
    }
    UnpackBlockGroupSliceBlkLen128_DotProd<2>(group, bw);
    {
        const int32_t dot = DotInt8_BlkLen128_DotProd(a_blk_base + 2 * sq2::kBlkLen128, bw);
        acc += a_scale_row[blk0 + 2] * b_scale_for_group[2] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 2] * b_blksum_for_4_blks[2];
    }
    UnpackBlockGroupSliceBlkLen128_DotProd<3>(group, bw);
    {
        const int32_t dot = DotInt8_BlkLen128_DotProd(a_blk_base + 3 * sq2::kBlkLen128, bw);
        acc += a_scale_row[blk0 + 3] * b_scale_for_group[3] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 3] * b_blksum_for_4_blks[3];
    }
}

MLAS_FORCEINLINE void
AccumLastBlockGroup_BlkLen128_DotProd(float& acc,
                                       const std::byte* group,
                                       const int8_t* a_blk_base,
                                       const float* a_scale_row,
                                       const float* a_blksum_row,
                                       const float* b_scale_for_group,
                                       const float b_blksum_for_4_blks[sq2::kBlockGroupBlks],
                                       size_t blk0,
                                       size_t blocks_in_tail)
{
    assert(blocks_in_tail >= 1 && blocks_in_tail <= sq2::kBlockGroupBlks);

    uint8x16_t bw[kBlkLen128VecsPerBlk];

    if (blocks_in_tail >= 1) {
        UnpackBlockGroupSliceBlkLen128_DotProd<0>(group, bw);
        const int32_t dot = DotInt8_BlkLen128_DotProd(a_blk_base + 0 * sq2::kBlkLen128, bw);
        acc += a_scale_row[blk0 + 0] * b_scale_for_group[0] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 0] * b_blksum_for_4_blks[0];
    }
    if (blocks_in_tail >= 2) {
        UnpackBlockGroupSliceBlkLen128_DotProd<1>(group, bw);
        const int32_t dot = DotInt8_BlkLen128_DotProd(a_blk_base + 1 * sq2::kBlkLen128, bw);
        acc += a_scale_row[blk0 + 1] * b_scale_for_group[1] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 1] * b_blksum_for_4_blks[1];
    }
    if (blocks_in_tail >= 3) {
        UnpackBlockGroupSliceBlkLen128_DotProd<2>(group, bw);
        const int32_t dot = DotInt8_BlkLen128_DotProd(a_blk_base + 2 * sq2::kBlkLen128, bw);
        acc += a_scale_row[blk0 + 2] * b_scale_for_group[2] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 2] * b_blksum_for_4_blks[2];
    }
}

size_t
SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd_BlkLen128(
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
    if (BlockCountK == 0) {
        return 0;
    }

    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, sq2::kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * sq2::kBlockGroupBlks;
    const size_t NMainLocal = (CountN / sq2::kNCols4) * sq2::kNCols4;

    const size_t MainBlockGroups = BlockCountK / sq2::kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % sq2::kBlockGroupBlks;

    const size_t lda = BlockCountK * sq2::kBlkLen128;
    const size_t lda_scale = BlockCountK;

    for (size_t m = 0; m < CountM; ++m) {
        const int8_t* a_row = reinterpret_cast<const int8_t*>(QuantA + m * lda);
        const float* a_scale_row = QuantAScale + m * lda_scale;
        const float* a_blksum_row = ABlockSum + m * lda_scale;
        float* c_row = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            float acc = (Bias != nullptr) ? Bias[n] : 0.0f;

            for (size_t g = 0; g < MainBlockGroups; ++g) {
                const size_t blk0 = g * sq2::kBlockGroupBlks;

                const size_t b_offset = sq2::PackedQuantBOffsetBytes_W2_BlkLen128(
                    n, g, BlockGroupCountKPadded, NMainLocal);
                const std::byte* group = QuantBData + b_offset;

                float b_scale_for_group[sq2::kBlockGroupBlks];
                for (size_t i = 0; i < sq2::kBlockGroupBlks; ++i) {
                    b_scale_for_group[i] = QuantBScale[sq2::PackedQuantBScaleOffset_W2(
                        n, blk0 + i, BlockCountKPadded, NMainLocal)];
                }

                float b_blksum_for_4_blks[sq2::kBlockGroupBlks];
                for (size_t i = 0; i < sq2::kBlockGroupBlks; ++i) {
                    const size_t off = ((n / 16) * BlockCountK + (blk0 + i)) * 16 + (n % 16);
                    b_blksum_for_4_blks[i] = QuantBBlkSum[off];
                }

                AccumOneBlockGroup_BlkLen128_DotProd(
                    acc, group,
                    a_row + blk0 * sq2::kBlkLen128,
                    a_scale_row, a_blksum_row,
                    b_scale_for_group, b_blksum_for_4_blks, blk0);
            }

            if (TailBlocks != 0) {
                const size_t g = MainBlockGroups;
                const size_t blk0 = g * sq2::kBlockGroupBlks;

                const size_t b_offset = sq2::PackedQuantBOffsetBytes_W2_BlkLen128(
                    n, g, BlockGroupCountKPadded, NMainLocal);
                const std::byte* group = QuantBData + b_offset;

                float b_scale_for_group[sq2::kBlockGroupBlks] = {};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    b_scale_for_group[i] = QuantBScale[sq2::PackedQuantBScaleOffset_W2(
                        n, blk0 + i, BlockCountKPadded, NMainLocal)];
                }

                float b_blksum_for_4_blks[sq2::kBlockGroupBlks] = {};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    const size_t off = ((n / 16) * BlockCountK + (blk0 + i)) * 16 + (n % 16);
                    b_blksum_for_4_blks[i] = QuantBBlkSum[off];
                }

                AccumLastBlockGroup_BlkLen128_DotProd(
                    acc, group,
                    a_row + blk0 * sq2::kBlkLen128,
                    a_scale_row, a_blksum_row,
                    b_scale_for_group, b_blksum_for_4_blks, blk0, TailBlocks);
            }

            c_row[n] = acc;
        }
    }

    return CountM;
}

// -----------------------------------------------------------------------------
// BlkLen=32 helpers + inner kernel.
//
// One K-block holds 32 weights = 2 int8x16 chunks. One block-group holds 4
// K-blocks = 32 packed bytes (fits in two NEON registers). Same bit-position
// layout as BlkLen=64/128.
// -----------------------------------------------------------------------------

constexpr size_t kBlkLen32VecsPerBlk = 2;  // 32 / 16

MLAS_FORCEINLINE int32_t
DotInt8_BlkLen32_DotProd(const int8_t* a_blk, const uint8x16_t bw[kBlkLen32VecsPerBlk])
{
    int32x4_t acc = vdupq_n_s32(0);
    const int8x16_t a0 = vld1q_s8(a_blk + 0);
    const int8x16_t a1 = vld1q_s8(a_blk + 16);
    acc = vdotq_s32(acc, a0, vreinterpretq_s8_u8(bw[0]));
    acc = vdotq_s32(acc, a1, vreinterpretq_s8_u8(bw[1]));
    return vaddvq_s32(acc);
}

template <size_t BlkInGroup>
MLAS_FORCEINLINE void
UnpackBlockGroupSliceBlkLen32_DotProd(const std::byte* group, uint8x16_t out[kBlkLen32VecsPerBlk])
{
    static_assert(BlkInGroup < sq2::kBlockGroupBlks, "BlkInGroup must be in [0, 4)");

    const uint8x16_t mask03 = vdupq_n_u8(0x03);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(group);
    const uint8x16_t v0 = vld1q_u8(p + 0);
    const uint8x16_t v1 = vld1q_u8(p + 16);

    if constexpr (BlkInGroup == 0) {
        out[0] = vandq_u8(v0, mask03);
        out[1] = vandq_u8(v1, mask03);
    } else {
        constexpr int kShift = 2 * static_cast<int>(BlkInGroup);
        out[0] = vandq_u8(vshrq_n_u8(v0, kShift), mask03);
        out[1] = vandq_u8(vshrq_n_u8(v1, kShift), mask03);
    }
}

MLAS_FORCEINLINE void
AccumOneBlockGroup_BlkLen32_DotProd(float& acc,
                                     const std::byte* group,
                                     const int8_t* a_blk_base,
                                     const float* a_scale_row,
                                     const float* a_blksum_row,
                                     const float* b_scale_for_group,
                                     const float b_blksum_for_4_blks[sq2::kBlockGroupBlks],
                                     size_t blk0)
{
    uint8x16_t bw[kBlkLen32VecsPerBlk];

    UnpackBlockGroupSliceBlkLen32_DotProd<0>(group, bw);
    {
        const int32_t dot = DotInt8_BlkLen32_DotProd(a_blk_base + 0 * sq2::kBlkLen32, bw);
        acc += a_scale_row[blk0 + 0] * b_scale_for_group[0] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 0] * b_blksum_for_4_blks[0];
    }
    UnpackBlockGroupSliceBlkLen32_DotProd<1>(group, bw);
    {
        const int32_t dot = DotInt8_BlkLen32_DotProd(a_blk_base + 1 * sq2::kBlkLen32, bw);
        acc += a_scale_row[blk0 + 1] * b_scale_for_group[1] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 1] * b_blksum_for_4_blks[1];
    }
    UnpackBlockGroupSliceBlkLen32_DotProd<2>(group, bw);
    {
        const int32_t dot = DotInt8_BlkLen32_DotProd(a_blk_base + 2 * sq2::kBlkLen32, bw);
        acc += a_scale_row[blk0 + 2] * b_scale_for_group[2] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 2] * b_blksum_for_4_blks[2];
    }
    UnpackBlockGroupSliceBlkLen32_DotProd<3>(group, bw);
    {
        const int32_t dot = DotInt8_BlkLen32_DotProd(a_blk_base + 3 * sq2::kBlkLen32, bw);
        acc += a_scale_row[blk0 + 3] * b_scale_for_group[3] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 3] * b_blksum_for_4_blks[3];
    }
}

MLAS_FORCEINLINE void
AccumLastBlockGroup_BlkLen32_DotProd(float& acc,
                                      const std::byte* group,
                                      const int8_t* a_blk_base,
                                      const float* a_scale_row,
                                      const float* a_blksum_row,
                                      const float* b_scale_for_group,
                                      const float b_blksum_for_4_blks[sq2::kBlockGroupBlks],
                                      size_t blk0,
                                      size_t blocks_in_tail)
{
    assert(blocks_in_tail >= 1 && blocks_in_tail <= sq2::kBlockGroupBlks);

    uint8x16_t bw[kBlkLen32VecsPerBlk];

    if (blocks_in_tail >= 1) {
        UnpackBlockGroupSliceBlkLen32_DotProd<0>(group, bw);
        const int32_t dot = DotInt8_BlkLen32_DotProd(a_blk_base + 0 * sq2::kBlkLen32, bw);
        acc += a_scale_row[blk0 + 0] * b_scale_for_group[0] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 0] * b_blksum_for_4_blks[0];
    }
    if (blocks_in_tail >= 2) {
        UnpackBlockGroupSliceBlkLen32_DotProd<1>(group, bw);
        const int32_t dot = DotInt8_BlkLen32_DotProd(a_blk_base + 1 * sq2::kBlkLen32, bw);
        acc += a_scale_row[blk0 + 1] * b_scale_for_group[1] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 1] * b_blksum_for_4_blks[1];
    }
    if (blocks_in_tail >= 3) {
        UnpackBlockGroupSliceBlkLen32_DotProd<2>(group, bw);
        const int32_t dot = DotInt8_BlkLen32_DotProd(a_blk_base + 2 * sq2::kBlkLen32, bw);
        acc += a_scale_row[blk0 + 2] * b_scale_for_group[2] * static_cast<float>(dot);
        acc += a_blksum_row[blk0 + 2] * b_blksum_for_4_blks[2];
    }
}

size_t
SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd_BlkLen32(
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
    if (BlockCountK == 0) {
        return 0;
    }

    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, sq2::kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * sq2::kBlockGroupBlks;
    const size_t NMainLocal = (CountN / sq2::kNCols4) * sq2::kNCols4;

    const size_t MainBlockGroups = BlockCountK / sq2::kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % sq2::kBlockGroupBlks;

    const size_t lda = BlockCountK * sq2::kBlkLen32;
    const size_t lda_scale = BlockCountK;

    for (size_t m = 0; m < CountM; ++m) {
        const int8_t* a_row = reinterpret_cast<const int8_t*>(QuantA + m * lda);
        const float* a_scale_row = QuantAScale + m * lda_scale;
        const float* a_blksum_row = ABlockSum + m * lda_scale;
        float* c_row = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            float acc = (Bias != nullptr) ? Bias[n] : 0.0f;

            for (size_t g = 0; g < MainBlockGroups; ++g) {
                const size_t blk0 = g * sq2::kBlockGroupBlks;

                const size_t b_offset = sq2::PackedQuantBOffsetBytes_W2_BlkLen32(
                    n, g, BlockGroupCountKPadded, NMainLocal);
                const std::byte* group = QuantBData + b_offset;

                float b_scale_for_group[sq2::kBlockGroupBlks];
                for (size_t i = 0; i < sq2::kBlockGroupBlks; ++i) {
                    b_scale_for_group[i] = QuantBScale[sq2::PackedQuantBScaleOffset_W2(
                        n, blk0 + i, BlockCountKPadded, NMainLocal)];
                }

                float b_blksum_for_4_blks[sq2::kBlockGroupBlks];
                for (size_t i = 0; i < sq2::kBlockGroupBlks; ++i) {
                    const size_t off = ((n / 16) * BlockCountK + (blk0 + i)) * 16 + (n % 16);
                    b_blksum_for_4_blks[i] = QuantBBlkSum[off];
                }

                AccumOneBlockGroup_BlkLen32_DotProd(
                    acc, group,
                    a_row + blk0 * sq2::kBlkLen32,
                    a_scale_row, a_blksum_row,
                    b_scale_for_group, b_blksum_for_4_blks, blk0);
            }

            if (TailBlocks != 0) {
                const size_t g = MainBlockGroups;
                const size_t blk0 = g * sq2::kBlockGroupBlks;

                const size_t b_offset = sq2::PackedQuantBOffsetBytes_W2_BlkLen32(
                    n, g, BlockGroupCountKPadded, NMainLocal);
                const std::byte* group = QuantBData + b_offset;

                float b_scale_for_group[sq2::kBlockGroupBlks] = {};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    b_scale_for_group[i] = QuantBScale[sq2::PackedQuantBScaleOffset_W2(
                        n, blk0 + i, BlockCountKPadded, NMainLocal)];
                }

                float b_blksum_for_4_blks[sq2::kBlockGroupBlks] = {};
                for (size_t i = 0; i < TailBlocks; ++i) {
                    const size_t off = ((n / 16) * BlockCountK + (blk0 + i)) * 16 + (n % 16);
                    b_blksum_for_4_blks[i] = QuantBBlkSum[off];
                }

                AccumLastBlockGroup_BlkLen32_DotProd(
                    acc, group,
                    a_row + blk0 * sq2::kBlkLen32,
                    a_scale_row, a_blksum_row,
                    b_scale_for_group, b_blksum_for_4_blks, blk0, TailBlocks);
            }

            c_row[n] = acc;
        }
    }

    return CountM;
}

}  // unnamed namespace

//
// W2 CompInt8 kernel entry point (DotProd backend).
//
// All three BlkLens ({32, 64, 128}) route to native NEON dotprod kernels.
//
size_t
SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd(
    size_t BlkLen,
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
    if (BlkLen == sq2::kBlkLen) {
        return SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd_BlkLen64(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
    }
    if (BlkLen == sq2::kBlkLen128) {
        return SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd_BlkLen128(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
    }
    if (BlkLen == sq2::kBlkLen32) {
        return SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd_BlkLen32(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
    }

    // Unsupported BlkLen for W2 -- defer to the portable scalar reference.
    // Should be unreachable: the W2 dispatch advertises only BlkLen
    // in {32, 64, 128} via MlasIsQNBitGemmAvailable.
    return sq2::SQ2BitGemmKernel_BlkSum_CompInt8_Scalar(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale, QuantBZeroPoint,
        C, CountM, CountN, CountK, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

}  // namespace sqnbitgemm_neon
