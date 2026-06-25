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

    Scope:
      * `BlkLen == 32`, `BlkLen == 64`, `BlkLen == 128`: native NEON dotprod
        (SDOT) inner loops. The packed B layout is the same 4-K-block group
        with bit positions {0..1, 2..3, 4..5, 6..7} per byte for all three
        BlkLens (only the bytes-per-K-block scales with BlkLen). One block-
        group is loaded as 2 / 4 / 8 int8x16 chunks respectively, the in-
        block-group selector is applied with one shift + mask per 16-byte
        chunk, and the per-block int8 dot is computed with 2 / 4 / 8 SDOTs.

    Tile grid (per BlkLen):
        R2xC8 (M-pair, main 8-N), R1xC8 (M-tail, main 8-N),
        R2xC4 (M-pair, main 4-N), R1xC4 (M-tail, main 4-N),
        R2xC1 (M-pair, 1..3-N tail), R1xC1 (M-tail, 1..3-N tail).
    All six variants are template instantiations of a single tile kernel
    (`Q2Int8GemmRxC_DotProd<BlkLen, NRows, NCols>`). The per-BlkLen
    dispatcher splits CountN into 8-aligned, 4-aligned, and 1..3 tail
    regions and CountM into M-pairs + odd-M-tail, then dispatches to up
    to six tile-shape kernels.

    Deviation vs. W4/W8 DotProd ARM reference: W4/W8 use R1-only on
    DotProd (R2 reserved for I8MM). W2 adds R2 to DotProd too because
    W2's B-unpack (8 weights/byte, 2 shifts+masks per chunk) is the
    heaviest of any width and amortising it + the per-K-block scale /
    BlkSum gather across an M-pair is a meaningful win.

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

// ============================================================================
// Shared templated tile machinery for the W2 DotProd kernel.
//
// One templated tile kernel `Q2Int8GemmRxC_DotProd<BlkLen, NRows, NCols>`
// covers all 18 BlkLen x tile combinations. The per-BlkLen public entries
// each delegate to `SQ2BitGemm_Dispatch_NeonDotProd<BlkLen>`, which splits
// the (M, N) plane into up to 6 sub-rectangles (R{1,2} x C{1,4,8}) and
// dispatches one instantiation per piece.
//
// Layout (see sqnbitgemm_kernel_avx512_2bit.h):
//   * Packed B is grouped into `kNCols4` (=4) col groups in the main region
//     (n < NMain = floor(CountN / kNCols4) * kNCols4) and column-major in
//     the tail region (n >= NMain). C8/C4 tiles operate entirely in the
//     main region; the C1 tile handles 1..3 trailing cols in the tail.
//   * `BlockCountK` is padded to a multiple of `kBlockGroupBlks` (=4) for
//     storage; padding contributes 0 to dot products and to the BlkSum
//     correction.
//   * QuantBBlkSum uses the existing width-16 chunked layout consumed by
//     the SGEMM correction step.
// ============================================================================

template <size_t BlkLen> struct BlkLenTraits;
template <> struct BlkLenTraits<32>  { static constexpr size_t kVecsPerBlk = 2; };
template <> struct BlkLenTraits<64>  { static constexpr size_t kVecsPerBlk = 4; };
template <> struct BlkLenTraits<128> { static constexpr size_t kVecsPerBlk = 8; };

template <size_t BlkLen> inline constexpr size_t kBlockGroupBytesV = 0;
template <> inline constexpr size_t kBlockGroupBytesV<32>  = sq2::kBlockGroupBytes32;
template <> inline constexpr size_t kBlockGroupBytesV<64>  = sq2::kBlockGroupBytes;
template <> inline constexpr size_t kBlockGroupBytesV<128> = sq2::kBlockGroupBytes128;

// Per-BlkLen packed-B byte offset (folds the BlkLen-specific overloads).
template <size_t BlkLen>
MLAS_FORCEINLINE size_t
PackedBOffsetBytes(size_t n, size_t blk_group,
                   size_t BlockGroupCountKPadded, size_t NMain)
{
    if constexpr (BlkLen == 32) {
        return sq2::PackedQuantBOffsetBytes_W2_BlkLen32(
            n, blk_group, BlockGroupCountKPadded, NMain);
    } else if constexpr (BlkLen == 128) {
        return sq2::PackedQuantBOffsetBytes_W2_BlkLen128(
            n, blk_group, BlockGroupCountKPadded, NMain);
    } else {
        return sq2::PackedQuantBOffsetBytes_W2(
            n, blk_group, BlockGroupCountKPadded, NMain);
    }
}

// Unpack one in-group block (BlkInGroup in [0,4)) into `kVecsPerBlk`
// uint8x16_t vectors of 2-bit weights in [0, 3].
template <size_t BlkLen, size_t BlkInGroup>
MLAS_FORCEINLINE void
UnpackBlockGroupSlice_DotProd(const std::byte* group,
                              uint8x16_t out[BlkLenTraits<BlkLen>::kVecsPerBlk])
{
    static_assert(BlkInGroup < sq2::kBlockGroupBlks, "BlkInGroup must be < 4");
    constexpr size_t kVecs = BlkLenTraits<BlkLen>::kVecsPerBlk;
    const uint8x16_t mask03 = vdupq_n_u8(0x03);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(group);
    for (size_t v = 0; v < kVecs; ++v) {
        const uint8x16_t raw = vld1q_u8(p + v * 16);
        if constexpr (BlkInGroup == 0) {
            // vshrq_n_u8 immediate must be in [1, 8] -- skip shift here.
            out[v] = vandq_u8(raw, mask03);
        } else {
            constexpr int kShift = 2 * static_cast<int>(BlkInGroup);
            out[v] = vandq_u8(vshrq_n_u8(raw, kShift), mask03);
        }
    }
}

// One in-group block's contribution to NRows x NCols scalar accumulators.
// All A vecs for the K-block are pre-loaded once per row, then for each
// output col we unpack one col's B and run NRows int dot+reduce paths.
// Worst-case live NEON regs (R2xC*, BlkLen=128): 16 A + 8 B + 1 scratch = 25.
template <size_t BlkLen, size_t NRows, size_t NCols, size_t BlkInGroup>
MLAS_FORCEINLINE void
AccumOneBlock_RxC_DotProd(
    float (&acc)[NRows][NCols],
    const std::byte* const group_cols[NCols],
    const int8_t* const a_blk_bases[NRows],
    const float* const a_scale_rows[NRows],
    const float* const a_blksum_rows[NRows],
    const float b_scales[NCols][sq2::kBlockGroupBlks],
    const float b_blksums[NCols][sq2::kBlockGroupBlks],
    size_t blk0)
{
    constexpr size_t kVecs = BlkLenTraits<BlkLen>::kVecsPerBlk;
    constexpr size_t kK = BlkLen;

    int8x16_t av[NRows][kVecs];
    for (size_t r = 0; r < NRows; ++r) {
        const int8_t* a = a_blk_bases[r] + BlkInGroup * kK;
        for (size_t v = 0; v < kVecs; ++v) {
            av[r][v] = vld1q_s8(a + v * 16);
        }
    }

    for (size_t c = 0; c < NCols; ++c) {
        uint8x16_t bw[kVecs];
        UnpackBlockGroupSlice_DotProd<BlkLen, BlkInGroup>(group_cols[c], bw);
        for (size_t r = 0; r < NRows; ++r) {
            int32x4_t s = vdupq_n_s32(0);
            for (size_t v = 0; v < kVecs; ++v) {
                s = vdotq_s32(s, av[r][v], vreinterpretq_s8_u8(bw[v]));
            }
            const int32_t dot = vaddvq_s32(s);
            const float a_s  = a_scale_rows[r][blk0 + BlkInGroup];
            const float a_bs = a_blksum_rows[r][blk0 + BlkInGroup];
            acc[r][c] += a_s * b_scales[c][BlkInGroup] * static_cast<float>(dot);
            acc[r][c] += a_bs * b_blksums[c][BlkInGroup];
        }
    }
}

// Full block-group: 4 in-group blocks unrolled at compile time.
template <size_t BlkLen, size_t NRows, size_t NCols>
MLAS_FORCEINLINE void
AccumOneBlockGroup_RxC_DotProd(
    float (&acc)[NRows][NCols],
    const std::byte* const group_cols[NCols],
    const int8_t* const a_blk_bases[NRows],
    const float* const a_scale_rows[NRows],
    const float* const a_blksum_rows[NRows],
    const float b_scales[NCols][sq2::kBlockGroupBlks],
    const float b_blksums[NCols][sq2::kBlockGroupBlks],
    size_t blk0)
{
    AccumOneBlock_RxC_DotProd<BlkLen, NRows, NCols, 0>(
        acc, group_cols, a_blk_bases, a_scale_rows, a_blksum_rows, b_scales, b_blksums, blk0);
    AccumOneBlock_RxC_DotProd<BlkLen, NRows, NCols, 1>(
        acc, group_cols, a_blk_bases, a_scale_rows, a_blksum_rows, b_scales, b_blksums, blk0);
    AccumOneBlock_RxC_DotProd<BlkLen, NRows, NCols, 2>(
        acc, group_cols, a_blk_bases, a_scale_rows, a_blksum_rows, b_scales, b_blksums, blk0);
    AccumOneBlock_RxC_DotProd<BlkLen, NRows, NCols, 3>(
        acc, group_cols, a_blk_bases, a_scale_rows, a_blksum_rows, b_scales, b_blksums, blk0);
}

// K-tail block-group: only `blocks_in_tail` (1..3) in-group blocks are
// real K-blocks (pack helper zero-padded the rest). We still must not
// read A past BlockCountK, so the run-time gate stays.
template <size_t BlkLen, size_t NRows, size_t NCols>
MLAS_FORCEINLINE void
AccumLastBlockGroup_RxC_DotProd(
    float (&acc)[NRows][NCols],
    const std::byte* const group_cols[NCols],
    const int8_t* const a_blk_bases[NRows],
    const float* const a_scale_rows[NRows],
    const float* const a_blksum_rows[NRows],
    const float b_scales[NCols][sq2::kBlockGroupBlks],
    const float b_blksums[NCols][sq2::kBlockGroupBlks],
    size_t blk0, size_t blocks_in_tail)
{
    assert(blocks_in_tail >= 1 && blocks_in_tail <= sq2::kBlockGroupBlks);
    if (blocks_in_tail >= 1) {
        AccumOneBlock_RxC_DotProd<BlkLen, NRows, NCols, 0>(
            acc, group_cols, a_blk_bases, a_scale_rows, a_blksum_rows,
            b_scales, b_blksums, blk0);
    }
    if (blocks_in_tail >= 2) {
        AccumOneBlock_RxC_DotProd<BlkLen, NRows, NCols, 1>(
            acc, group_cols, a_blk_bases, a_scale_rows, a_blksum_rows,
            b_scales, b_blksums, blk0);
    }
    if (blocks_in_tail >= 3) {
        AccumOneBlock_RxC_DotProd<BlkLen, NRows, NCols, 2>(
            acc, group_cols, a_blk_bases, a_scale_rows, a_blksum_rows,
            b_scales, b_blksums, blk0);
    }
    // blocks_in_tail == 4 is the main-loop case, not invoked here.
}

// Templated tile kernel for the sub-rectangle
//   [m_start, m_start+m_count) x [n_start, n_start+n_count)
// Caller guarantees: m_count is a multiple of NRows; n_count is a multiple
// of NCols (for NCols==1 this is trivial -- the C1 tile iterates the 1..3
// trailing cols one at a time).
template <size_t BlkLen, size_t NRows, size_t NCols>
MLAS_FORCEINLINE void
Q2Int8GemmRxC_DotProd(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountN,                // full N (for NMain calc + BlkSum width-16)
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum,
    size_t m_start,
    size_t m_count,
    size_t n_start,
    size_t n_count)
{
    if (m_count == 0 || n_count == 0) return;

    constexpr size_t kK = BlkLen;
    constexpr size_t kBlockGroupBytes = kBlockGroupBytesV<BlkLen>;
    // C8/C4 tiles operate fully in the main NCols4-grouped region; C1 tile
    // operates fully in the column-major tail region.
    constexpr bool kInMain = (NCols > 1);
    constexpr size_t kGroupStrideBytes =
        kInMain ? (sq2::kNCols4 * kBlockGroupBytes) : kBlockGroupBytes;

    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, sq2::kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * sq2::kBlockGroupBlks;
    const size_t NMainLocal = (CountN / sq2::kNCols4) * sq2::kNCols4;
    const size_t MainBlockGroups = BlockCountK / sq2::kBlockGroupBlks;
    const size_t TailBlocks = BlockCountK % sq2::kBlockGroupBlks;
    const size_t lda = BlockCountK * kK;
    const size_t lda_scale = BlockCountK;

    assert(m_count % NRows == 0);

    for (size_t m = m_start; m < m_start + m_count; m += NRows) {
        const int8_t* a_rows[NRows];
        const float* a_scale_rows[NRows];
        const float* a_blksum_rows[NRows];
        float* c_rows[NRows];
        for (size_t r = 0; r < NRows; ++r) {
            a_rows[r] = reinterpret_cast<const int8_t*>(QuantA + (m + r) * lda);
            a_scale_rows[r] = QuantAScale + (m + r) * lda_scale;
            a_blksum_rows[r] = ABlockSum + (m + r) * lda_scale;
            c_rows[r] = C + (m + r) * ldc;
        }

        for (size_t n = n_start; n < n_start + n_count; n += NCols) {
            float acc[NRows][NCols];
            for (size_t r = 0; r < NRows; ++r) {
                for (size_t c = 0; c < NCols; ++c) {
                    acc[r][c] = (Bias != nullptr) ? Bias[n + c] : 0.0f;
                }
            }

            // Per-col g=0 base pointers; per-group advance is `kGroupStrideBytes`
            // (constant for this tile shape / region).
            const std::byte* group_cols_g0[NCols];
            for (size_t c = 0; c < NCols; ++c) {
                group_cols_g0[c] = QuantBData + PackedBOffsetBytes<BlkLen>(
                    n + c, 0, BlockGroupCountKPadded, NMainLocal);
            }

            for (size_t g = 0; g < MainBlockGroups; ++g) {
                const size_t blk0 = g * sq2::kBlockGroupBlks;

                const std::byte* group_cols[NCols];
                for (size_t c = 0; c < NCols; ++c) {
                    group_cols[c] = group_cols_g0[c] + g * kGroupStrideBytes;
                }

                float b_scales[NCols][sq2::kBlockGroupBlks];
                float b_blksums[NCols][sq2::kBlockGroupBlks];
                for (size_t c = 0; c < NCols; ++c) {
                    for (size_t i = 0; i < sq2::kBlockGroupBlks; ++i) {
                        b_scales[c][i] = QuantBScale[sq2::PackedQuantBScaleOffset_W2(
                            n + c, blk0 + i, BlockCountKPadded, NMainLocal)];
                        const size_t off =
                            (((n + c) / 16) * BlockCountK + (blk0 + i)) * 16 + ((n + c) % 16);
                        b_blksums[c][i] = QuantBBlkSum[off];
                    }
                }

                const int8_t* a_blk_bases[NRows];
                for (size_t r = 0; r < NRows; ++r) {
                    a_blk_bases[r] = a_rows[r] + blk0 * kK;
                }

                AccumOneBlockGroup_RxC_DotProd<BlkLen, NRows, NCols>(
                    acc, group_cols, a_blk_bases,
                    a_scale_rows, a_blksum_rows,
                    b_scales, b_blksums, blk0);
            }

            if (TailBlocks != 0) {
                const size_t g = MainBlockGroups;
                const size_t blk0 = g * sq2::kBlockGroupBlks;

                const std::byte* group_cols[NCols];
                for (size_t c = 0; c < NCols; ++c) {
                    group_cols[c] = group_cols_g0[c] + g * kGroupStrideBytes;
                }

                float b_scales[NCols][sq2::kBlockGroupBlks] = {};
                float b_blksums[NCols][sq2::kBlockGroupBlks] = {};
                for (size_t c = 0; c < NCols; ++c) {
                    for (size_t i = 0; i < TailBlocks; ++i) {
                        b_scales[c][i] = QuantBScale[sq2::PackedQuantBScaleOffset_W2(
                            n + c, blk0 + i, BlockCountKPadded, NMainLocal)];
                        const size_t off =
                            (((n + c) / 16) * BlockCountK + (blk0 + i)) * 16 + ((n + c) % 16);
                        b_blksums[c][i] = QuantBBlkSum[off];
                    }
                }

                const int8_t* a_blk_bases[NRows];
                for (size_t r = 0; r < NRows; ++r) {
                    a_blk_bases[r] = a_rows[r] + blk0 * kK;
                }

                AccumLastBlockGroup_RxC_DotProd<BlkLen, NRows, NCols>(
                    acc, group_cols, a_blk_bases,
                    a_scale_rows, a_blksum_rows,
                    b_scales, b_blksums, blk0, TailBlocks);
            }

            for (size_t r = 0; r < NRows; ++r) {
                for (size_t c = 0; c < NCols; ++c) {
                    c_rows[r][n + c] = acc[r][c];
                }
            }
        }
    }
}

// Per-BlkLen tile-grid dispatcher. Splits (CountM, CountN) into up to 6
// sub-rectangles and dispatches one tile-shape instantiation per piece.
template <size_t BlkLen>
size_t
SQ2BitGemm_Dispatch_NeonDotProd(
    const std::byte* QuantA, const float* QuantAScale,
    const std::byte* QuantBData, const float* QuantBScale,
    float* C, size_t CountM, size_t CountN, size_t BlockCountK,
    const float* Bias, size_t ldc,
    const float* ABlockSum, const float* QuantBBlkSum)
{
    if (BlockCountK == 0) return 0;
    if (CountM == 0 || CountN == 0) return CountM;

    const size_t NMain8 = (CountN / 8) * 8;
    const size_t NRem   = CountN - NMain8;
    const size_t NMain4 = (NRem / 4) * 4;    // 0 or 4
    const size_t NTail  = NRem - NMain4;     // 0..3
    const size_t M_main = (CountM / 2) * 2;
    const size_t M_tail = CountM - M_main;
    const size_t n_tail_start = NMain8 + NMain4;

    if (NMain8 > 0) {
        if (M_main > 0)
            Q2Int8GemmRxC_DotProd<BlkLen, 2, 8>(
                QuantA, QuantAScale, QuantBData, QuantBScale, C,
                CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum,
                0, M_main, 0, NMain8);
        if (M_tail > 0)
            Q2Int8GemmRxC_DotProd<BlkLen, 1, 8>(
                QuantA, QuantAScale, QuantBData, QuantBScale, C,
                CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum,
                M_main, M_tail, 0, NMain8);
    }
    if (NMain4 > 0) {
        if (M_main > 0)
            Q2Int8GemmRxC_DotProd<BlkLen, 2, 4>(
                QuantA, QuantAScale, QuantBData, QuantBScale, C,
                CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum,
                0, M_main, NMain8, NMain4);
        if (M_tail > 0)
            Q2Int8GemmRxC_DotProd<BlkLen, 1, 4>(
                QuantA, QuantAScale, QuantBData, QuantBScale, C,
                CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum,
                M_main, M_tail, NMain8, NMain4);
    }
    if (NTail > 0) {
        if (M_main > 0)
            Q2Int8GemmRxC_DotProd<BlkLen, 2, 1>(
                QuantA, QuantAScale, QuantBData, QuantBScale, C,
                CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum,
                0, M_main, n_tail_start, NTail);
        if (M_tail > 0)
            Q2Int8GemmRxC_DotProd<BlkLen, 1, 1>(
                QuantA, QuantAScale, QuantBData, QuantBScale, C,
                CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum,
                M_main, M_tail, n_tail_start, NTail);
    }

    return CountM;
}

// Thin per-BlkLen public entry. Pins BlkLen at compile time so the templated
// tile kernel constant-folds the BlkLen-specific layout.
size_t
SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd_BlkLen64(
    const std::byte* QuantA, const float* QuantAScale,
    const std::byte* QuantBData, const float* QuantBScale,
    float* C, size_t CountM, size_t CountN, size_t BlockCountK,
    const float* Bias, size_t ldc,
    const float* ABlockSum, const float* QuantBBlkSum)
{
    return SQ2BitGemm_Dispatch_NeonDotProd<64>(
        QuantA, QuantAScale, QuantBData, QuantBScale, C, CountM, CountN,
        BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

size_t
SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd_BlkLen128(
    const std::byte* QuantA, const float* QuantAScale,
    const std::byte* QuantBData, const float* QuantBScale,
    float* C, size_t CountM, size_t CountN, size_t BlockCountK,
    const float* Bias, size_t ldc,
    const float* ABlockSum, const float* QuantBBlkSum)
{
    return SQ2BitGemm_Dispatch_NeonDotProd<128>(
        QuantA, QuantAScale, QuantBData, QuantBScale, C, CountM, CountN,
        BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}

size_t
SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd_BlkLen32(
    const std::byte* QuantA, const float* QuantAScale,
    const std::byte* QuantBData, const float* QuantBScale,
    float* C, size_t CountM, size_t CountN, size_t BlockCountK,
    const float* Bias, size_t ldc,
    const float* ABlockSum, const float* QuantBBlkSum)
{
    return SQ2BitGemm_Dispatch_NeonDotProd<32>(
        QuantA, QuantAScale, QuantBData, QuantBScale, C, CountM, CountN,
        BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
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
