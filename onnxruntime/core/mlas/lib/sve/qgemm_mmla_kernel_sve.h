/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_mmla_kernel_sve.h

Abstract:

    Shared SVE i8mm compute core for the int8 QGEMM kernels. One templated
    routine covers both the signed (S8S8, svmmla_s32) and unsigned-A / signed-B
    (U8S8, svusmmla_s32) cases; the two MLAS kernel-type wrappers
    (qgemm_kernel_smmla_sve.cpp and qgemm_kernel_ummla_sve.cpp) both delegate
    here, so there is a single optimized implementation.

    Contract (identical to the NEON smmla/ummla kernels): the caller supplies
    packed A and packed B in the 8-wide, PackedK=8 MLAS layout, plus RowSum /
    ColumnSum / (optional per-column) ZeroPointB. This routine computes

        C[i][j] = RowSum[i]*ZeroPointB[j] + ColumnSum[j] + sum_k A[i][k]*B[j][k]

    (ZeroPointB[j] == 1 when the pointer is null / per-tensor). Integer math is
    exact, so the result is bit-identical to the NEON kernels.

    Performance shape (transliterated from aarch64/QgemmS8S8KernelSmmla.S): for an
    8-row group all 16 accumulators (4 row-pairs x 4 col-pairs) are kept resident
    so the packed B panel is read ONCE per K-block, and the 2x2 MMLA tiles are
    deinterleaved to row-major with svuzp1/svuzp2 (no scalar round-trip). Each
    svmmla consumes one 128-bit segment; loads are predicated to 16 bytes so the
    kernel is correct at any SVE vector length (uses only the first segment).

--*/

#pragma once

#include <algorithm>
#include <arm_sve.h>

//
// MMLA instruction select, matching the NEON kernels this shares packing with:
//   - S8S8 (AUnsigned=false): signed x signed -> svmmla_s32 (like NEON `smmla`).
//   - U8X8 (AUnsigned=true):  MLAS packs A unsigned and bit-flips B to unsigned
//     (the ummla CopyPackB XORs B with 0x80 and fixes up ZeroPointB), so both
//     operands are unsigned -> svmmla_u32 (like NEON `ummla`, unsigned x
//     unsigned). The u32 accumulator is bit-identical to the s32 seed/store
//     (two's-complement add), so we just reinterpret.
// A and B are loaded as raw bytes (svint8_t) and reinterpreted as needed.
//
template <bool AUnsigned>
static MLAS_FORCEINLINE svint32_t
MlasQGemmMmlaSve(svint32_t acc, svint8_t a, svint8_t b)
{
    if constexpr (AUnsigned) {
        return svreinterpret_s32_u32(
            svmmla_u32(svreinterpret_u32_s32(acc),
                       svreinterpret_u8_s8(a), svreinterpret_u8_s8(b)));
    } else {
        return svmmla_s32(acc, a, b);
    }
}

//
// Build the seed for one 2x2 col-pair tile: [r0c0, r0c1, r1c0, r1c1] with
// value RowSum[r]*ZeroPointB[c] + ColumnSum[c]. c0 is the pair's first column.
//
static MLAS_FORCEINLINE svint32_t
MlasQGemmSeedTileColSve(size_t r0, size_t r1, size_t Rows, size_t c0,
                        const int32_t* RowSumBuffer, const int32_t* ColumnSumBuffer,
                        const int32_t* ZeroPointB, svbool_t pgTile)
{
    const int32_t rs0 = RowSumBuffer[r0];
    const int32_t rs1 = (r1 < Rows) ? RowSumBuffer[r1] : 0;
    const size_t c1 = c0 + 1;
    const int32_t zpb0 = ZeroPointB ? ZeroPointB[c0] : 1;
    const int32_t zpb1 = ZeroPointB ? ZeroPointB[c1] : 1;
    const int32_t cs0 = ColumnSumBuffer[c0];
    const int32_t cs1 = ColumnSumBuffer[c1];
    const int32_t init[4] = {
        rs0 * zpb0 + cs0,
        rs0 * zpb1 + cs1,
        rs1 * zpb0 + cs0,
        rs1 * zpb1 + cs1,
    };
    return svld1_s32(pgTile, init);
}

static MLAS_FORCEINLINE svint32_t
MlasQGemmSeedTileSve(size_t r0, size_t r1, size_t Rows, size_t nn, size_t j,
                     const int32_t* RowSumBuffer, const int32_t* ColumnSumBuffer,
                     const int32_t* ZeroPointB, svbool_t pgTile)
{
    return MlasQGemmSeedTileColSve(r0, r1, Rows, nn + 2 * j,
                                   RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
}

//
// Store one output row's up-to-8 columns. At VL==128 an svint32_t holds 4
// lanes, so the row is written as two 4-column halves under a length predicate.
// `lo` holds columns 0..3, `hi` holds columns 4..7.
//
static MLAS_FORCEINLINE void
MlasQGemmStoreRowSve(int32_t* dst, size_t ColsThis, bool ZeroMode, svint32_t lo, svint32_t hi)
{
    {
        const size_t cols = std::min<size_t>(4, ColsThis);
        const svbool_t pg = svwhilelt_b32_u64(uint64_t(0), uint64_t(cols));
        svint32_t v = lo;
        if (!ZeroMode) {
            v = svadd_s32_x(pg, v, svld1_s32(pg, dst));
        }
        svst1_s32(pg, dst, v);
    }
    if (ColsThis > 4) {
        const size_t cols = ColsThis - 4;
        const svbool_t pg = svwhilelt_b32_u64(uint64_t(0), uint64_t(cols));
        svint32_t v = hi;
        if (!ZeroMode) {
            v = svadd_s32_x(pg, v, svld1_s32(pg, dst + 4));
        }
        svst1_s32(pg, dst + 4, v);
    }
}

//
// Deinterleave a row-pair's four 2x2 col-pair tiles (t0..t3 for col-pairs 0..3)
// into row-major and store rows r0 (and r1 when valid). Mirrors the NEON
// `uzp1/uzp2 .2d` output stage: at VL==128 the 128-bit view is 2x int64, so
// uzp1(int64) gathers the r0 halves and uzp2 the r1 halves.
//
static MLAS_FORCEINLINE void
MlasQGemmStoreRowPairSve(int32_t* C, size_t ldc, size_t r0, size_t r1, size_t Rows,
                         size_t nn, size_t ColsThis, bool ZeroMode,
                         svint32_t t0, svint32_t t1, svint32_t t2, svint32_t t3)
{
    const svuint64_t u0 = svreinterpret_u64_s32(t0);
    const svuint64_t u1 = svreinterpret_u64_s32(t1);
    const svuint64_t u2 = svreinterpret_u64_s32(t2);
    const svuint64_t u3 = svreinterpret_u64_s32(t3);

    const svint32_t r0lo = svreinterpret_s32_u64(svuzp1_u64(u0, u1));  // cols 0..3
    const svint32_t r0hi = svreinterpret_s32_u64(svuzp1_u64(u2, u3));  // cols 4..7
    const svint32_t r1lo = svreinterpret_s32_u64(svuzp2_u64(u0, u1));
    const svint32_t r1hi = svreinterpret_s32_u64(svuzp2_u64(u2, u3));

    MlasQGemmStoreRowSve(C + r0 * ldc + nn, ColsThis, ZeroMode, r0lo, r0hi);
    if (r1 < Rows) {
        MlasQGemmStoreRowSve(C + r1 * ldc + nn, ColsThis, ZeroMode, r1lo, r1hi);
    }
}

//
// Shared int8 QGEMM inner kernel. AUnsigned selects svusmmla (U8S8) vs svmmla
// (S8S8). Returns the number of rows handled (8/4/2/1), matching the driver's
// packed-A advance contract.
//
template <bool AUnsigned>
static MLAS_FORCEINLINE size_t
MlasQGemmMmlaKernelSve(const uint8_t* A,
                       const uint8_t* B,
                       int32_t* C,
                       size_t PackedCountK,
                       size_t CountM,
                       size_t CountN,
                       size_t ldc,
                       const int32_t* RowSumBuffer,
                       const int32_t* ColumnSumBuffer,
                       const int32_t* ZeroPointB,
                       bool ZeroMode)
{
    const size_t Rows =
#if defined(MLAS_SVE_QGEMM_TILE_12X8) && MLAS_SVE_QGEMM_TILE_12X8
        // 12 rows are consumable only when CopyPackA's greedy 8/4/2/1 grouping
        // emitted an [8-group][4-group] pair, i.e. exactly when 12 <= CountM < 16.
        // (With Strides.M == 12 the driver never sends more than 12.)
        (CountM >= 12 && CountM < 16) ? 12 :
#endif
        (CountM >= 8) ? 8 : (CountM >= 4) ? 4 : (CountM >= 2) ? 2 : 1;

    const svbool_t pg16 = svptrue_pat_b8(SV_VL16);
    const svbool_t pgTile = svptrue_pat_b32(SV_VL4);

    const int8_t* PackedA = reinterpret_cast<const int8_t*>(A);
    const int8_t* PackedB = reinterpret_cast<const int8_t*>(B);
    const size_t BGroupStride = PackedCountK * 64;

#if defined(MLAS_SVE_QGEMM_TILE_12X8) && MLAS_SVE_QGEMM_TILE_12X8
    if (Rows == 12) {
        //
        // EXPERIMENT (taller register blocking): 12x8 tile = 24 resident
        // accumulators (6 row-pairs x 4 col-pairs), 24 svmmla per K-block
        // against 10 operand loads (4 B kept live, 6 A streamed) = 2.4
        // MMLA/load. Each invocation covers 12 output rows, so an M-stripe
        // takes fewer full sweeps of the packed-B panel than the 8-row tile.
        //
        // Packed A for CountM in [12,16) is an 8-group followed by a 4-group:
        //   row-pairs 0-3: A8 + kb*64 + ip*16        (8-group, 64 B/K-block)
        //   row-pairs 4-5: A4 + kb*32 + (ip-4)*16    (4-group, 32 B/K-block)
        //
        const int8_t* A8 = PackedA;
        const int8_t* A4 = PackedA + PackedCountK * 64;

        size_t g = 0;
        for (size_t nn = 0; nn < CountN; nn += 8, ++g) {
            const size_t ColsThis = std::min<size_t>(8, CountN - nn);
            const int8_t* Bgroup = PackedB + g * BGroupStride;

            svint32_t v00 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v01 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v02 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v03 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v10 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v11 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v12 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v13 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v20 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v21 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v22 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v23 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v30 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v31 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v32 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v33 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v40 = MlasQGemmSeedTileSve(8, 9, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v41 = MlasQGemmSeedTileSve(8, 9, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v42 = MlasQGemmSeedTileSve(8, 9, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v43 = MlasQGemmSeedTileSve(8, 9, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v50 = MlasQGemmSeedTileSve(10, 11, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v51 = MlasQGemmSeedTileSve(10, 11, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v52 = MlasQGemmSeedTileSve(10, 11, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t v53 = MlasQGemmSeedTileSve(10, 11, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);

            //
            // One K-block: 4 B loads kept live + 6 streamed A row-pair vectors
            // (each loaded immediately before its 4 uses) feeding the 24
            // resident accumulators. Expressed as a macro so the K-unroll
            // variant replicates the exact straight-line pattern the register
            // allocator already handles without spills.
            //
#if defined(MLAS_SVE_QGEMM_PREFETCH_B) && MLAS_SVE_QGEMM_PREFETCH_B
#define MLAS_QGEMM_12X8_PF(KB) __builtin_prefetch(Bgroup + ((KB) + 1) * 64, 0, 3);
#else
#define MLAS_QGEMM_12X8_PF(KB)
#endif
#define MLAS_QGEMM_12X8_KBLOCK(KB)                                            \
            {                                                                 \
                const size_t kb_ = (KB);                                      \
                const int8_t* bptr = Bgroup + kb_ * 64;                       \
                MLAS_QGEMM_12X8_PF(kb_)                                       \
                const svint8_t b0 = svld1_s8(pg16, bptr + 0);                 \
                const svint8_t b1 = svld1_s8(pg16, bptr + 16);                \
                const svint8_t b2 = svld1_s8(pg16, bptr + 32);                \
                const svint8_t b3 = svld1_s8(pg16, bptr + 48);                \
                {                                                             \
                    const svint8_t a = svld1_s8(pg16, A8 + kb_ * 64 + 0);     \
                    v00 = MlasQGemmMmlaSve<AUnsigned>(v00, a, b0);            \
                    v01 = MlasQGemmMmlaSve<AUnsigned>(v01, a, b1);            \
                    v02 = MlasQGemmMmlaSve<AUnsigned>(v02, a, b2);            \
                    v03 = MlasQGemmMmlaSve<AUnsigned>(v03, a, b3);            \
                }                                                             \
                {                                                             \
                    const svint8_t a = svld1_s8(pg16, A8 + kb_ * 64 + 16);    \
                    v10 = MlasQGemmMmlaSve<AUnsigned>(v10, a, b0);            \
                    v11 = MlasQGemmMmlaSve<AUnsigned>(v11, a, b1);            \
                    v12 = MlasQGemmMmlaSve<AUnsigned>(v12, a, b2);            \
                    v13 = MlasQGemmMmlaSve<AUnsigned>(v13, a, b3);            \
                }                                                             \
                {                                                             \
                    const svint8_t a = svld1_s8(pg16, A8 + kb_ * 64 + 32);    \
                    v20 = MlasQGemmMmlaSve<AUnsigned>(v20, a, b0);            \
                    v21 = MlasQGemmMmlaSve<AUnsigned>(v21, a, b1);            \
                    v22 = MlasQGemmMmlaSve<AUnsigned>(v22, a, b2);            \
                    v23 = MlasQGemmMmlaSve<AUnsigned>(v23, a, b3);            \
                }                                                             \
                {                                                             \
                    const svint8_t a = svld1_s8(pg16, A8 + kb_ * 64 + 48);    \
                    v30 = MlasQGemmMmlaSve<AUnsigned>(v30, a, b0);            \
                    v31 = MlasQGemmMmlaSve<AUnsigned>(v31, a, b1);            \
                    v32 = MlasQGemmMmlaSve<AUnsigned>(v32, a, b2);            \
                    v33 = MlasQGemmMmlaSve<AUnsigned>(v33, a, b3);            \
                }                                                             \
                {                                                             \
                    const svint8_t a = svld1_s8(pg16, A4 + kb_ * 32 + 0);     \
                    v40 = MlasQGemmMmlaSve<AUnsigned>(v40, a, b0);            \
                    v41 = MlasQGemmMmlaSve<AUnsigned>(v41, a, b1);            \
                    v42 = MlasQGemmMmlaSve<AUnsigned>(v42, a, b2);            \
                    v43 = MlasQGemmMmlaSve<AUnsigned>(v43, a, b3);            \
                }                                                             \
                {                                                             \
                    const svint8_t a = svld1_s8(pg16, A4 + kb_ * 32 + 16);    \
                    v50 = MlasQGemmMmlaSve<AUnsigned>(v50, a, b0);            \
                    v51 = MlasQGemmMmlaSve<AUnsigned>(v51, a, b1);            \
                    v52 = MlasQGemmMmlaSve<AUnsigned>(v52, a, b2);            \
                    v53 = MlasQGemmMmlaSve<AUnsigned>(v53, a, b3);            \
                }                                                             \
            }

#if defined(MLAS_SVE_QGEMM_K_UNROLL2) && MLAS_SVE_QGEMM_K_UNROLL2
            size_t kb = 0;
            for (; kb + 2 <= PackedCountK; kb += 2) {
                MLAS_QGEMM_12X8_KBLOCK(kb)
                MLAS_QGEMM_12X8_KBLOCK(kb + 1)
            }
            if (kb < PackedCountK) {
                MLAS_QGEMM_12X8_KBLOCK(kb)
            }
#else
            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                MLAS_QGEMM_12X8_KBLOCK(kb)
            }
#endif
#undef MLAS_QGEMM_12X8_KBLOCK
#undef MLAS_QGEMM_12X8_PF

            MlasQGemmStoreRowPairSve(C, ldc, 0, 1, Rows, nn, ColsThis, ZeroMode, v00, v01, v02, v03);
            MlasQGemmStoreRowPairSve(C, ldc, 2, 3, Rows, nn, ColsThis, ZeroMode, v10, v11, v12, v13);
            MlasQGemmStoreRowPairSve(C, ldc, 4, 5, Rows, nn, ColsThis, ZeroMode, v20, v21, v22, v23);
            MlasQGemmStoreRowPairSve(C, ldc, 6, 7, Rows, nn, ColsThis, ZeroMode, v30, v31, v32, v33);
            MlasQGemmStoreRowPairSve(C, ldc, 8, 9, Rows, nn, ColsThis, ZeroMode, v40, v41, v42, v43);
            MlasQGemmStoreRowPairSve(C, ldc, 10, 11, Rows, nn, ColsThis, ZeroMode, v50, v51, v52, v53);
        }
        return Rows;
    }
#endif  // MLAS_SVE_QGEMM_TILE_12X8

    if (Rows == 8) {
        const size_t AStride = 64;  // 4 row-pairs * 16 bytes per K-block
#if defined(MLAS_SVE_QGEMM_WIDE_TILE) && MLAS_SVE_QGEMM_WIDE_TILE
        //
        // EXPERIMENT (VL=128 register-blocking ceiling): 8x16 tile = 32 resident
        // accumulators (4 row-pairs x 8 col-pairs), reading TWO 8-column B groups
        // per K-block. 32 accumulators + up to 12 operand loads far exceed the 32
        // physical Z-registers, so this deliberately spills. Columns < 16 fall
        // back to the 8x8 loop. Purpose: measure spills + throughput vs the 8x8
        // tile at VL=128 (expected: no speedup).
        //
        size_t nn = 0;
        size_t g = 0;
        for (; nn + 16 <= CountN; nn += 16, g += 2) {
            const int8_t* Bg0 = PackedB + g * BGroupStride;
            const int8_t* Bg1 = PackedB + (g + 1) * BGroupStride;

            svint32_t w00 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w01 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w02 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w03 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w04 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 4, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w05 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 5, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w06 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 6, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w07 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 7, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w10 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w11 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w12 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w13 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w14 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 4, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w15 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 5, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w16 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 6, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w17 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 7, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w20 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w21 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w22 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w23 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w24 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 4, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w25 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 5, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w26 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 6, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w27 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 7, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w30 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w31 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w32 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w33 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w34 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 4, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w35 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 5, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w36 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 6, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w37 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 7, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);

            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                const int8_t* b0p = Bg0 + kb * 64;
                const int8_t* b1p = Bg1 + kb * 64;
                const svint8_t b0 = svld1_s8(pg16, b0p + 0);
                const svint8_t b1 = svld1_s8(pg16, b0p + 16);
                const svint8_t b2 = svld1_s8(pg16, b0p + 32);
                const svint8_t b3 = svld1_s8(pg16, b0p + 48);
                const svint8_t b4 = svld1_s8(pg16, b1p + 0);
                const svint8_t b5 = svld1_s8(pg16, b1p + 16);
                const svint8_t b6 = svld1_s8(pg16, b1p + 32);
                const svint8_t b7 = svld1_s8(pg16, b1p + 48);

                const int8_t* aptr = PackedA + kb * AStride;
                const svint8_t a0 = svld1_s8(pg16, aptr + 0);
                const svint8_t a1 = svld1_s8(pg16, aptr + 16);
                const svint8_t a2 = svld1_s8(pg16, aptr + 32);
                const svint8_t a3 = svld1_s8(pg16, aptr + 48);

                w00 = MlasQGemmMmlaSve<AUnsigned>(w00, a0, b0);
                w01 = MlasQGemmMmlaSve<AUnsigned>(w01, a0, b1);
                w02 = MlasQGemmMmlaSve<AUnsigned>(w02, a0, b2);
                w03 = MlasQGemmMmlaSve<AUnsigned>(w03, a0, b3);
                w04 = MlasQGemmMmlaSve<AUnsigned>(w04, a0, b4);
                w05 = MlasQGemmMmlaSve<AUnsigned>(w05, a0, b5);
                w06 = MlasQGemmMmlaSve<AUnsigned>(w06, a0, b6);
                w07 = MlasQGemmMmlaSve<AUnsigned>(w07, a0, b7);
                w10 = MlasQGemmMmlaSve<AUnsigned>(w10, a1, b0);
                w11 = MlasQGemmMmlaSve<AUnsigned>(w11, a1, b1);
                w12 = MlasQGemmMmlaSve<AUnsigned>(w12, a1, b2);
                w13 = MlasQGemmMmlaSve<AUnsigned>(w13, a1, b3);
                w14 = MlasQGemmMmlaSve<AUnsigned>(w14, a1, b4);
                w15 = MlasQGemmMmlaSve<AUnsigned>(w15, a1, b5);
                w16 = MlasQGemmMmlaSve<AUnsigned>(w16, a1, b6);
                w17 = MlasQGemmMmlaSve<AUnsigned>(w17, a1, b7);
                w20 = MlasQGemmMmlaSve<AUnsigned>(w20, a2, b0);
                w21 = MlasQGemmMmlaSve<AUnsigned>(w21, a2, b1);
                w22 = MlasQGemmMmlaSve<AUnsigned>(w22, a2, b2);
                w23 = MlasQGemmMmlaSve<AUnsigned>(w23, a2, b3);
                w24 = MlasQGemmMmlaSve<AUnsigned>(w24, a2, b4);
                w25 = MlasQGemmMmlaSve<AUnsigned>(w25, a2, b5);
                w26 = MlasQGemmMmlaSve<AUnsigned>(w26, a2, b6);
                w27 = MlasQGemmMmlaSve<AUnsigned>(w27, a2, b7);
                w30 = MlasQGemmMmlaSve<AUnsigned>(w30, a3, b0);
                w31 = MlasQGemmMmlaSve<AUnsigned>(w31, a3, b1);
                w32 = MlasQGemmMmlaSve<AUnsigned>(w32, a3, b2);
                w33 = MlasQGemmMmlaSve<AUnsigned>(w33, a3, b3);
                w34 = MlasQGemmMmlaSve<AUnsigned>(w34, a3, b4);
                w35 = MlasQGemmMmlaSve<AUnsigned>(w35, a3, b5);
                w36 = MlasQGemmMmlaSve<AUnsigned>(w36, a3, b6);
                w37 = MlasQGemmMmlaSve<AUnsigned>(w37, a3, b7);
            }

            // Store 8 rows x 16 cols: each row-pair writes cols nn..nn+7 then nn+8..nn+15.
            MlasQGemmStoreRowPairSve(C, ldc, 0, 1, Rows, nn, 8, ZeroMode, w00, w01, w02, w03);
            MlasQGemmStoreRowPairSve(C, ldc, 0, 1, Rows, nn + 8, 8, ZeroMode, w04, w05, w06, w07);
            MlasQGemmStoreRowPairSve(C, ldc, 2, 3, Rows, nn, 8, ZeroMode, w10, w11, w12, w13);
            MlasQGemmStoreRowPairSve(C, ldc, 2, 3, Rows, nn + 8, 8, ZeroMode, w14, w15, w16, w17);
            MlasQGemmStoreRowPairSve(C, ldc, 4, 5, Rows, nn, 8, ZeroMode, w20, w21, w22, w23);
            MlasQGemmStoreRowPairSve(C, ldc, 4, 5, Rows, nn + 8, 8, ZeroMode, w24, w25, w26, w27);
            MlasQGemmStoreRowPairSve(C, ldc, 6, 7, Rows, nn, 8, ZeroMode, w30, w31, w32, w33);
            MlasQGemmStoreRowPairSve(C, ldc, 6, 7, Rows, nn + 8, 8, ZeroMode, w34, w35, w36, w37);
        }
        // Remainder columns (< 16): plain 8x8 col-groups.
        for (; nn < CountN; nn += 8, ++g) {
            const size_t ColsThis = std::min<size_t>(8, CountN - nn);
            const int8_t* Bgroup = PackedB + g * BGroupStride;

            svint32_t acc0 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc1 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc2 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc3 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc4 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc5 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc6 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc7 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc8 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc9 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc10 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc11 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc12 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc13 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc14 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc15 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);

            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                const int8_t* bptr = Bgroup + kb * 64;
                const svint8_t b0 = svld1_s8(pg16, bptr + 0);
                const svint8_t b1 = svld1_s8(pg16, bptr + 16);
                const svint8_t b2 = svld1_s8(pg16, bptr + 32);
                const svint8_t b3 = svld1_s8(pg16, bptr + 48);
                const int8_t* aptr = PackedA + kb * AStride;
                const svint8_t a0 = svld1_s8(pg16, aptr + 0);
                const svint8_t a1 = svld1_s8(pg16, aptr + 16);
                const svint8_t a2 = svld1_s8(pg16, aptr + 32);
                const svint8_t a3 = svld1_s8(pg16, aptr + 48);
                acc0 = MlasQGemmMmlaSve<AUnsigned>(acc0, a0, b0);
                acc1 = MlasQGemmMmlaSve<AUnsigned>(acc1, a0, b1);
                acc2 = MlasQGemmMmlaSve<AUnsigned>(acc2, a0, b2);
                acc3 = MlasQGemmMmlaSve<AUnsigned>(acc3, a0, b3);
                acc4 = MlasQGemmMmlaSve<AUnsigned>(acc4, a1, b0);
                acc5 = MlasQGemmMmlaSve<AUnsigned>(acc5, a1, b1);
                acc6 = MlasQGemmMmlaSve<AUnsigned>(acc6, a1, b2);
                acc7 = MlasQGemmMmlaSve<AUnsigned>(acc7, a1, b3);
                acc8 = MlasQGemmMmlaSve<AUnsigned>(acc8, a2, b0);
                acc9 = MlasQGemmMmlaSve<AUnsigned>(acc9, a2, b1);
                acc10 = MlasQGemmMmlaSve<AUnsigned>(acc10, a2, b2);
                acc11 = MlasQGemmMmlaSve<AUnsigned>(acc11, a2, b3);
                acc12 = MlasQGemmMmlaSve<AUnsigned>(acc12, a3, b0);
                acc13 = MlasQGemmMmlaSve<AUnsigned>(acc13, a3, b1);
                acc14 = MlasQGemmMmlaSve<AUnsigned>(acc14, a3, b2);
                acc15 = MlasQGemmMmlaSve<AUnsigned>(acc15, a3, b3);
            }

            MlasQGemmStoreRowPairSve(C, ldc, 0, 1, Rows, nn, ColsThis, ZeroMode, acc0, acc1, acc2, acc3);
            MlasQGemmStoreRowPairSve(C, ldc, 2, 3, Rows, nn, ColsThis, ZeroMode, acc4, acc5, acc6, acc7);
            MlasQGemmStoreRowPairSve(C, ldc, 4, 5, Rows, nn, ColsThis, ZeroMode, acc8, acc9, acc10, acc11);
            MlasQGemmStoreRowPairSve(C, ldc, 6, 7, Rows, nn, ColsThis, ZeroMode, acc12, acc13, acc14, acc15);
        }
        return Rows;
#elif defined(MLAS_SVE_QGEMM_TILE_8X12) && MLAS_SVE_QGEMM_TILE_8X12
        //
        // EXPERIMENT (intermediate register blocking): 8x12 tile = 24 resident
        // accumulators (4 row-pairs x 6 col-pairs). Per K-block: 10 operand
        // loads (4 A + 6 B) feed 24 svmmla = 2.4 MMLA/load, vs 2.0 for the 8x8
        // tile. 24 accumulators + 4 A + one streamed B leave peak live pressure
        // at ~29-31 of the 32 Z-registers, unlike the 8x16/32-accumulator
        // variant which could not fit and was split by the compiler.
        //
        // A 12-wide tile spans the 8-column packed-B groups, so each col-pair's
        // kb-invariant base pointer is computed generically from its column.
        // The <12-column tail (possibly mid-group) uses the same generic
        // addressing with out-of-range pairs clamped to the last valid pair
        // (their lanes are excluded by the store predicates).
        //
        const size_t LastPairCol = (CountN - 1) & ~size_t(1);
        auto PairBase = [&](size_t col) {
            return PackedB + (col / 8) * BGroupStride + (col % 8) * 8;
        };

        size_t nn = 0;
        for (; nn + 12 <= CountN; nn += 12) {
            const int8_t* bp0 = PairBase(nn + 0);
            const int8_t* bp1 = PairBase(nn + 2);
            const int8_t* bp2 = PairBase(nn + 4);
            const int8_t* bp3 = PairBase(nn + 6);
            const int8_t* bp4 = PairBase(nn + 8);
            const int8_t* bp5 = PairBase(nn + 10);

            svint32_t w00 = MlasQGemmSeedTileColSve(0, 1, Rows, nn + 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w01 = MlasQGemmSeedTileColSve(0, 1, Rows, nn + 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w02 = MlasQGemmSeedTileColSve(0, 1, Rows, nn + 4, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w03 = MlasQGemmSeedTileColSve(0, 1, Rows, nn + 6, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w04 = MlasQGemmSeedTileColSve(0, 1, Rows, nn + 8, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w05 = MlasQGemmSeedTileColSve(0, 1, Rows, nn + 10, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w10 = MlasQGemmSeedTileColSve(2, 3, Rows, nn + 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w11 = MlasQGemmSeedTileColSve(2, 3, Rows, nn + 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w12 = MlasQGemmSeedTileColSve(2, 3, Rows, nn + 4, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w13 = MlasQGemmSeedTileColSve(2, 3, Rows, nn + 6, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w14 = MlasQGemmSeedTileColSve(2, 3, Rows, nn + 8, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w15 = MlasQGemmSeedTileColSve(2, 3, Rows, nn + 10, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w20 = MlasQGemmSeedTileColSve(4, 5, Rows, nn + 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w21 = MlasQGemmSeedTileColSve(4, 5, Rows, nn + 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w22 = MlasQGemmSeedTileColSve(4, 5, Rows, nn + 4, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w23 = MlasQGemmSeedTileColSve(4, 5, Rows, nn + 6, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w24 = MlasQGemmSeedTileColSve(4, 5, Rows, nn + 8, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w25 = MlasQGemmSeedTileColSve(4, 5, Rows, nn + 10, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w30 = MlasQGemmSeedTileColSve(6, 7, Rows, nn + 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w31 = MlasQGemmSeedTileColSve(6, 7, Rows, nn + 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w32 = MlasQGemmSeedTileColSve(6, 7, Rows, nn + 4, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w33 = MlasQGemmSeedTileColSve(6, 7, Rows, nn + 6, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w34 = MlasQGemmSeedTileColSve(6, 7, Rows, nn + 8, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t w35 = MlasQGemmSeedTileColSve(6, 7, Rows, nn + 10, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);

            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                const int8_t* aptr = PackedA + kb * AStride;
                const svint8_t a0 = svld1_s8(pg16, aptr + 0);
                const svint8_t a1 = svld1_s8(pg16, aptr + 16);
                const svint8_t a2 = svld1_s8(pg16, aptr + 32);
                const svint8_t a3 = svld1_s8(pg16, aptr + 48);

                // Each B col-pair is loaded immediately before its 4 uses to
                // keep its live range short (24 accumulators stay resident).
                {
                    const svint8_t b = svld1_s8(pg16, bp0 + kb * 64);
                    w00 = MlasQGemmMmlaSve<AUnsigned>(w00, a0, b);
                    w10 = MlasQGemmMmlaSve<AUnsigned>(w10, a1, b);
                    w20 = MlasQGemmMmlaSve<AUnsigned>(w20, a2, b);
                    w30 = MlasQGemmMmlaSve<AUnsigned>(w30, a3, b);
                }
                {
                    const svint8_t b = svld1_s8(pg16, bp1 + kb * 64);
                    w01 = MlasQGemmMmlaSve<AUnsigned>(w01, a0, b);
                    w11 = MlasQGemmMmlaSve<AUnsigned>(w11, a1, b);
                    w21 = MlasQGemmMmlaSve<AUnsigned>(w21, a2, b);
                    w31 = MlasQGemmMmlaSve<AUnsigned>(w31, a3, b);
                }
                {
                    const svint8_t b = svld1_s8(pg16, bp2 + kb * 64);
                    w02 = MlasQGemmMmlaSve<AUnsigned>(w02, a0, b);
                    w12 = MlasQGemmMmlaSve<AUnsigned>(w12, a1, b);
                    w22 = MlasQGemmMmlaSve<AUnsigned>(w22, a2, b);
                    w32 = MlasQGemmMmlaSve<AUnsigned>(w32, a3, b);
                }
                {
                    const svint8_t b = svld1_s8(pg16, bp3 + kb * 64);
                    w03 = MlasQGemmMmlaSve<AUnsigned>(w03, a0, b);
                    w13 = MlasQGemmMmlaSve<AUnsigned>(w13, a1, b);
                    w23 = MlasQGemmMmlaSve<AUnsigned>(w23, a2, b);
                    w33 = MlasQGemmMmlaSve<AUnsigned>(w33, a3, b);
                }
                {
                    const svint8_t b = svld1_s8(pg16, bp4 + kb * 64);
                    w04 = MlasQGemmMmlaSve<AUnsigned>(w04, a0, b);
                    w14 = MlasQGemmMmlaSve<AUnsigned>(w14, a1, b);
                    w24 = MlasQGemmMmlaSve<AUnsigned>(w24, a2, b);
                    w34 = MlasQGemmMmlaSve<AUnsigned>(w34, a3, b);
                }
                {
                    const svint8_t b = svld1_s8(pg16, bp5 + kb * 64);
                    w05 = MlasQGemmMmlaSve<AUnsigned>(w05, a0, b);
                    w15 = MlasQGemmMmlaSve<AUnsigned>(w15, a1, b);
                    w25 = MlasQGemmMmlaSve<AUnsigned>(w25, a2, b);
                    w35 = MlasQGemmMmlaSve<AUnsigned>(w35, a3, b);
                }
            }

            // Cols nn..nn+7 from pairs 0-3, cols nn+8..nn+11 from pairs 4-5
            // (hi vector unused under the 4-column predicate).
            MlasQGemmStoreRowPairSve(C, ldc, 0, 1, Rows, nn, 8, ZeroMode, w00, w01, w02, w03);
            MlasQGemmStoreRowPairSve(C, ldc, 0, 1, Rows, nn + 8, 4, ZeroMode, w04, w05, w04, w05);
            MlasQGemmStoreRowPairSve(C, ldc, 2, 3, Rows, nn, 8, ZeroMode, w10, w11, w12, w13);
            MlasQGemmStoreRowPairSve(C, ldc, 2, 3, Rows, nn + 8, 4, ZeroMode, w14, w15, w14, w15);
            MlasQGemmStoreRowPairSve(C, ldc, 4, 5, Rows, nn, 8, ZeroMode, w20, w21, w22, w23);
            MlasQGemmStoreRowPairSve(C, ldc, 4, 5, Rows, nn + 8, 4, ZeroMode, w24, w25, w24, w25);
            MlasQGemmStoreRowPairSve(C, ldc, 6, 7, Rows, nn, 8, ZeroMode, w30, w31, w32, w33);
            MlasQGemmStoreRowPairSve(C, ldc, 6, 7, Rows, nn + 8, 4, ZeroMode, w34, w35, w34, w35);
        }

        // Tail (< 12 columns, nn possibly mid-group): 8x8-style iterations with
        // generic pair addressing. Pairs starting at or past CountN are clamped
        // to the last valid pair; their results are masked off by ColsThis.
        for (; nn < CountN; nn += 8) {
            const size_t ColsThis = std::min<size_t>(8, CountN - nn);
            const size_t c0 = std::min(nn + 0, LastPairCol);
            const size_t c1 = std::min(nn + 2, LastPairCol);
            const size_t c2 = std::min(nn + 4, LastPairCol);
            const size_t c3 = std::min(nn + 6, LastPairCol);
            const int8_t* bq0 = PairBase(c0);
            const int8_t* bq1 = PairBase(c1);
            const int8_t* bq2 = PairBase(c2);
            const int8_t* bq3 = PairBase(c3);

            svint32_t acc00 = MlasQGemmSeedTileColSve(0, 1, Rows, c0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc01 = MlasQGemmSeedTileColSve(0, 1, Rows, c1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc02 = MlasQGemmSeedTileColSve(0, 1, Rows, c2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc03 = MlasQGemmSeedTileColSve(0, 1, Rows, c3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc10 = MlasQGemmSeedTileColSve(2, 3, Rows, c0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc11 = MlasQGemmSeedTileColSve(2, 3, Rows, c1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc12 = MlasQGemmSeedTileColSve(2, 3, Rows, c2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc13 = MlasQGemmSeedTileColSve(2, 3, Rows, c3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc20 = MlasQGemmSeedTileColSve(4, 5, Rows, c0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc21 = MlasQGemmSeedTileColSve(4, 5, Rows, c1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc22 = MlasQGemmSeedTileColSve(4, 5, Rows, c2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc23 = MlasQGemmSeedTileColSve(4, 5, Rows, c3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc30 = MlasQGemmSeedTileColSve(6, 7, Rows, c0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc31 = MlasQGemmSeedTileColSve(6, 7, Rows, c1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc32 = MlasQGemmSeedTileColSve(6, 7, Rows, c2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc33 = MlasQGemmSeedTileColSve(6, 7, Rows, c3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);

            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                const svint8_t b0 = svld1_s8(pg16, bq0 + kb * 64);
                const svint8_t b1 = svld1_s8(pg16, bq1 + kb * 64);
                const svint8_t b2 = svld1_s8(pg16, bq2 + kb * 64);
                const svint8_t b3 = svld1_s8(pg16, bq3 + kb * 64);
                const int8_t* aptr = PackedA + kb * AStride;
                const svint8_t a0 = svld1_s8(pg16, aptr + 0);
                const svint8_t a1 = svld1_s8(pg16, aptr + 16);
                const svint8_t a2 = svld1_s8(pg16, aptr + 32);
                const svint8_t a3 = svld1_s8(pg16, aptr + 48);
                acc00 = MlasQGemmMmlaSve<AUnsigned>(acc00, a0, b0);
                acc01 = MlasQGemmMmlaSve<AUnsigned>(acc01, a0, b1);
                acc02 = MlasQGemmMmlaSve<AUnsigned>(acc02, a0, b2);
                acc03 = MlasQGemmMmlaSve<AUnsigned>(acc03, a0, b3);
                acc10 = MlasQGemmMmlaSve<AUnsigned>(acc10, a1, b0);
                acc11 = MlasQGemmMmlaSve<AUnsigned>(acc11, a1, b1);
                acc12 = MlasQGemmMmlaSve<AUnsigned>(acc12, a1, b2);
                acc13 = MlasQGemmMmlaSve<AUnsigned>(acc13, a1, b3);
                acc20 = MlasQGemmMmlaSve<AUnsigned>(acc20, a2, b0);
                acc21 = MlasQGemmMmlaSve<AUnsigned>(acc21, a2, b1);
                acc22 = MlasQGemmMmlaSve<AUnsigned>(acc22, a2, b2);
                acc23 = MlasQGemmMmlaSve<AUnsigned>(acc23, a2, b3);
                acc30 = MlasQGemmMmlaSve<AUnsigned>(acc30, a3, b0);
                acc31 = MlasQGemmMmlaSve<AUnsigned>(acc31, a3, b1);
                acc32 = MlasQGemmMmlaSve<AUnsigned>(acc32, a3, b2);
                acc33 = MlasQGemmMmlaSve<AUnsigned>(acc33, a3, b3);
            }

            MlasQGemmStoreRowPairSve(C, ldc, 0, 1, Rows, nn, ColsThis, ZeroMode, acc00, acc01, acc02, acc03);
            MlasQGemmStoreRowPairSve(C, ldc, 2, 3, Rows, nn, ColsThis, ZeroMode, acc10, acc11, acc12, acc13);
            MlasQGemmStoreRowPairSve(C, ldc, 4, 5, Rows, nn, ColsThis, ZeroMode, acc20, acc21, acc22, acc23);
            MlasQGemmStoreRowPairSve(C, ldc, 6, 7, Rows, nn, ColsThis, ZeroMode, acc30, acc31, acc32, acc33);
        }
        return Rows;
#else
        // Fast path: 16 resident accumulators, packed B read once per K-block.
        size_t g = 0;
        for (size_t nn = 0; nn < CountN; nn += 8, ++g) {
            const size_t ColsThis = std::min<size_t>(8, CountN - nn);
            const int8_t* Bgroup = PackedB + g * BGroupStride;

            svint32_t acc00 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc01 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc02 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc03 = MlasQGemmSeedTileSve(0, 1, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc10 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc11 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc12 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc13 = MlasQGemmSeedTileSve(2, 3, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc20 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc21 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc22 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc23 = MlasQGemmSeedTileSve(4, 5, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc30 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc31 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc32 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc33 = MlasQGemmSeedTileSve(6, 7, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);

            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                const int8_t* bptr = Bgroup + kb * 64;
                const svint8_t b0 = svld1_s8(pg16, bptr + 0);
                const svint8_t b1 = svld1_s8(pg16, bptr + 16);
                const svint8_t b2 = svld1_s8(pg16, bptr + 32);
                const svint8_t b3 = svld1_s8(pg16, bptr + 48);

                const int8_t* aptr = PackedA + kb * AStride;
                const svint8_t a0 = svld1_s8(pg16, aptr + 0);
                const svint8_t a1 = svld1_s8(pg16, aptr + 16);
                const svint8_t a2 = svld1_s8(pg16, aptr + 32);
                const svint8_t a3 = svld1_s8(pg16, aptr + 48);

                acc00 = MlasQGemmMmlaSve<AUnsigned>(acc00, a0, b0);
                acc01 = MlasQGemmMmlaSve<AUnsigned>(acc01, a0, b1);
                acc02 = MlasQGemmMmlaSve<AUnsigned>(acc02, a0, b2);
                acc03 = MlasQGemmMmlaSve<AUnsigned>(acc03, a0, b3);
                acc10 = MlasQGemmMmlaSve<AUnsigned>(acc10, a1, b0);
                acc11 = MlasQGemmMmlaSve<AUnsigned>(acc11, a1, b1);
                acc12 = MlasQGemmMmlaSve<AUnsigned>(acc12, a1, b2);
                acc13 = MlasQGemmMmlaSve<AUnsigned>(acc13, a1, b3);
                acc20 = MlasQGemmMmlaSve<AUnsigned>(acc20, a2, b0);
                acc21 = MlasQGemmMmlaSve<AUnsigned>(acc21, a2, b1);
                acc22 = MlasQGemmMmlaSve<AUnsigned>(acc22, a2, b2);
                acc23 = MlasQGemmMmlaSve<AUnsigned>(acc23, a2, b3);
                acc30 = MlasQGemmMmlaSve<AUnsigned>(acc30, a3, b0);
                acc31 = MlasQGemmMmlaSve<AUnsigned>(acc31, a3, b1);
                acc32 = MlasQGemmMmlaSve<AUnsigned>(acc32, a3, b2);
                acc33 = MlasQGemmMmlaSve<AUnsigned>(acc33, a3, b3);
            }

            MlasQGemmStoreRowPairSve(C, ldc, 0, 1, Rows, nn, ColsThis, ZeroMode, acc00, acc01, acc02, acc03);
            MlasQGemmStoreRowPairSve(C, ldc, 2, 3, Rows, nn, ColsThis, ZeroMode, acc10, acc11, acc12, acc13);
            MlasQGemmStoreRowPairSve(C, ldc, 4, 5, Rows, nn, ColsThis, ZeroMode, acc20, acc21, acc22, acc23);
            MlasQGemmStoreRowPairSve(C, ldc, 6, 7, Rows, nn, ColsThis, ZeroMode, acc30, acc31, acc32, acc33);
        }
        return Rows;
#endif  // MLAS_SVE_QGEMM_WIDE_TILE
    }

    //
    // Tail path for Rows in {4,2,1}: process one row-pair at a time. B is
    // re-read per row-pair, but these partial groups are rare (only the final
    // M-tile) and small, so the simpler form is fine. Output still uses the
    // vectorized uzp store.
    //
    const size_t RowPairs = (Rows == 1) ? 1 : (Rows / 2);
    const svbool_t pgA = (Rows == 1) ? svptrue_pat_b8(SV_VL8) : pg16;
    const size_t AStride = (Rows == 1) ? 8 : (RowPairs * 16);

    size_t g = 0;
    for (size_t nn = 0; nn < CountN; nn += 8, ++g) {
        const size_t ColsThis = std::min<size_t>(8, CountN - nn);
        const int8_t* Bgroup = PackedB + g * BGroupStride;

        for (size_t ip = 0; ip < RowPairs; ++ip) {
            const size_t r0 = 2 * ip;
            const size_t r1 = 2 * ip + 1;

            svint32_t acc0 = MlasQGemmSeedTileSve(r0, r1, Rows, nn, 0, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc1 = MlasQGemmSeedTileSve(r0, r1, Rows, nn, 1, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc2 = MlasQGemmSeedTileSve(r0, r1, Rows, nn, 2, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);
            svint32_t acc3 = MlasQGemmSeedTileSve(r0, r1, Rows, nn, 3, RowSumBuffer, ColumnSumBuffer, ZeroPointB, pgTile);

            const int8_t* aptr = PackedA + ip * 16;
            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                const int8_t* bptr = Bgroup + kb * 64;
                const svint8_t b0 = svld1_s8(pg16, bptr + 0);
                const svint8_t b1 = svld1_s8(pg16, bptr + 16);
                const svint8_t b2 = svld1_s8(pg16, bptr + 32);
                const svint8_t b3 = svld1_s8(pg16, bptr + 48);
                const svint8_t a = svld1_s8(pgA, aptr + kb * AStride);

                acc0 = MlasQGemmMmlaSve<AUnsigned>(acc0, a, b0);
                acc1 = MlasQGemmMmlaSve<AUnsigned>(acc1, a, b1);
                acc2 = MlasQGemmMmlaSve<AUnsigned>(acc2, a, b2);
                acc3 = MlasQGemmMmlaSve<AUnsigned>(acc3, a, b3);
            }

            MlasQGemmStoreRowPairSve(C, ldc, r0, r1, Rows, nn, ColsThis, ZeroMode, acc0, acc1, acc2, acc3);
        }
    }
    return Rows;
}
