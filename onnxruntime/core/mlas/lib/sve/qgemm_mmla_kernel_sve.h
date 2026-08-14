/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_mmla_kernel_sve.h

Abstract:

    Shared SVE i8mm compute core for the int8 QGEMM kernels. One templated
    routine covers both the signed (S8S8, svmmla_s32) and unsigned-A / signed-B
    (U8S8, svmmla_u32) cases; the two MLAS kernel-type wrappers
    (qgemm_kernel_smmla_sve.cpp and qgemm_kernel_ummla_sve.cpp) both delegate
    here, so there is a single optimized implementation.

    Contract (identical to the NEON smmla/ummla kernels): the caller supplies
    packed A and packed B in the 8-wide, PackedK=8 MLAS layout, plus RowSum /
    ColumnSum / (optional per-column) ZeroPointB. This routine computes

        C[i][j] = RowSum[i]*ZeroPointB[j] + ColumnSum[j] + sum_k A[i][k]*B[j][k]

    (ZeroPointB[j] == 1 when the pointer is null / per-tensor). Integer math is
    exact, so the result is bit-identical to the NEON kernels.

    VECTOR-LENGTH AGNOSTIC. `svmmla` is a segment-wise operation: each 128-bit
    segment multiplies a 2x8 block of A by a 2x8 block of B into a 2x2 block of
    C. The packed layouts make this scale for free, with no repacking:

      packed B: [8-col panel][k-block][col-pair][2 cols x 8 k] - col-pair
                stride 16 B, k-block stride 64 B. So one full-width load puts
                *consecutive column pairs in consecutive segments*.
      packed A: the mirror - row-pair stride 16 B, k-block stride 64 B. So one
                `svld1rq` (LD1RQB) replicates a row-pair quad to *every*
                segment.

    Combining the two, a single svmmla computes 2 rows x (2 columns per
    segment): 2 columns at VL=128, 4 at VL=256, 8 at VL=512 - the same
    instruction count covers proportionally more of C as the vector grows.
    The uzp1/uzp2 output stage needs no change either: uzp1 over 64-bit
    granules gathers the even (row r0) halves across *all* segments, so it
    yields 2x as many contiguous output columns at 2x the vector length.

    Scaling bound: a k-block is 64 bytes and the final panel of a k-slice ends
    exactly at the end of the packed buffer, so operand loads are capped at 64
    bytes. The kernel therefore scales fully through VL=512 and stays correct
    (just without further scaling) at any larger VL. Going beyond would need a
    cross-segment reduction at the end, since a wider load would span into the
    next k-block.

    Everything here is base SVE - no SVE2 - because Neoverse V1 (Graviton3),
    the primary wide-vector target, implements SVE only.

--*/

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <arm_sve.h>

// This header is self-contained: it is normally included after mlasi.h (which
// defines MLAS_FORCEINLINE), but must also compile stand-alone -- the frozen
// path builds qgemm_mmla_sve_impl.cpp with gen_sve_asm.py, outside the MLAS
// include graph.
#if !defined(MLAS_FORCEINLINE)
#if defined(_MSC_VER)
#define MLAS_FORCEINLINE __forceinline
#else
#define MLAS_FORCEINLINE __attribute__((always_inline)) inline
#endif
#endif

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
// Number of 128-bit segments an operand load covers, i.e. how many column
// pairs one B load feeds. Capped at the 64-byte k-block so a load never runs
// past the end of the packed buffer (see the scaling bound above).
//
static MLAS_FORCEINLINE size_t
MlasQGemmSveSegments(void)
{
    const size_t VectorBytes = svcntb();
    return (VectorBytes < 64 ? VectorBytes : 64) / 16;
}

//
// Address of the packed-B bytes for `col` at k-block `kb`. Columns live in
// 8-wide panels of PackedCountK*64 bytes; within a panel each k-block is 64
// bytes holding the 8 columns in order, 8 bytes each.
//
static MLAS_FORCEINLINE const int8_t*
MlasQGemmSveBPtr(const int8_t* PackedB, size_t BGroupStride, size_t kb, size_t col)
{
    return PackedB + (col / 8) * BGroupStride + kb * 64 + (col % 8) * 8;
}

//
// Column to load packed B from. A pass can span more columns than the caller
// asked for, and the panels past CountN were never allocated, so accumulators
// that are entirely out of range reload the last valid aligned column instead.
// Their results are discarded by the store mask, and the substitute column is
// a multiple of ColsPerAcc, so the load still sits inside one k-block.
//
static MLAS_FORCEINLINE size_t
MlasQGemmSveSafeCol(size_t col, size_t CountN, size_t ColsPerAcc)
{
    return (col < CountN) ? col : (((CountN - 1) / ColsPerAcc) * ColsPerAcc);
}

//
// One packed-A row-pair quad (2 rows x 8 k), replicated to every segment so
// all segments compute the same rows against different columns.
//
static MLAS_FORCEINLINE svint8_t
MlasQGemmSveLoadARowPair(const int8_t* aptr)
{
    return svld1rq_s8(svptrue_b8(), aptr);
}

//
// Same, for Rows == 1, but cheaper: replicate the row's 8 bytes to *both*
// halves of every segment instead of zero-filling the upper half. The mmla's
// "row 1" lanes then recompute row 0, and are discarded by the store (which
// only writes r1 when r1 < Rows). One `dup` replaces memcpy + 2 dup + zip, and
// this sits in the K loop, so it is the hottest of the M == 1 savings.
//
static MLAS_FORCEINLINE svint8_t
MlasQGemmSveLoadASingleRowDup(const int8_t* aptr)
{
    uint64_t Row;
    std::memcpy(&Row, aptr, sizeof(Row));
    return svreinterpret_s8_u64(svdup_n_u64(Row));
}

//
// Store one output row from an accumulator pair. Rows == 1 only: uzp1 gathers
// row 0's halves across all segments, and uzp2 (row 1) is never computed --
// unlike MlasQGemmSveStoreAccPair, which builds both and then discards one.
//
static MLAS_FORCEINLINE void
MlasQGemmSveStoreSingleRow(int32_t* C, size_t ColBase, size_t ColsValid, bool ZeroMode,
                           svint32_t tA, svint32_t tB)
{
    if (ColsValid == 0) {
        return;
    }

    const svint32_t Row0 = svreinterpret_s32_u64(
        svuzp1_u64(svreinterpret_u64_s32(tA), svreinterpret_u64_s32(tB)));

    const svbool_t pg = svwhilelt_b32_u64(uint64_t(0), uint64_t(ColsValid));

    int32_t* d0 = C + ColBase;
    svint32_t v0 = Row0;
    if (!ZeroMode) {
        v0 = svadd_s32_x(pg, v0, svld1_s32(pg, d0));
    }
    svst1_s32(pg, d0, v0);
}

//
// RowSum pattern for a row pair: [rs0, rs0, rs1, rs1] replicated to every
// segment, so it pairs lane-wise with the per-segment [ca, cb, ca, cb] column
// terms. Hoisted out of the column loop.
//
static MLAS_FORCEINLINE svint32_t
MlasQGemmSveRowSumPattern(size_t r0, size_t r1, size_t Rows, const int32_t* RowSumBuffer)
{
    const int32_t rs0 = RowSumBuffer[r0];
    const int32_t rs1 = (r1 < Rows) ? RowSumBuffer[r1] : 0;
    const int32_t Quad[4] = {rs0, rs0, rs1, rs1};
    return svld1rq_s32(svptrue_b32(), Quad);
}

//
// Seed one accumulator: per 128-bit segment [r0ca, r0cb, r1ca, r1cb] holding
// RowSum[r]*ZeroPointB[c] + ColumnSum[c], where (ca, cb) is that segment's
// column pair. ColumnSum and ZeroPointB are read contiguously and expanded to
// the segment pattern with zip1 over 64-bit granules:
//   [c0,c1,c2,c3,..] -> [c0,c1,c0,c1, c2,c3,c2,c3, ..]
// The loads are predicated to the columns that actually exist, so this never
// reads past the caller's buffers.
//
// `col` must be a MlasQGemmSveSafeCol() result, not the raw pass column: a tail
// pass can span past CountN, and forming ColumnSumBuffer + col would then be
// undefined behaviour (past one-past-the-end) even though the all-false
// predicate makes the load touch no memory. MlasQGemmSveSafeCol returns the
// column unchanged whenever col < CountN -- i.e. whenever ColsValid != 0, the
// only case whose result is kept -- and otherwise substitutes the last valid
// aligned column, so this both stays in bounds and computes the same values.
// The substituted accumulator is never stored: both store helpers return early
// on ColsValid == 0, and in the paired store the second accumulator only feeds
// lanes beyond ColsPerAcc, which the store mask excludes.
//
static MLAS_FORCEINLINE svint32_t
MlasQGemmSveSeed(svint32_t RowSumPattern, size_t col, size_t ColsValid,
                 const int32_t* ColumnSumBuffer, const int32_t* ZeroPointB)
{
    const svbool_t pgc = svwhilelt_b32_u64(uint64_t(0), uint64_t(ColsValid));

    const svuint64_t cs = svreinterpret_u64_s32(svld1_s32(pgc, ColumnSumBuffer + col));
    const svint32_t ColumnTerm = svreinterpret_s32_u64(svzip1_u64(cs, cs));

    svint32_t ZeroPointTerm;
    if (ZeroPointB != nullptr) {
        const svuint64_t z = svreinterpret_u64_s32(svld1_s32(pgc, ZeroPointB + col));
        ZeroPointTerm = svreinterpret_s32_u64(svzip1_u64(z, z));
    } else {
        ZeroPointTerm = svdup_n_s32(1);
    }

    return svmla_s32_x(svptrue_b32(), ColumnTerm, RowSumPattern, ZeroPointTerm);
}

//
// Deinterleave two adjacent accumulators into output rows and store. Each
// accumulator holds, per segment, the 2x2 tile [r0ca, r0cb, r1ca, r1cb]; uzp1
// over 64-bit granules gathers every segment's r0 half (and uzp2 the r1 half),
// producing 2 * (columns per accumulator) contiguous output columns. Stores are
// masked to the columns that remain, and accumulate into C when !ZeroMode.
//
static MLAS_FORCEINLINE void
MlasQGemmSveStoreAccPair(int32_t* C, size_t ldc, size_t r0, size_t r1, size_t Rows,
                         size_t ColBase, size_t ColsValid, bool ZeroMode,
                         svint32_t tA, svint32_t tB)
{
    if (ColsValid == 0) {
        return;
    }

    const svuint64_t a = svreinterpret_u64_s32(tA);
    const svuint64_t b = svreinterpret_u64_s32(tB);
    const svint32_t Row0 = svreinterpret_s32_u64(svuzp1_u64(a, b));
    const svint32_t Row1 = svreinterpret_s32_u64(svuzp2_u64(a, b));

    const svbool_t pg = svwhilelt_b32_u64(uint64_t(0), uint64_t(ColsValid));

    int32_t* d0 = C + r0 * ldc + ColBase;
    svint32_t v0 = Row0;
    if (!ZeroMode) {
        v0 = svadd_s32_x(pg, v0, svld1_s32(pg, d0));
    }
    svst1_s32(pg, d0, v0);

    if (r1 < Rows) {
        int32_t* d1 = C + r1 * ldc + ColBase;
        svint32_t v1 = Row1;
        if (!ZeroMode) {
            v1 = svadd_s32_x(pg, v1, svld1_s32(pg, d1));
        }
        svst1_s32(pg, d1, v1);
    }
}

//
// Shared int8 QGEMM inner kernel. AUnsigned selects svmmla_u32 (U8S8) vs
// svmmla_s32 (S8S8). Returns the number of rows handled (12/8/4/2/1), matching
// the driver's packed-A advance contract.
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

    const int8_t* PackedA = reinterpret_cast<const int8_t*>(A);
    const int8_t* PackedB = reinterpret_cast<const int8_t*>(B);
    const size_t BGroupStride = PackedCountK * 64;

    //
    // Vector-length derived geometry. `Segments` column pairs ride in one
    // operand load, so one accumulator covers ColsPerAcc columns and an
    // accumulator pair stores twice that.
    //
    const size_t Segments = MlasQGemmSveSegments();
    const size_t LoadBytes = Segments * 16;
    const size_t ColsPerAcc = Segments * 2;
    const svbool_t pgB = svwhilelt_b8_u64(uint64_t(0), uint64_t(LoadBytes));

#if defined(MLAS_SVE_QGEMM_TILE_12X8) && MLAS_SVE_QGEMM_TILE_12X8
    if (Rows == 12) {
        //
        // 6 row-pairs x 4 column-groups = 24 accumulators. CopyPackA emitted an
        // 8-group (64 B per K-block) followed by a 4-group (32 B per K-block).
        //
        const int8_t* A8 = PackedA;
        const int8_t* A4 = PackedA + PackedCountK * 64;
        const size_t ColsPerPass = 4 * ColsPerAcc;

        const svint32_t rp0 = MlasQGemmSveRowSumPattern(0, 1, Rows, RowSumBuffer);
        const svint32_t rp1 = MlasQGemmSveRowSumPattern(2, 3, Rows, RowSumBuffer);
        const svint32_t rp2 = MlasQGemmSveRowSumPattern(4, 5, Rows, RowSumBuffer);
        const svint32_t rp3 = MlasQGemmSveRowSumPattern(6, 7, Rows, RowSumBuffer);
        const svint32_t rp4 = MlasQGemmSveRowSumPattern(8, 9, Rows, RowSumBuffer);
        const svint32_t rp5 = MlasQGemmSveRowSumPattern(10, 11, Rows, RowSumBuffer);

        for (size_t nn = 0; nn < CountN; nn += ColsPerPass) {
            const size_t c0 = nn;
            const size_t c1 = nn + ColsPerAcc;
            const size_t c2 = nn + 2 * ColsPerAcc;
            const size_t c3 = nn + 3 * ColsPerAcc;

            const size_t v0 = (CountN > c0) ? std::min(ColsPerAcc, CountN - c0) : 0;
            const size_t v1 = (CountN > c1) ? std::min(ColsPerAcc, CountN - c1) : 0;
            const size_t v2 = (CountN > c2) ? std::min(ColsPerAcc, CountN - c2) : 0;
            const size_t v3 = (CountN > c3) ? std::min(ColsPerAcc, CountN - c3) : 0;

            const size_t l0 = MlasQGemmSveSafeCol(c0, CountN, ColsPerAcc);
            const size_t l1 = MlasQGemmSveSafeCol(c1, CountN, ColsPerAcc);
            const size_t l2 = MlasQGemmSveSafeCol(c2, CountN, ColsPerAcc);
            const size_t l3 = MlasQGemmSveSafeCol(c3, CountN, ColsPerAcc);

            svint32_t a00 = MlasQGemmSveSeed(rp0, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t a01 = MlasQGemmSveSeed(rp0, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t a02 = MlasQGemmSveSeed(rp0, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t a03 = MlasQGemmSveSeed(rp0, l3, v3, ColumnSumBuffer, ZeroPointB);
            svint32_t a10 = MlasQGemmSveSeed(rp1, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t a11 = MlasQGemmSveSeed(rp1, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t a12 = MlasQGemmSveSeed(rp1, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t a13 = MlasQGemmSveSeed(rp1, l3, v3, ColumnSumBuffer, ZeroPointB);
            svint32_t a20 = MlasQGemmSveSeed(rp2, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t a21 = MlasQGemmSveSeed(rp2, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t a22 = MlasQGemmSveSeed(rp2, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t a23 = MlasQGemmSveSeed(rp2, l3, v3, ColumnSumBuffer, ZeroPointB);
            svint32_t a30 = MlasQGemmSveSeed(rp3, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t a31 = MlasQGemmSveSeed(rp3, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t a32 = MlasQGemmSveSeed(rp3, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t a33 = MlasQGemmSveSeed(rp3, l3, v3, ColumnSumBuffer, ZeroPointB);
            svint32_t a40 = MlasQGemmSveSeed(rp4, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t a41 = MlasQGemmSveSeed(rp4, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t a42 = MlasQGemmSveSeed(rp4, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t a43 = MlasQGemmSveSeed(rp4, l3, v3, ColumnSumBuffer, ZeroPointB);
            svint32_t a50 = MlasQGemmSveSeed(rp5, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t a51 = MlasQGemmSveSeed(rp5, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t a52 = MlasQGemmSveSeed(rp5, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t a53 = MlasQGemmSveSeed(rp5, l3, v3, ColumnSumBuffer, ZeroPointB);

            // Column bases are loop-invariant; only the K-block offset moves.
            const int8_t* bp0 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l0);
            const int8_t* bp1 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l1);
            const int8_t* bp2 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l2);
            const int8_t* bp3 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l3);

            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                const size_t koff = kb * 64;
                const svint8_t b0 = svld1_s8(pgB, bp0 + koff);
                const svint8_t b1 = svld1_s8(pgB, bp1 + koff);
                const svint8_t b2 = svld1_s8(pgB, bp2 + koff);
                const svint8_t b3 = svld1_s8(pgB, bp3 + koff);

                const int8_t* a8p = A8 + kb * 64;
                const int8_t* a4p = A4 + kb * 32;

                const svint8_t q0 = MlasQGemmSveLoadARowPair(a8p + 0);
                a00 = MlasQGemmMmlaSve<AUnsigned>(a00, q0, b0);
                a01 = MlasQGemmMmlaSve<AUnsigned>(a01, q0, b1);
                a02 = MlasQGemmMmlaSve<AUnsigned>(a02, q0, b2);
                a03 = MlasQGemmMmlaSve<AUnsigned>(a03, q0, b3);

                const svint8_t q1 = MlasQGemmSveLoadARowPair(a8p + 16);
                a10 = MlasQGemmMmlaSve<AUnsigned>(a10, q1, b0);
                a11 = MlasQGemmMmlaSve<AUnsigned>(a11, q1, b1);
                a12 = MlasQGemmMmlaSve<AUnsigned>(a12, q1, b2);
                a13 = MlasQGemmMmlaSve<AUnsigned>(a13, q1, b3);

                const svint8_t q2 = MlasQGemmSveLoadARowPair(a8p + 32);
                a20 = MlasQGemmMmlaSve<AUnsigned>(a20, q2, b0);
                a21 = MlasQGemmMmlaSve<AUnsigned>(a21, q2, b1);
                a22 = MlasQGemmMmlaSve<AUnsigned>(a22, q2, b2);
                a23 = MlasQGemmMmlaSve<AUnsigned>(a23, q2, b3);

                const svint8_t q3 = MlasQGemmSveLoadARowPair(a8p + 48);
                a30 = MlasQGemmMmlaSve<AUnsigned>(a30, q3, b0);
                a31 = MlasQGemmMmlaSve<AUnsigned>(a31, q3, b1);
                a32 = MlasQGemmMmlaSve<AUnsigned>(a32, q3, b2);
                a33 = MlasQGemmMmlaSve<AUnsigned>(a33, q3, b3);

                const svint8_t q4 = MlasQGemmSveLoadARowPair(a4p + 0);
                a40 = MlasQGemmMmlaSve<AUnsigned>(a40, q4, b0);
                a41 = MlasQGemmMmlaSve<AUnsigned>(a41, q4, b1);
                a42 = MlasQGemmMmlaSve<AUnsigned>(a42, q4, b2);
                a43 = MlasQGemmMmlaSve<AUnsigned>(a43, q4, b3);

                const svint8_t q5 = MlasQGemmSveLoadARowPair(a4p + 16);
                a50 = MlasQGemmMmlaSve<AUnsigned>(a50, q5, b0);
                a51 = MlasQGemmMmlaSve<AUnsigned>(a51, q5, b1);
                a52 = MlasQGemmMmlaSve<AUnsigned>(a52, q5, b2);
                a53 = MlasQGemmMmlaSve<AUnsigned>(a53, q5, b3);
            }

            const size_t s0 = v0 + v1;
            const size_t s1 = v2 + v3;

            MlasQGemmSveStoreAccPair(C, ldc, 0, 1, Rows, c0, s0, ZeroMode, a00, a01);
            MlasQGemmSveStoreAccPair(C, ldc, 0, 1, Rows, c2, s1, ZeroMode, a02, a03);
            MlasQGemmSveStoreAccPair(C, ldc, 2, 3, Rows, c0, s0, ZeroMode, a10, a11);
            MlasQGemmSveStoreAccPair(C, ldc, 2, 3, Rows, c2, s1, ZeroMode, a12, a13);
            MlasQGemmSveStoreAccPair(C, ldc, 4, 5, Rows, c0, s0, ZeroMode, a20, a21);
            MlasQGemmSveStoreAccPair(C, ldc, 4, 5, Rows, c2, s1, ZeroMode, a22, a23);
            MlasQGemmSveStoreAccPair(C, ldc, 6, 7, Rows, c0, s0, ZeroMode, a30, a31);
            MlasQGemmSveStoreAccPair(C, ldc, 6, 7, Rows, c2, s1, ZeroMode, a32, a33);
            MlasQGemmSveStoreAccPair(C, ldc, 8, 9, Rows, c0, s0, ZeroMode, a40, a41);
            MlasQGemmSveStoreAccPair(C, ldc, 8, 9, Rows, c2, s1, ZeroMode, a42, a43);
            MlasQGemmSveStoreAccPair(C, ldc, 10, 11, Rows, c0, s0, ZeroMode, a50, a51);
            MlasQGemmSveStoreAccPair(C, ldc, 10, 11, Rows, c2, s1, ZeroMode, a52, a53);
        }

        return Rows;
    }
#endif  // MLAS_SVE_QGEMM_TILE_12X8

    if (Rows == 8) {
        //
        // 4 row-pairs x 6 column-groups = 24 accumulators, so the packed B
        // panel is read once per K-block and each B load feeds 4 mmla
        // (2.4 mmla per load counting the A quads). Covers 12 columns per pass
        // at VL=128, 24 at VL=256, 48 at VL=512.
        //
        const size_t AStride = 64;
        const size_t ColsPerPass = 6 * ColsPerAcc;

        const svint32_t rp0 = MlasQGemmSveRowSumPattern(0, 1, Rows, RowSumBuffer);
        const svint32_t rp1 = MlasQGemmSveRowSumPattern(2, 3, Rows, RowSumBuffer);
        const svint32_t rp2 = MlasQGemmSveRowSumPattern(4, 5, Rows, RowSumBuffer);
        const svint32_t rp3 = MlasQGemmSveRowSumPattern(6, 7, Rows, RowSumBuffer);

        for (size_t nn = 0; nn < CountN; nn += ColsPerPass) {
            const size_t c0 = nn;
            const size_t c1 = nn + ColsPerAcc;
            const size_t c2 = nn + 2 * ColsPerAcc;
            const size_t c3 = nn + 3 * ColsPerAcc;
            const size_t c4 = nn + 4 * ColsPerAcc;
            const size_t c5 = nn + 5 * ColsPerAcc;

            const size_t v0 = (CountN > c0) ? std::min(ColsPerAcc, CountN - c0) : 0;
            const size_t v1 = (CountN > c1) ? std::min(ColsPerAcc, CountN - c1) : 0;
            const size_t v2 = (CountN > c2) ? std::min(ColsPerAcc, CountN - c2) : 0;
            const size_t v3 = (CountN > c3) ? std::min(ColsPerAcc, CountN - c3) : 0;
            const size_t v4 = (CountN > c4) ? std::min(ColsPerAcc, CountN - c4) : 0;
            const size_t v5 = (CountN > c5) ? std::min(ColsPerAcc, CountN - c5) : 0;

            const size_t l0 = MlasQGemmSveSafeCol(c0, CountN, ColsPerAcc);
            const size_t l1 = MlasQGemmSveSafeCol(c1, CountN, ColsPerAcc);
            const size_t l2 = MlasQGemmSveSafeCol(c2, CountN, ColsPerAcc);
            const size_t l3 = MlasQGemmSveSafeCol(c3, CountN, ColsPerAcc);
            const size_t l4 = MlasQGemmSveSafeCol(c4, CountN, ColsPerAcc);
            const size_t l5 = MlasQGemmSveSafeCol(c5, CountN, ColsPerAcc);

            svint32_t a00 = MlasQGemmSveSeed(rp0, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t a01 = MlasQGemmSveSeed(rp0, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t a02 = MlasQGemmSveSeed(rp0, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t a03 = MlasQGemmSveSeed(rp0, l3, v3, ColumnSumBuffer, ZeroPointB);
            svint32_t a04 = MlasQGemmSveSeed(rp0, l4, v4, ColumnSumBuffer, ZeroPointB);
            svint32_t a05 = MlasQGemmSveSeed(rp0, l5, v5, ColumnSumBuffer, ZeroPointB);
            svint32_t a10 = MlasQGemmSveSeed(rp1, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t a11 = MlasQGemmSveSeed(rp1, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t a12 = MlasQGemmSveSeed(rp1, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t a13 = MlasQGemmSveSeed(rp1, l3, v3, ColumnSumBuffer, ZeroPointB);
            svint32_t a14 = MlasQGemmSveSeed(rp1, l4, v4, ColumnSumBuffer, ZeroPointB);
            svint32_t a15 = MlasQGemmSveSeed(rp1, l5, v5, ColumnSumBuffer, ZeroPointB);
            svint32_t a20 = MlasQGemmSveSeed(rp2, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t a21 = MlasQGemmSveSeed(rp2, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t a22 = MlasQGemmSveSeed(rp2, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t a23 = MlasQGemmSveSeed(rp2, l3, v3, ColumnSumBuffer, ZeroPointB);
            svint32_t a24 = MlasQGemmSveSeed(rp2, l4, v4, ColumnSumBuffer, ZeroPointB);
            svint32_t a25 = MlasQGemmSveSeed(rp2, l5, v5, ColumnSumBuffer, ZeroPointB);
            svint32_t a30 = MlasQGemmSveSeed(rp3, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t a31 = MlasQGemmSveSeed(rp3, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t a32 = MlasQGemmSveSeed(rp3, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t a33 = MlasQGemmSveSeed(rp3, l3, v3, ColumnSumBuffer, ZeroPointB);
            svint32_t a34 = MlasQGemmSveSeed(rp3, l4, v4, ColumnSumBuffer, ZeroPointB);
            svint32_t a35 = MlasQGemmSveSeed(rp3, l5, v5, ColumnSumBuffer, ZeroPointB);

            // Column bases are loop-invariant; only the K-block offset moves.
            const int8_t* bp0 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l0);
            const int8_t* bp1 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l1);
            const int8_t* bp2 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l2);
            const int8_t* bp3 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l3);
            const int8_t* bp4 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l4);
            const int8_t* bp5 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l5);

            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                const size_t koff = kb * 64;
                const int8_t* aptr = PackedA + kb * AStride;

                const svint8_t q0 = MlasQGemmSveLoadARowPair(aptr + 0);
                const svint8_t q1 = MlasQGemmSveLoadARowPair(aptr + 16);
                const svint8_t q2 = MlasQGemmSveLoadARowPair(aptr + 32);
                const svint8_t q3 = MlasQGemmSveLoadARowPair(aptr + 48);

                // Each B group is loaded once and consumed by all four row
                // pairs immediately, keeping its live range short.
                const svint8_t b0 = svld1_s8(pgB, bp0 + koff);
                a00 = MlasQGemmMmlaSve<AUnsigned>(a00, q0, b0);
                a10 = MlasQGemmMmlaSve<AUnsigned>(a10, q1, b0);
                a20 = MlasQGemmMmlaSve<AUnsigned>(a20, q2, b0);
                a30 = MlasQGemmMmlaSve<AUnsigned>(a30, q3, b0);

                const svint8_t b1 = svld1_s8(pgB, bp1 + koff);
                a01 = MlasQGemmMmlaSve<AUnsigned>(a01, q0, b1);
                a11 = MlasQGemmMmlaSve<AUnsigned>(a11, q1, b1);
                a21 = MlasQGemmMmlaSve<AUnsigned>(a21, q2, b1);
                a31 = MlasQGemmMmlaSve<AUnsigned>(a31, q3, b1);

                const svint8_t b2 = svld1_s8(pgB, bp2 + koff);
                a02 = MlasQGemmMmlaSve<AUnsigned>(a02, q0, b2);
                a12 = MlasQGemmMmlaSve<AUnsigned>(a12, q1, b2);
                a22 = MlasQGemmMmlaSve<AUnsigned>(a22, q2, b2);
                a32 = MlasQGemmMmlaSve<AUnsigned>(a32, q3, b2);

                const svint8_t b3 = svld1_s8(pgB, bp3 + koff);
                a03 = MlasQGemmMmlaSve<AUnsigned>(a03, q0, b3);
                a13 = MlasQGemmMmlaSve<AUnsigned>(a13, q1, b3);
                a23 = MlasQGemmMmlaSve<AUnsigned>(a23, q2, b3);
                a33 = MlasQGemmMmlaSve<AUnsigned>(a33, q3, b3);

                const svint8_t b4 = svld1_s8(pgB, bp4 + koff);
                a04 = MlasQGemmMmlaSve<AUnsigned>(a04, q0, b4);
                a14 = MlasQGemmMmlaSve<AUnsigned>(a14, q1, b4);
                a24 = MlasQGemmMmlaSve<AUnsigned>(a24, q2, b4);
                a34 = MlasQGemmMmlaSve<AUnsigned>(a34, q3, b4);

                const svint8_t b5 = svld1_s8(pgB, bp5 + koff);
                a05 = MlasQGemmMmlaSve<AUnsigned>(a05, q0, b5);
                a15 = MlasQGemmMmlaSve<AUnsigned>(a15, q1, b5);
                a25 = MlasQGemmMmlaSve<AUnsigned>(a25, q2, b5);
                a35 = MlasQGemmMmlaSve<AUnsigned>(a35, q3, b5);
            }

            const size_t s0 = v0 + v1;
            const size_t s1 = v2 + v3;
            const size_t s2 = v4 + v5;

            MlasQGemmSveStoreAccPair(C, ldc, 0, 1, Rows, c0, s0, ZeroMode, a00, a01);
            MlasQGemmSveStoreAccPair(C, ldc, 0, 1, Rows, c2, s1, ZeroMode, a02, a03);
            MlasQGemmSveStoreAccPair(C, ldc, 0, 1, Rows, c4, s2, ZeroMode, a04, a05);
            MlasQGemmSveStoreAccPair(C, ldc, 2, 3, Rows, c0, s0, ZeroMode, a10, a11);
            MlasQGemmSveStoreAccPair(C, ldc, 2, 3, Rows, c2, s1, ZeroMode, a12, a13);
            MlasQGemmSveStoreAccPair(C, ldc, 2, 3, Rows, c4, s2, ZeroMode, a14, a15);
            MlasQGemmSveStoreAccPair(C, ldc, 4, 5, Rows, c0, s0, ZeroMode, a20, a21);
            MlasQGemmSveStoreAccPair(C, ldc, 4, 5, Rows, c2, s1, ZeroMode, a22, a23);
            MlasQGemmSveStoreAccPair(C, ldc, 4, 5, Rows, c4, s2, ZeroMode, a24, a25);
            MlasQGemmSveStoreAccPair(C, ldc, 6, 7, Rows, c0, s0, ZeroMode, a30, a31);
            MlasQGemmSveStoreAccPair(C, ldc, 6, 7, Rows, c2, s1, ZeroMode, a32, a33);
            MlasQGemmSveStoreAccPair(C, ldc, 6, 7, Rows, c4, s2, ZeroMode, a34, a35);
        }

        return Rows;
    }

    if (Rows == 1) {
        //
        // Dedicated single-row path.
        //
        // mmla cannot beat udot at M == 1: only one of the two A rows is real,
        // so 32 MACs/instruction degrade to 16 -- exactly udot's rate. Parity is
        // therefore the ceiling here, and the entire job of this path is to
        // strip the per-call overhead that the shared row-pair path pays and
        // that a single row has almost no work to amortise:
        //
        //   * RowSum needs no [rs0,rs0,rs1,rs1] pattern (there is no row 1), so
        //     svdup replaces building a 4-int array on the stack and reloading
        //     it with ld1rq;
        //   * the A quad is loaded with one dup instead of memcpy + 2 dup + zip
        //     -- this is inside the K loop, so it is the dominant saving;
        //   * the store skips uzp2 entirely rather than computing row 1 and
        //     throwing it away.
        //
        // All three are safe for the same reason: the mmla's row-1 lanes are
        // computed but never stored, so whatever lands in them is irrelevant.
        //
        // Four column groups, matching the four accumulators declared below.
        // (SVE types are sizeless, so accumulators cannot live in an array;
        // widening the group count means adding variables here.)
        const size_t ColsPerPass = 4 * ColsPerAcc;
        const svint32_t rp = svdup_n_s32(RowSumBuffer[0]);

        for (size_t nn = 0; nn < CountN; nn += ColsPerPass) {
            const size_t c0 = nn;
            const size_t c1 = nn + ColsPerAcc;
            const size_t c2 = nn + 2 * ColsPerAcc;
            const size_t c3 = nn + 3 * ColsPerAcc;

            const size_t v0 = (CountN > c0) ? std::min(ColsPerAcc, CountN - c0) : 0;
            const size_t v1 = (CountN > c1) ? std::min(ColsPerAcc, CountN - c1) : 0;
            const size_t v2 = (CountN > c2) ? std::min(ColsPerAcc, CountN - c2) : 0;
            const size_t v3 = (CountN > c3) ? std::min(ColsPerAcc, CountN - c3) : 0;

            const size_t l0 = MlasQGemmSveSafeCol(c0, CountN, ColsPerAcc);
            const size_t l1 = MlasQGemmSveSafeCol(c1, CountN, ColsPerAcc);
            const size_t l2 = MlasQGemmSveSafeCol(c2, CountN, ColsPerAcc);
            const size_t l3 = MlasQGemmSveSafeCol(c3, CountN, ColsPerAcc);

            const int8_t* bp0 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l0);
            const int8_t* bp1 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l1);
            const int8_t* bp2 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l2);
            const int8_t* bp3 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l3);

            svint32_t acc0 = MlasQGemmSveSeed(rp, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t acc1 = MlasQGemmSveSeed(rp, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t acc2 = MlasQGemmSveSeed(rp, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t acc3 = MlasQGemmSveSeed(rp, l3, v3, ColumnSumBuffer, ZeroPointB);

            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                const size_t koff = kb * 64;
                // Rows == 1: the packed group holds 8 bytes per k-block.
                const svint8_t q = MlasQGemmSveLoadASingleRowDup(PackedA + kb * 8);

                acc0 = MlasQGemmMmlaSve<AUnsigned>(acc0, q, svld1_s8(pgB, bp0 + koff));
                acc1 = MlasQGemmMmlaSve<AUnsigned>(acc1, q, svld1_s8(pgB, bp1 + koff));
                acc2 = MlasQGemmMmlaSve<AUnsigned>(acc2, q, svld1_s8(pgB, bp2 + koff));
                acc3 = MlasQGemmMmlaSve<AUnsigned>(acc3, q, svld1_s8(pgB, bp3 + koff));
            }

            MlasQGemmSveStoreSingleRow(C, c0, v0 + v1, ZeroMode, acc0, acc1);
            MlasQGemmSveStoreSingleRow(C, c2, v2 + v3, ZeroMode, acc2, acc3);
        }

        return Rows;
    }

    //
    // Tail path for Rows in {4,2}: process one row-pair at a time against 4
    // column groups. B is re-read per row-pair, but these partial groups are
    // rare (only the final M-tile) and small, so the simpler form is fine.
    //
    const size_t RowPairs = Rows / 2;
    const size_t AStride = RowPairs * 16;
    const size_t ColsPerPass = 4 * ColsPerAcc;

    for (size_t nn = 0; nn < CountN; nn += ColsPerPass) {
        const size_t c0 = nn;
        const size_t c1 = nn + ColsPerAcc;
        const size_t c2 = nn + 2 * ColsPerAcc;
        const size_t c3 = nn + 3 * ColsPerAcc;

        const size_t v0 = (CountN > c0) ? std::min(ColsPerAcc, CountN - c0) : 0;
        const size_t v1 = (CountN > c1) ? std::min(ColsPerAcc, CountN - c1) : 0;
        const size_t v2 = (CountN > c2) ? std::min(ColsPerAcc, CountN - c2) : 0;
        const size_t v3 = (CountN > c3) ? std::min(ColsPerAcc, CountN - c3) : 0;

        const size_t l0 = MlasQGemmSveSafeCol(c0, CountN, ColsPerAcc);
        const size_t l1 = MlasQGemmSveSafeCol(c1, CountN, ColsPerAcc);
        const size_t l2 = MlasQGemmSveSafeCol(c2, CountN, ColsPerAcc);
        const size_t l3 = MlasQGemmSveSafeCol(c3, CountN, ColsPerAcc);

        // Column bases are loop-invariant; only the K-block offset moves.
        const int8_t* bp0 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l0);
        const int8_t* bp1 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l1);
        const int8_t* bp2 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l2);
        const int8_t* bp3 = MlasQGemmSveBPtr(PackedB, BGroupStride, 0, l3);

        for (size_t ip = 0; ip < RowPairs; ++ip) {
            const size_t r0 = 2 * ip;
            const size_t r1 = 2 * ip + 1;

            const svint32_t rp = MlasQGemmSveRowSumPattern(r0, r1, Rows, RowSumBuffer);

            svint32_t acc0 = MlasQGemmSveSeed(rp, l0, v0, ColumnSumBuffer, ZeroPointB);
            svint32_t acc1 = MlasQGemmSveSeed(rp, l1, v1, ColumnSumBuffer, ZeroPointB);
            svint32_t acc2 = MlasQGemmSveSeed(rp, l2, v2, ColumnSumBuffer, ZeroPointB);
            svint32_t acc3 = MlasQGemmSveSeed(rp, l3, v3, ColumnSumBuffer, ZeroPointB);

            const int8_t* aptr = PackedA + ip * 16;

            for (size_t kb = 0; kb < PackedCountK; ++kb) {
                const size_t koff = kb * 64;
                const svint8_t q = MlasQGemmSveLoadARowPair(aptr + kb * AStride);

                acc0 = MlasQGemmMmlaSve<AUnsigned>(
                    acc0, q, svld1_s8(pgB, bp0 + koff));
                acc1 = MlasQGemmMmlaSve<AUnsigned>(
                    acc1, q, svld1_s8(pgB, bp1 + koff));
                acc2 = MlasQGemmMmlaSve<AUnsigned>(
                    acc2, q, svld1_s8(pgB, bp2 + koff));
                acc3 = MlasQGemmMmlaSve<AUnsigned>(
                    acc3, q, svld1_s8(pgB, bp3 + koff));
            }

            MlasQGemmSveStoreAccPair(C, ldc, r0, r1, Rows, c0, v0 + v1, ZeroMode, acc0, acc1);
            MlasQGemmSveStoreAccPair(C, ldc, r0, r1, Rows, c2, v2 + v3, ZeroMode, acc2, acc3);
        }
    }

    return Rows;
}
