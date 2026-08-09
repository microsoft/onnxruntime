/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_wasmsimd.cpp

Abstract:

    This module implements QGEMM kernel for WebAssembly SIMD128.

--*/

#include "mlasi.h"
#include "qgemm.h"

// wasm implementation of "_mm_unpacklo_epi8"
v128_t __attribute__((__always_inline__, __nodebug__)) wasm_i8x16_unpacklo(v128_t a, v128_t b) {
    return wasm_i8x16_shuffle(a, b, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
}

// wasm implementation of "_mm_unpackhi_epi8"
v128_t __attribute__((__always_inline__, __nodebug__)) wasm_i8x16_unpackhi(v128_t a, v128_t b) {
    return wasm_i8x16_shuffle(a, b, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
}

struct MLAS_GEMM_U8X8_KERNEL_WASMSIMD
{
    typedef int16_t PackedAType;
    typedef int16_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 12, 128, 128 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{0, 0, 0};
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8X8_KERNEL_WASMSIMD::Strides;

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_WASMSIMD>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (!BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_WASMSIMD::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_U8X8_KERNEL_WASMSIMD>(
    MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    const v128_t ZeroVector = wasm_i64x2_const(0, 0);
    const v128_t OnesWordBroadcast = wasm_i16x8_splat(1);
    uint8_t PaddedMatrixAData[8] = { 0 };

    //
    // Process a single row of matrix A in a loop.
    //

    while (CountM > 0) {

        const uint8_t* a = A;
        size_t k = CountK;
        v128_t ReductionVector = ZeroVector;

        //
        // Zero extend the source bytes to 16-bits and write to the packed
        // buffer.
        //
        // The packed buffer has the same data ordering as the source bytes,
        // but CountK is aligned up to a multiple of 2 to maintain 32-bit
        // alignment. All extra bytes are zero-padded.
        //
        // These 16-bit values are also accumulated into an intermediate per-row
        // accumulator. CountK cannot be greater than 128 to avoid overflowing
        // these signed 16-bit accumulators.
        //

        while (k >= 8) {

            v128_t Bytes = wasm_v128_load64_zero(&a[0]);
            v128_t Words = wasm_i8x16_unpacklo(Bytes, ZeroVector);

            ReductionVector = wasm_i16x8_add(ReductionVector, Words);

            wasm_v128_store(&D[0], Words);

            a += 8;
            D += 8;
            k -= 8;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            uint8_t* padded = PaddedMatrixAData;
            uint8_t* padded_end = padded + k;

            do {
                padded[0] = a[0];
                padded++;
                a++;
            } while (padded < padded_end);

            v128_t Bytes = wasm_v128_load64_zero(PaddedMatrixAData);
            v128_t Words = wasm_i8x16_unpacklo(Bytes, ZeroVector);

            ReductionVector = wasm_i16x8_add(ReductionVector, Words);

            //
            // Copy pairs of 16-bit values from the vector to the packed
            // buffer and rotate the vector for the next iteration.
            //

            for (size_t pairs = (k + 1) / 2; pairs > 0; pairs--) {
                *((int32_t*)D) = wasm_i32x4_extract_lane(Words, 0);
                D += 2;
                Words = wasm_i32x4_shuffle(Words, wasm_i32x4_splat(0), 1, 2, 3, 0);
            }
        }

        //
        // Reduce the partial accumulators.
        //

        ReductionVector = wasm_i32x4_dot_i16x8(ReductionVector, OnesWordBroadcast);
        ReductionVector = wasm_i32x4_add(ReductionVector,
                                         wasm_i32x4_shuffle(ReductionVector, wasm_i32x4_splat(0), 2, 3, 2, 3));
        ReductionVector = wasm_i32x4_add(ReductionVector,
                                         wasm_i32x4_shuffle(ReductionVector, wasm_i32x4_splat(0), 1, 0, 1, 0));

        *RowSumBuffer++ = wasm_i32x4_extract_lane(ReductionVector, 0);

        A += lda;
        CountM -= 1;
    }
}


MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessWasmSimd(
    MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedBType* D,
    v128_t BytesRow0,
    v128_t BytesRow1,
    v128_t BitFlipVector,
    v128_t ColumnSums[2]
)
{
    v128_t BytesInterleaved = wasm_i8x16_unpacklo(BytesRow0, BytesRow1);

    BytesInterleaved = wasm_v128_xor(BytesInterleaved, BitFlipVector);

    v128_t WordsInterleaved0 = wasm_i16x8_shr(wasm_i8x16_unpacklo(BytesInterleaved, BytesInterleaved), 8);
    v128_t WordsInterleaved1 = wasm_i16x8_shr(wasm_i8x16_unpackhi(BytesInterleaved, BytesInterleaved), 8);

    ColumnSums[0] = wasm_i16x8_add(ColumnSums[0], WordsInterleaved0);
    ColumnSums[1] = wasm_i16x8_add(ColumnSums[1], WordsInterleaved1);

    wasm_v128_store(&D[0], WordsInterleaved0);
    wasm_v128_store(&D[8], WordsInterleaved1);
}

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_U8X8_KERNEL_WASMSIMD>(
    MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const v128_t OnesWordBroadcast = wasm_i16x8_splat(1);
    const v128_t BitFlipVector = wasm_i32x4_splat(BIsSigned ? 0 : 0x80808080);

    //
    // Process 8 columns of matrix B in a loop.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        size_t k = CountK;
        v128_t ColumnSums[2];

        ColumnSums[0] = wasm_i64x2_const(0, 0);
        ColumnSums[1] = wasm_i64x2_const(0, 0);

        //
        // Interleave rows of matrix B and write to the packed buffer.
        //
        // These values are also zero-extended and accumulated into an
        // intermediate per-column accumulator. CountK cannot be greater than
        // 128 to avoid overflowing these signed 16-bit accumulators.
        //

        while (k >= MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedK) {

            v128_t BytesRow0 = wasm_v128_load64_zero(&b[0]);
            v128_t BytesRow1 = wasm_v128_load64_zero(&b[ldb]);

            MlasGemmU8X8CopyPackBProcessWasmSimd(D, BytesRow0, BytesRow1, BitFlipVector, ColumnSums);

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            v128_t BytesRow0 = wasm_v128_load64_zero(&b[0]);

            MlasGemmU8X8CopyPackBProcessWasmSimd(D, BytesRow0, BitFlipVector, BitFlipVector, ColumnSums);

            D += 16;
        }

        ColumnSums[0] = wasm_i32x4_dot_i16x8(ColumnSums[0], OnesWordBroadcast);
        ColumnSums[1] = wasm_i32x4_dot_i16x8(ColumnSums[1], OnesWordBroadcast);

        wasm_v128_store(&ColumnSumBuffer[0], ColumnSums[0]);
        wasm_v128_store(&ColumnSumBuffer[4], ColumnSums[1]);
        ColumnSumBuffer += 8;

        B += 8;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {

        const uint8_t* b = B;
        size_t k = CountK;
        v128_t ColumnSums[2];
        uint8_t PaddedMatrixBData[16];

        wasm_v128_store(PaddedMatrixBData, BitFlipVector);

        ColumnSums[0] = wasm_i64x2_const(0, 0);
        ColumnSums[1] = wasm_i64x2_const(0, 0);

        //
        // Interleave rows of matrix B using an intermediate zero padded stack
        // buffer and write to the packed buffer.
        //

        while (k >= MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedK) {

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded[8] = bcopy[ldb];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            v128_t BytesRow0 = wasm_v128_load64_zero(&PaddedMatrixBData[0]);
            v128_t BytesRow1 = wasm_v128_load64_zero(&PaddedMatrixBData[8]);

            MlasGemmU8X8CopyPackBProcessWasmSimd(D, BytesRow0, BytesRow1, BitFlipVector, ColumnSums);

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            v128_t BytesRow0 = wasm_v128_load64_zero(&PaddedMatrixBData[0]);

            MlasGemmU8X8CopyPackBProcessWasmSimd(D, BytesRow0, BitFlipVector, BitFlipVector, ColumnSums);
        }

        ColumnSums[0] = wasm_i32x4_dot_i16x8(ColumnSums[0], OnesWordBroadcast);
        ColumnSums[1] = wasm_i32x4_dot_i16x8(ColumnSums[1], OnesWordBroadcast);

        wasm_v128_store(&ColumnSumBuffer[0], ColumnSums[0]);
        wasm_v128_store(&ColumnSumBuffer[4], ColumnSums[1]);
    }
}

//------------------------------------------------------------------
// Helper – dot‑product add for i16×i16 → i32 pairs.
//------------------------------------------------------------------
MLAS_FORCEINLINE void DotPairAddI16(v128_t ABroadcast,
                                    v128_t BVec0,
                                    v128_t BVec1,
                                    v128_t Acc[2]) {
    Acc[0] = wasm_i32x4_add(Acc[0], wasm_i32x4_dot_i16x8(BVec0, ABroadcast));
    Acc[1] = wasm_i32x4_add(Acc[1], wasm_i32x4_dot_i16x8(BVec1, ABroadcast));
}

//------------------------------------------------------------------
// Generic RowCount×8 kernel (RowCount = 4 or 1) for WASM SIMD.
//------------------------------------------------------------------

template<size_t RowCount>
static size_t GemmQuantKernelNx8Impl(
    const MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t /*CountM — ignored*/,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode)
{

    constexpr size_t ColBlock = 8;
    const auto PackedK = MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedK; // ==2

    const MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedAType* a[RowCount];
    int32_t* c[RowCount];
    for (size_t r = 0; r < RowCount; ++r) {
        a[r] = (const MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedAType*)(A + r * PackedK * PackedCountK);
        c[r] = (int32_t*)(C + r * ldc);
    }

    while (CountN > 0) {
        // ------------------------------------------------------------------
        // 1) Initialize accumulators with row & column sums (and zero‑points)
        // ------------------------------------------------------------------
        v128_t Acc[RowCount][2];
        v128_t col0 = wasm_v128_load(ColumnSumBuffer + 0);
        v128_t col1 = wasm_v128_load(ColumnSumBuffer + 4);

        if (ZeroPointB) {
            v128_t zp0 = wasm_v128_load(ZeroPointB + 0);
            v128_t zp1 = wasm_v128_load(ZeroPointB + 4);
            ZeroPointB += 8;

            for (size_t r = 0; r < RowCount; ++r) {
                v128_t RowSumValues = wasm_v128_load32_splat(RowSumBuffer + r);
                Acc[r][0] = wasm_i32x4_add(wasm_i32x4_mul(RowSumValues, zp0), col0);
                Acc[r][1] = wasm_i32x4_add(wasm_i32x4_mul(RowSumValues, zp1), col1);
            }
        } else {
            for (size_t r = 0; r < RowCount; ++r) {
                v128_t RowSumValues = wasm_v128_load32_splat(RowSumBuffer + r);
                Acc[r][0] = wasm_i32x4_add(RowSumValues, col0);
                Acc[r][1] = wasm_i32x4_add(RowSumValues, col1);
            }
        }
        ColumnSumBuffer += 8;

        // ----------------------------------------------------------------------
        // 2) Broadcast each pair of 16-bit values from the matrix A and multiply
        // with the pair of 16-bit values from matrix B, and add the 32-bit
        // intermediate into the accumulator registers.
        // ----------------------------------------------------------------------
        size_t k = PackedCountK;
        while (k > 0) {
            v128_t ABroadcast[RowCount];
            for (size_t r = 0; r < RowCount; ++r) {
                ABroadcast[r] = wasm_v128_load32_splat(a[r]);
                a[r] += 2;
            }

            v128_t B0 = wasm_v128_load(B + 0);  // cols 0‑3 (8 i16)
            v128_t B1 = wasm_v128_load(B + 8);  // cols 4‑7 (8 i16)


            for (size_t r = 0; r < RowCount; ++r) {
                DotPairAddI16(ABroadcast[r], B0, B1, Acc[r]);
            }

            B += 16;
            k -= 1;
        }

        // ------------------------------------------------------------------
        // 3) Output the accumulator block after optionally accumulating the values
        // from matrix C.
        // ------------------------------------------------------------------
        if (CountN >= 8) {
            for (size_t r = 0; r < RowCount; ++r) {
                if (!ZeroMode) {
                    Acc[r][0] = wasm_i32x4_add(Acc[r][0], wasm_v128_load(c[r] + 0));
                    Acc[r][1] = wasm_i32x4_add(Acc[r][1], wasm_v128_load(c[r] + 4));
                }
                wasm_v128_store(c[r] + 0, Acc[r][0]);
                wasm_v128_store(c[r] + 4, Acc[r][1]);
                c[r] += ColBlock;
                a[r] -= PackedCountK * 2; // Rewind a[r] for next N-tile (PackedCountK * 2 elements, 16-bit each).
            }
            CountN -= 8;
        } else {
            // ---- 4/2/1‑column tails ----
            auto Tail = [&](size_t cols, auto load_c, auto store_c) {
                for (size_t r = 0; r < RowCount; ++r) {
                    if (!ZeroMode) Acc[r][0] = wasm_i32x4_add(Acc[r][0], load_c(c[r]));
                }
                for (size_t r = 0; r < RowCount; ++r) store_c(c[r], Acc[r][0]);
                for (size_t r = 0; r < RowCount; ++r) c[r] += cols;
            };

            if (CountN & 4) {
                Tail(4,
                     [](int32_t* p) { return wasm_v128_load(p); },
                     [](int32_t* p, v128_t v) { wasm_v128_store(p, v); });
                for (size_t r = 0; r < RowCount; ++r) Acc[r][0] = Acc[r][1];
            }
            if (CountN & 2) {
                Tail(2,
                     [](int32_t* p) { return wasm_v128_load64_zero(p); },
                     [](int32_t* p, v128_t v) { wasm_v128_store64_lane(p, v, 0); });
                for (size_t r = 0; r < RowCount; ++r)
                    Acc[r][0] = wasm_i32x4_shuffle(Acc[r][0], wasm_i32x4_splat(0), 2, 3, 2, 3);
            }
            if (CountN & 1) {
                for (size_t r = 0; r < RowCount; ++r) {
                    int32_t v = wasm_i32x4_extract_lane(Acc[r][0], 0);
                    if (!ZeroMode) v += *c[r];
                    *c[r] = v;
                }
            }
            CountN = 0;
        }
    }
    return RowCount;
}

size_t MlasGemmQuantKernel4x8(
    const MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode) {
    MLAS_UNREFERENCED_PARAMETER(CountM);
    return GemmQuantKernelNx8Impl<4>(A, B, C, PackedCountK, 0, CountN, ldc,
                                     RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

size_t MlasGemmQuantKernel1x8(
    const MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode) {
    MLAS_UNREFERENCED_PARAMETER(CountM);
    return GemmQuantKernelNx8Impl<1>(A, B, C, PackedCountK, 0, CountN, ldc,
                                     RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_U8X8_KERNEL_WASMSIMD>(
    const MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    size_t RowsHandled = 0;
    if (CountM >= 4) {
        RowsHandled = MlasGemmQuantKernel4x8(A, B, C, PackedCountK, CountM, CountN, ldc,
                                             RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
    } else {
        RowsHandled = MlasGemmQuantKernel1x8(A, B, C, PackedCountK, CountM, CountN, ldc,
                                             RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
    }
    return RowsHandled;
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8X8DispatchWasmSimd = {
    MlasGemmQuantOperation<MLAS_GEMM_U8X8_KERNEL_WASMSIMD>,
    nullptr,
    nullptr,
    MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedK,
    0,
    4 // multiple of kernel stride M
};
