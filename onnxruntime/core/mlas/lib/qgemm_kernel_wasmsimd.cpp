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

MLAS_FORCEINLINE
void
MlasGemmU8X8MultiplyAccumulateRowWasmSimd(
    v128_t ABroadcast,
    const int16_t* B,
    v128_t Accumulators[2]
)
{
    v128_t BElements0 = wasm_v128_load(&B[0]);
    v128_t BElements1 = wasm_v128_load(&B[8]);

    Accumulators[0] = wasm_i32x4_add(Accumulators[0], wasm_i32x4_dot_i16x8(BElements0, ABroadcast));
    Accumulators[1] = wasm_i32x4_add(Accumulators[1], wasm_i32x4_dot_i16x8(BElements1, ABroadcast));
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
    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(ldc);

    while (CountN > 0) {

        v128_t Accumulators[2];

        //
        // Initialize the accumulators with the row and column sums.
        //

        int32_t RowSumValue = RowSumBuffer[0];

        if (ZeroPointB != nullptr) {

            int32_t ScaledRowSumBuffer[8];

            for (size_t i = 0; i < 8; i++) {
                ScaledRowSumBuffer[i] = RowSumValue * ZeroPointB[i];
            }

            ZeroPointB += 8;

            Accumulators[0] = wasm_v128_load(&ScaledRowSumBuffer[0]);
            Accumulators[1] = wasm_v128_load(&ScaledRowSumBuffer[4]);

        }
        else {

            Accumulators[0] = wasm_i32x4_splat(RowSumValue);
            Accumulators[1] = Accumulators[0];
        }

        Accumulators[0] = wasm_i32x4_add(Accumulators[0], wasm_v128_load(&ColumnSumBuffer[0]));
        Accumulators[1] = wasm_i32x4_add(Accumulators[1], wasm_v128_load(&ColumnSumBuffer[4]));
        ColumnSumBuffer += 8;

        //
        // Broadcast each pair of 16-bit values from the matrix A and multiply
        // with the pair of 16-bit values from matrix B, and add the 32-bit
        // intermediate into the accumulator registers.
        //

        const int16_t* a = A;
        size_t k = PackedCountK;

        while (k >= 4) {

            v128_t AElements = wasm_v128_load((v128_t*)a);
            v128_t ABroadcast;

            ABroadcast = wasm_i32x4_shuffle(AElements, wasm_i32x4_splat(0), 0, 0, 0, 0);
            MlasGemmU8X8MultiplyAccumulateRowWasmSimd(ABroadcast, &B[0], Accumulators);

            ABroadcast = wasm_i32x4_shuffle(AElements, wasm_i32x4_splat(0), 1, 1, 1, 1);
            MlasGemmU8X8MultiplyAccumulateRowWasmSimd(ABroadcast, &B[16], Accumulators);

            ABroadcast = wasm_i32x4_shuffle(AElements, wasm_i32x4_splat(0), 2, 2, 2, 2);
            MlasGemmU8X8MultiplyAccumulateRowWasmSimd(ABroadcast, &B[32], Accumulators);

            ABroadcast = wasm_i32x4_shuffle(AElements, wasm_i32x4_splat(0), 3, 3, 3, 3);
            MlasGemmU8X8MultiplyAccumulateRowWasmSimd(ABroadcast, &B[48], Accumulators);

            a += 4 * 2;
            B += 4 * 16;
            k -= 4;
        }

        while (k > 0) {

            v128_t ABroadcast = wasm_i32x4_splat(*((int32_t*)a));
            MlasGemmU8X8MultiplyAccumulateRowWasmSimd(ABroadcast, &B[0], Accumulators);

            a += 2;
            B += 16;
            k -= 1;
        }

        //
        // Output the accumulator block after optionally accumulating the values
        // from matrix C.
        //

        if (CountN >= 8) {

            if (!ZeroMode) {
                Accumulators[0] = wasm_i32x4_add(Accumulators[0], wasm_v128_load(&C[0]));
                Accumulators[1] = wasm_i32x4_add(Accumulators[1], wasm_v128_load(&C[4]));
            }

            wasm_v128_store(&C[0], Accumulators[0]);
            wasm_v128_store(&C[4], Accumulators[1]);

            C += 8;
            CountN -= 8;

        }
        else {

            //
            // Output the remaining partial output block.
            //

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = wasm_i32x4_add(Accumulators[0], wasm_v128_load(&C[0]));
                }

                wasm_v128_store(&C[0], Accumulators[0]);
                C += 4;

                Accumulators[0] = Accumulators[1];
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = wasm_i32x4_add(Accumulators[0], wasm_v128_load64_zero(&C[0]));
                }

                wasm_v128_store64_lane(&C[0], Accumulators[0], 0);
                C += 2;

                Accumulators[0] = wasm_i32x4_shuffle(Accumulators[0], wasm_i32x4_splat(0), 2, 3, 2, 3);
            }

            if ((CountN & 1) != 0) {

                int32_t AccumulatorValue = wasm_i32x4_extract_lane(Accumulators[0], 0);

                if (!ZeroMode) {
                    AccumulatorValue += C[0];
                }

                C[0] = AccumulatorValue;
            }

            CountN = 0;
        }
    }

    return 1;
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8X8DispatchWasmSimd = {
    MlasGemmQuantOperation<MLAS_GEMM_U8X8_KERNEL_WASMSIMD>,
    nullptr,
    nullptr,
    MLAS_GEMM_U8X8_KERNEL_WASMSIMD::PackedK,
    0,
    4 // multiple of kernel stride M
};
