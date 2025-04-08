/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_wasmrelaxedsimd.cpp

Abstract:

    This module implements QGEMM kernel for WebAssembly Relaxed SIMD128.

--*/

#include "mlasi.h"
#include "qgemm.h"

bool HasUSDot() {
// Check out-of-bounds behavior of Relaxed Integer Dot Product with Accumulation with signed and unsigned input (e.g. vpdpbusd).
      const v128_t int8_input = wasm_i8x16_const(0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0);
      const volatile v128_t xint8_input = wasm_i8x16_const(0, 0, 0, -128, 0, 0, -128, 0, 0, -128, 0, 0, -128, 0, 0, 0);  // volatile to confuse Clang which otherwise ICE's
      const v128_t xint8_output = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(int8_input, xint8_input, wasm_i8x16_const_splat(0));

      const volatile v128_t overflow_input = wasm_i8x16_const(-128, -128, -128, -128, -128, -128, -1, -1, -1, -1, -128, -128, -1, -1, -1, -1);  // volatile to confuse Clang which otherwise ICE's
      const v128_t overflow_output = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_i8x16_const_splat(-128), overflow_input, wasm_i8x16_const_splat(0));
      return !wasm_v128_any_true(wasm_v128_or(
        wasm_v128_xor(xint8_output, wasm_i32x4_const_splat(128)),
        wasm_v128_xor(overflow_output, wasm_i32x4_const(-65536, -98048, -98048, -130560))));
}

// wasm implementation of "_mm_unpacklo_epi8"
v128_t __attribute__((__always_inline__, __nodebug__)) wasm_i8x16_unpacklo_relaxed(v128_t a, v128_t b) {
    return wasm_i8x16_shuffle(a, b, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
}

// wasm implementation of "_mm_unpacklo_epi16"
v128_t __attribute__((__always_inline__, __nodebug__)) wasm_i16x8_unpacklo_relaxed(v128_t a, v128_t b) {
    return wasm_i8x16_shuffle(a, b, 0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
}

// wasm implementation of "_mm_unpackhi_epi16"
v128_t __attribute__((__always_inline__, __nodebug__)) wasm_i16x8_unpackhi_relaxed(v128_t a, v128_t b) {
    return wasm_i8x16_shuffle(a, b, 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
}

struct MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 12, 128, 128 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{0, 0, 0};
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::Strides;

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (!BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD>(
    MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::PackedAType* D,
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
        // Copy the source bytes to the packed buffer.
        //
        // The packed buffer has the same data ordering as the source bytes,
        // but CountK is aligned up to a multiple of 4 to maintain 32-bit
        // alignment. All extra bytes are zero-padded.
        //
        // Zero extend the source bytes to 16-bits and accumulate
        // into an intermediate per-row
        // accumulator. CountK cannot be greater than 128 to avoid overflowing
        // these signed 16-bit accumulators.
        //

        while (k >= 8) {

            v128_t Bytes = wasm_v128_load64_zero(&a[0]);
            v128_t Words = wasm_i8x16_unpacklo_relaxed(Bytes, ZeroVector);

            ReductionVector = wasm_i16x8_add(ReductionVector, Words);

            wasm_v128_store64_lane(&D[0], Bytes, 0);

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
            v128_t Words = wasm_i8x16_unpacklo_relaxed(Bytes, ZeroVector);

            ReductionVector = wasm_i16x8_add(ReductionVector, Words);

            //
            // Copy quads of 8-bit values from the vector to the packed
            // buffer and rotate the vector for the next iteration.
            //

            for (size_t quads = (k + 3) / 4; quads > 0; quads--) {
                *((int32_t*)D) = wasm_i32x4_extract_lane(Bytes, 0);
                D += 4;
                Bytes = wasm_i32x4_shuffle(Bytes, wasm_i32x4_splat(0), 1, 2, 3, 0);
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
MlasGemmU8X8CopyPackBProcessWasmRelaxedSimd(
    MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::PackedBType* D,
    v128_t BytesRow0,
    v128_t BytesRow1,
    v128_t BytesRow2,
    v128_t BytesRow3,
    v128_t BitFlipVector,
    v128_t OnesByteBroadcast,
    v128_t ColumnSums[2]
)
{
    v128_t PairsInterleaved0 = wasm_i8x16_unpacklo_relaxed(BytesRow0, BytesRow1);
    v128_t PairsInterleaved1 = wasm_i8x16_unpacklo_relaxed(BytesRow2, BytesRow3);

    PairsInterleaved0 = wasm_v128_xor(PairsInterleaved0, BitFlipVector);
    PairsInterleaved1 = wasm_v128_xor(PairsInterleaved1, BitFlipVector);

    v128_t QuadsInterleaved0 = wasm_i16x8_unpacklo_relaxed(PairsInterleaved0, PairsInterleaved1);
    v128_t QuadsInterleaved1 = wasm_i16x8_unpackhi_relaxed(PairsInterleaved0, PairsInterleaved1);

    ColumnSums[0] = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(QuadsInterleaved0, OnesByteBroadcast, ColumnSums[0]);
    ColumnSums[1] = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(QuadsInterleaved1, OnesByteBroadcast, ColumnSums[1]);

    wasm_v128_store(&D[0], QuadsInterleaved0);
    wasm_v128_store(&D[16], QuadsInterleaved1);
}

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD>(
    MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const v128_t OnesByteBroadcast = wasm_i8x16_splat(1);
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

        while (k >= MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::PackedK) {

            v128_t BytesRow0 = wasm_v128_load64_zero(&b[0]);
            v128_t BytesRow1 = wasm_v128_load64_zero(&b[ldb]);
            v128_t BytesRow2 = wasm_v128_load64_zero(&b[ldb * 2]);
            v128_t BytesRow3 = wasm_v128_load64_zero(&b[ldb * 3]);

            MlasGemmU8X8CopyPackBProcessWasmRelaxedSimd(D, BytesRow0, BytesRow1, BytesRow2, BytesRow3, BitFlipVector, OnesByteBroadcast, ColumnSums);

            b += ldb * 4;
            D += 32;
            k -= 4;
        }

        if (k > 0) {

            v128_t BytesRow0 = wasm_v128_load64_zero(&b[0]);
            v128_t BytesRow1 = BitFlipVector;
            v128_t BytesRow2 = BitFlipVector;
            v128_t BytesRow3 = BitFlipVector;

            if (k >= 2) {
                BytesRow1 = wasm_v128_load64_zero(&b[ldb]);
            }

            if (k >= 3) {
                BytesRow2 = wasm_v128_load64_zero(&b[ldb * 2]);
            }

            MlasGemmU8X8CopyPackBProcessWasmRelaxedSimd(D, BytesRow0, BytesRow1, BytesRow2, BytesRow3, BitFlipVector, OnesByteBroadcast, ColumnSums);

            D += 32;
        }

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
        uint8_t PaddedMatrixBData[32];

        wasm_v128_store(&PaddedMatrixBData[0], BitFlipVector);
        wasm_v128_store(&PaddedMatrixBData[16], BitFlipVector);

        ColumnSums[0] = wasm_i64x2_const(0, 0);
        ColumnSums[1] = wasm_i64x2_const(0, 0);

        //
        // Interleave rows of matrix B using an intermediate zero padded stack
        // buffer and write to the packed buffer.
        //

        while (k >= MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::PackedK) {

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded[8] = bcopy[ldb];
                padded[16] = bcopy[ldb * 2];
                padded[24] = bcopy[ldb * 3];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            v128_t BytesRow0 = wasm_v128_load64_zero(&PaddedMatrixBData[0]);
            v128_t BytesRow1 = wasm_v128_load64_zero(&PaddedMatrixBData[8]);
            v128_t BytesRow2 = wasm_v128_load64_zero(&PaddedMatrixBData[16]);
            v128_t BytesRow3 = wasm_v128_load64_zero(&PaddedMatrixBData[24]);

            MlasGemmU8X8CopyPackBProcessWasmRelaxedSimd(D, BytesRow0, BytesRow1, BytesRow2, BytesRow3, BitFlipVector, OnesByteBroadcast, ColumnSums);

            b += ldb * 4;
            D += 32;
            k -= 4;
        }

        if (k > 0) {

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            wasm_v128_store(&PaddedMatrixBData[0], BitFlipVector);
            wasm_v128_store(&PaddedMatrixBData[16], BitFlipVector);

            if (k == 3) {
              do {
                  padded[0] = bcopy[0];
                  padded[8] = bcopy[ldb];
                  padded[16] = bcopy[ldb * 2];
                  padded++;
                  bcopy++;
              } while (padded < padded_end);
            } else if (k == 2) {
              do {
                  padded[0] = bcopy[0];
                  padded[8] = bcopy[ldb];
                  padded++;
                  bcopy++;
              } while (padded < padded_end);
            } else {
              do {
                  padded[0] = bcopy[0];
                  padded++;
                  bcopy++;
              } while (padded < padded_end);
            }

            v128_t BytesRow0 = wasm_v128_load64_zero(&PaddedMatrixBData[0]);
            v128_t BytesRow1 = wasm_v128_load64_zero(&PaddedMatrixBData[8]);
            v128_t BytesRow2 = wasm_v128_load64_zero(&PaddedMatrixBData[16]);
            v128_t BytesRow3 = wasm_v128_load64_zero(&PaddedMatrixBData[24]);

            MlasGemmU8X8CopyPackBProcessWasmRelaxedSimd(D, BytesRow0, BytesRow1, BytesRow2, BytesRow3, BitFlipVector, OnesByteBroadcast, ColumnSums);
        }

        wasm_v128_store(&ColumnSumBuffer[0], ColumnSums[0]);
        wasm_v128_store(&ColumnSumBuffer[4], ColumnSums[1]);
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8MultiplyAccumulateRowWasmRelaxedSimd(
    v128_t ABroadcast,
    const uint8_t* B,
    v128_t Accumulators[2]
)
{
    v128_t BElements0 = wasm_v128_load(&B[0]);
    v128_t BElements1 = wasm_v128_load(&B[16]);

    Accumulators[0] = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(BElements0, ABroadcast, Accumulators[0]);
    Accumulators[1] = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(BElements1, ABroadcast, Accumulators[1]);
}


template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD>(
    const MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::PackedBType* B,
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

        const uint8_t* a = A;
        size_t k = PackedCountK;

        while (k >= 4) {

            v128_t AElements = wasm_v128_load((v128_t*)a);
            v128_t ABroadcast;

            ABroadcast = wasm_i32x4_shuffle(AElements, wasm_i32x4_splat(0), 0, 0, 0, 0);
            MlasGemmU8X8MultiplyAccumulateRowWasmRelaxedSimd(ABroadcast, &B[0], Accumulators);

            ABroadcast = wasm_i32x4_shuffle(AElements, wasm_i32x4_splat(0), 1, 1, 1, 1);
            MlasGemmU8X8MultiplyAccumulateRowWasmRelaxedSimd(ABroadcast, &B[32], Accumulators);

            ABroadcast = wasm_i32x4_shuffle(AElements, wasm_i32x4_splat(0), 2, 2, 2, 2);
            MlasGemmU8X8MultiplyAccumulateRowWasmRelaxedSimd(ABroadcast, &B[64], Accumulators);

            ABroadcast = wasm_i32x4_shuffle(AElements, wasm_i32x4_splat(0), 3, 3, 3, 3);
            MlasGemmU8X8MultiplyAccumulateRowWasmRelaxedSimd(ABroadcast, &B[96], Accumulators);

            a += 4 * 4;
            B += 4 * 32;
            k -= 4;
        }

        while (k > 0) {

            v128_t ABroadcast = wasm_i32x4_splat(*((int32_t*)a));
            MlasGemmU8X8MultiplyAccumulateRowWasmRelaxedSimd(ABroadcast, &B[0], Accumulators);

            a += 4;
            B += 32;
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

const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8X8DispatchWasmRelaxedSimd = {
    MlasGemmQuantOperation<MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD>,
    nullptr,
    nullptr,
    MLAS_GEMM_U8X8_KERNEL_WASMRELAXEDSIMD::PackedK,
    0,
    4 // multiple of kernel stride M
};
