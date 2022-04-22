/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_default.cpp

Abstract:

    This module implements default QGEMM kernel.

--*/

#include "mlasi.h"
#include "qgemm.h"

struct MLAS_GEMM_QUANT_KERNEL_DEFAULT
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 16, 128, 128 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 16, 128, 128 };
};

constexpr size_t MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_DEFAULT::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedStrides;

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_QUANT_KERNEL_DEFAULT>(
    int32_t ZeroPointA,
    bool AIsSigned
    )
{
    if (AIsSigned) {
        ZeroPointA = (uint8_t)(ZeroPointA ^ 0x80);
    }

    return ZeroPointA;
}

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_QUANT_KERNEL_DEFAULT>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_QUANT_KERNEL_DEFAULT::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_QUANT_KERNEL_DEFAULT>(
    MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    const size_t AlignedCountK = (CountK + MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedK - 1) &
                                 ~(MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedK - 1);

    const uint8_t BitFlipValue = (AIsSigned ? 0x80 : 0);

    //
    // Process a single row of matrix A in a loop.
    //

    while (CountM-- > 0) {

        int32_t RowSum = 0;

        for (size_t k = 0; k < CountK; k++) {

            uint8_t a0 = A[k] ^ BitFlipValue;
            D[k] = a0;

            RowSum += a0;
        }

        for (size_t k = CountK; k < AlignedCountK; k++) {
            D[k] = 0;
        }

        *RowSumBuffer++ = RowSum;

        A += lda;
        D += AlignedCountK;
    }
}

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_DEFAULT>(
    MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedK - 1) & ~(MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedK - 1);
    const uint8_t BitFlipValue = (BIsSigned ? 0x80 : 0);

    //
    // Process a single column of matrix B in a loop.
    //

    while (CountN-- > 0) {

        const uint8_t* b = B;
        int32_t ColumnSum = 0;

        //
        // Transpose the data from matrix B to the packed buffer.
        //

        for (size_t k = 0; k < CountK; k++) {

            uint8_t b0 = b[0] ^ BitFlipValue;
            D[k] = b0;

            ColumnSum += b0;

            b += ldb;
        }

        for (size_t k = CountK; k < AlignedCountK; k++) {
            D[k] = 0;
        }

        *ColumnSumBuffer++ = ColumnSum;

        B += 1;
        D += AlignedCountK;
    }
}

template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_QUANT_KERNEL_DEFAULT>(
    const MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedAType* A,
    const MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedBType* B,
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

    //
    // Process a single column of matrix B in a loop.
    //

    while (CountN-- > 0) {

        int32_t Accumulator = *RowSumBuffer;

        if (ZeroPointB != nullptr) {
            Accumulator *= *ZeroPointB++;
        }

        Accumulator += *ColumnSumBuffer++;

        const auto* a = A;

        for (size_t k = 0; k < PackedCountK; k++) {

            Accumulator += a[0] * B[0];
            Accumulator += a[1] * B[1];
            Accumulator += a[2] * B[2];
            Accumulator += a[3] * B[3];

            a += 4;
            B += 4;
        }

        if (!ZeroMode) {
            Accumulator += C[0];
        }

        C[0] = Accumulator;
        C += 1;
    }

    return 1;
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemmQuantDispatchDefault = {
    MlasGemmQuantOperation<MLAS_GEMM_QUANT_KERNEL_DEFAULT>,
    nullptr,
    nullptr,
    MLAS_GEMM_QUANT_KERNEL_DEFAULT::PackedK,
    0,
};
