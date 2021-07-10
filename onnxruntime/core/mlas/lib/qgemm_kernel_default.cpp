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

struct MLAS_GEMM_U8X8_KERNEL_DEFAULT
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 16, 128, 128 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 16, 128, 128 };
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_DEFAULT::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedStrides;

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmU8X8FixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_DEFAULT>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_DEFAULT::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8X8_KERNEL_DEFAULT>(
    MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK - 1) & ~(MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK - 1);

    //
    // Process a single row of matrix A in a loop.
    //

    while (CountM-- > 0) {

        int32_t RowSum = 0;

        for (size_t k = 0; k < CountK; k++) {

            uint8_t a0 = A[k];
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
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_DEFAULT>(
    MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK - 1) & ~(MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK - 1);
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
MlasGemmU8X8Kernel<MLAS_GEMM_U8X8_KERNEL_DEFAULT>(
    const MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedBType* B,
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

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8X8DispatchDefault = {
    MlasGemmU8X8Operation<MLAS_GEMM_U8X8_KERNEL_DEFAULT>,
    nullptr,
    nullptr,
    MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK,
    0,
};
