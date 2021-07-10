/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_avx2.cpp

Abstract:

    This module implements QGEMM kernels for avx2.

--*/

#include "mlasi.h"
#include "qgemm.h"

//
// Stores a vector to transpose a 4x4 byte vector using vpshufb.
//

MLAS_INTERNAL_DATA MLAS_DECLSPEC_ALIGN(const uint8_t MlasTranspose4x4BytesAvx[16], 16) =
{ 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

//
// Define the prototypes of the AVX2/AVX512 routines written in assembly.
//

extern "C" {

    void
    MLASCALL
    MlasGemmU8S8CopyPackAAvx2(
        uint8_t* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumBuffer
        );

    void
    MLASCALL
    MlasGemmU8S8CopyPackBAvx2(
        uint8_t* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumBuffer,
        bool BIsSigned
        );

    void
    MLASCALL
    MlasGemmU8U8CopyPackAAvx2(
        int16_t* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumBuffer
        );

    void
    MLASCALL
    MlasGemmU8U8CopyPackBAvx2(
        uint8_t* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumBuffer
        );
}

struct MLAS_GEMM_U8S8_KERNEL_AVX2
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 24, 256, 128 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 48, 256, 384 };
};

constexpr size_t MLAS_GEMM_U8S8_KERNEL_AVX2::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8S8_KERNEL_AVX2::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8S8_KERNEL_AVX2::PackedStrides;

template<>
MLAS_FORCEINLINE
bool
MlasGemmU8X8TryGemvKernel<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    const uint8_t* A,
    const uint8_t* B,
    size_t ldb,
    int32_t* C,
    size_t CountK,
    size_t CountN,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        MlasPlatform.GemvU8S8Kernel(A, B, C, CountK, CountN, ldb);
        return true;
    }

    return false;
}

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmU8X8FixupZeroPointB<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (!BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8S8_KERNEL_AVX2::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    MLAS_GEMM_U8S8_KERNEL_AVX2::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    MlasGemmU8S8CopyPackAAvx2(D, A, lda, CountM, CountK, RowSumBuffer);
}

template<>
MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    MLAS_GEMM_U8S8_KERNEL_AVX2::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    MlasGemmU8S8CopyPackBAvx2(D, B, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);
}

template<>
MLAS_FORCEINLINE
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    const MLAS_GEMM_U8S8_KERNEL_AVX2::PackedAType* A,
    const MLAS_GEMM_U8S8_KERNEL_AVX2::PackedBType* B,
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
    return MlasPlatform.GemmU8S8Kernel(A, B, C, PackedCountK, CountM, CountN, ldc,
                                       RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8S8DispatchAvx2 = {
    MlasGemmU8X8Operation<MLAS_GEMM_U8S8_KERNEL_AVX2>,
    MlasGemmU8X8PackedOperation<MLAS_GEMM_U8S8_KERNEL_AVX2>,
    MlasGemmU8X8CopyPackB<MLAS_GEMM_U8S8_KERNEL_AVX2>,
    MLAS_GEMM_U8S8_KERNEL_AVX2::PackedK,
    MLAS_GEMM_U8S8_KERNEL_AVX2::PackedStrides.K,
};

struct MLAS_GEMM_U8U8_KERNEL_AVX2
{
    typedef int16_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 24, 256, 128 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 48, 256, 384 };
};

constexpr size_t MLAS_GEMM_U8U8_KERNEL_AVX2::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8U8_KERNEL_AVX2::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8U8_KERNEL_AVX2::PackedStrides;


template<>
MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8U8_KERNEL_AVX2>(
    MLAS_GEMM_U8U8_KERNEL_AVX2::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    MlasGemmU8U8CopyPackAAvx2(D, A, lda, CountM, CountK, RowSumBuffer);
}

template<>
MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8U8_KERNEL_AVX2>(
    MLAS_GEMM_U8U8_KERNEL_AVX2::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    MlasGemmU8U8CopyPackBAvx2(D, B, ldb, CountN, CountK, ColumnSumBuffer);
}

template<>
MLAS_FORCEINLINE
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8U8_KERNEL_AVX2>(
    const MLAS_GEMM_U8U8_KERNEL_AVX2::PackedAType* A,
    const MLAS_GEMM_U8U8_KERNEL_AVX2::PackedBType* B,
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
    return MlasPlatform.GemmU8U8Kernel(A, B, C, PackedCountK, CountM, CountN, ldc,
                                       RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8U8DispatchAvx2 = {
    MlasGemmU8X8Operation<MLAS_GEMM_U8U8_KERNEL_AVX2>,
    MlasGemmU8X8PackedOperation<MLAS_GEMM_U8U8_KERNEL_AVX2>,
    MlasGemmU8X8CopyPackB<MLAS_GEMM_U8U8_KERNEL_AVX2>,
    MLAS_GEMM_U8U8_KERNEL_AVX2::PackedK,
    MLAS_GEMM_U8U8_KERNEL_AVX2::PackedStrides.K,
};
