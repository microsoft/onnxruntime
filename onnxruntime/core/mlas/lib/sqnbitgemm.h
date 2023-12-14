/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm.h

Abstract:

    This module includes:  // TODO update

    - Declaration of the set of template functions used to implement a kernel
    for a matrix/matrix multiplication, A*B, where A is a float matrix and B is
    a n-bit quantized integer matrix (QNBitGemm).

    - A shared kernel driver function template, MlasSQNBitGemmOperation.

    - Kernel dispatch structure.

    The B matrix is block quantized, which means that its values are grouped
    into blocks which each have one scale and optional zero point. Each
    quantized value in B is n-bits wide.

--*/

#pragma once

#include "mlas_qnbit.h"
#include "mlasi.h"

constexpr MLAS_FORCEINLINE size_t
MlasQNBitBlkDataSizeInBytes(size_t BlkBitWidth, size_t BlkLen)
{
    return BlkLen * BlkBitWidth / 8;
}

template <size_t BlkBitWidth>
constexpr MLAS_FORCEINLINE size_t
MlasQNBitZeroPointsForBlksSizeInBytes(size_t BlkCount)
{
    if constexpr (BlkBitWidth <= 4) {
        return MlasDivRoundup(BlkCount, 2);  // 2 blocks per byte
    } else {
        return BlkCount;
    }
}

//
// Quantized int8 block helpers.
//

MLAS_FORCEINLINE
const float&
Q8BlkScale(const std::byte* BlkPtr)
{
    return *reinterpret_cast<const float*>(BlkPtr);
}

MLAS_FORCEINLINE
float&
Q8BlkScale(std::byte* BlkPtr)
{
    return *reinterpret_cast<float*>(BlkPtr);
}

MLAS_FORCEINLINE
const int8_t*
Q8BlkData(const std::byte* BlkPtr)
{
    return reinterpret_cast<const int8_t*>(BlkPtr + sizeof(float));
}

MLAS_FORCEINLINE
int8_t*
Q8BlkData(std::byte* BlkPtr)
{
    return reinterpret_cast<int8_t*>(BlkPtr + sizeof(float));
}

MLAS_FORCEINLINE
constexpr size_t
Q8BlkSize(size_t BlkLen)
{
    const size_t BlkSize = sizeof(float) + BlkLen * sizeof(int8_t);
    // Currently, the strictest alignment requirement of a block is for a float.
    // Ensure contiguous blocks are suitably aligned.
    // assert(BlkSize % alignof(float) == 0);  // TODO needs include, put it in .cpp?
    return BlkSize;
}

MLAS_FORCEINLINE
constexpr size_t
Q8BlkAlignment(size_t BlkLen)
{
    MLAS_UNREFERENCED_PARAMETER(BlkLen);
    return alignof(float);
}

//
// Kernel dispatch structure.
//

struct MLAS_SQNBIT_GEMM_DISPATCH {
    //
    // CompFp32 kernels
    //

    /**
     * @brief Multiply float matrix A with quantized n-bit integer matrix B.
     *        B is block quantized and column major.
     *        This kernel handles the special case where M, the number of rows of A and C, is 1.
     *
     * @param       BlkLen              Number of values in a block.
     * @param       A                   Supplies the A matrix.
     * @param       QuantBData          Supplies the quantized B matrix block data.
     * @param       QuantBScale         Supplies the quantized B matrix block scale values.
     * @param       QuantBZeroPoint     Supplies the quantized B matrix block zero point values. Optional.
     * @param[out]  C                   Supplies the output C matrix.
     * @param       CountN              Number of columns of B and C.
     * @param       CountK              Number of columns of A and rows of B.
     * @param       BlockStrideQuantB   Number of blocks between adjacent columns of the quantized B matrix.
     * @param       Bias                Bias vector of length N.
     */
    typedef void(SQNBitGemmM1Kernel_BlkBitWidth4_CompFp32_Fn)(
        size_t BlkLen,
        const float* A,
        const uint8_t* QuantBData,
        const float* QuantBScale,
        const uint8_t* QuantBZeroPoint,
        float* C,
        size_t CountN,
        size_t CountK,
        size_t BlockStrideQuantB,
        const float* Bias
    );

    SQNBitGemmM1Kernel_BlkBitWidth4_CompFp32_Fn* SQNBitGemmM1Kernel_BlkBitWidth4_CompFp32 = nullptr;

    /**
     * @brief Dequantize B into the format expected by the Sgemm kernel.
     *        B is block quantized and column major.
     *        This is equivalent to dequantizing B and then running
     *        MlasSgemmCopyPackB.
     *
     * @param       BlkLen              Number of values in a block.
     * @param[out]  FpData              Supplies the output buffer for the dequantized B float data.
     * @param       QuantBData          Supplies the quantized B matrix block data.
     * @param       QuantBScale         Supplies the quantized B matrix block scale values.
     * @param       QuantBZeroPoint     Supplies the quantized B matrix block zero point values. Optional.
     * @param       CountN              Number of columns of B.
     * @param       CountK              Number of rows of B.
     * @param       BlockStrideQuantB   Number of blocks between adjacent columns of the quantized B matrix.
     */
    typedef void(QNBitBlkDequantBForSgemm_BlkBitWidth4_CompFp32_Fn)(
        size_t BlkLen,
        float* FpData,
        const uint8_t* QuantBData,
        const float* QuantBScale,
        const uint8_t* QuantBZeroPoint,
        size_t CountN,
        size_t CountK,
        size_t BlockStrideQuantB
    );

    QNBitBlkDequantBForSgemm_BlkBitWidth4_CompFp32_Fn* QNBitBlkDequantBForSgemm_BlkBitWidth4_CompFp32 = nullptr;

    //
    // CompInt8 kernels
    //

    typedef void(SQNBitGemmM1Kernel_BlkBitWidth4_CompInt8_Fn)(
        size_t BlkLen,
        const std::byte* QuantA,
        const uint8_t* QuantBData,
        const float* QuantBScale,
        const uint8_t* QuantBZeroPoint,
        float* C,
        size_t CountN,
        size_t CountK,
        size_t BlockStrideQuantB,
        const float* Bias
    );

    SQNBitGemmM1Kernel_BlkBitWidth4_CompInt8_Fn* SQNBitGemmM1Kernel_BlkBitWidth4_CompInt8 = nullptr;

    typedef void(QuantizeA_CompInt8_Fn)(
        size_t BlkLen,
        const float* A,
        size_t CountM,
        size_t CountK,
        size_t lda,
        std::byte* QuantA
    );

    QuantizeA_CompInt8_Fn* QuantizeA_CompInt8 = nullptr;
};
