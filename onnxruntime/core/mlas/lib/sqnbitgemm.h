/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm.h

Abstract:

    This module includes kernel function prototypes and helper functions for
    implementing SQNBitGemm.

    SQNBitGemm is a matrix/matrix multiplication, A*B, where A is a float
    matrix and B is a n-bit quantized integer matrix. B is block quantized,
    meaning values of B are divided into blocks and each block has its own
    scale and optional zero point.

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
float
Q8BlkScale(const std::byte* BlkPtr)
{
    float Scale;
    memcpy(&Scale, BlkPtr, sizeof(float));
    return Scale;
}

MLAS_FORCEINLINE
void
Q8BlkSetScale(std::byte* BlkPtr, float Scale)
{
    memcpy(BlkPtr, &Scale, sizeof(float));
}

MLAS_FORCEINLINE
const int8_t*
Q8BlkData(const std::byte* BlkPtr)
{
    return reinterpret_cast<const int8_t*>(BlkPtr + sizeof(float));
}

MLAS_FORCEINLINE
int8_t*
Q8BlkMutableData(std::byte* BlkPtr)
{
    return reinterpret_cast<int8_t*>(BlkPtr + sizeof(float));
}

MLAS_FORCEINLINE
constexpr size_t
Q8BlkSize(size_t BlkLen)
{
    const size_t BlkSize = sizeof(float) + BlkLen * sizeof(int8_t);
    return BlkSize;
}

//
// Quantized int4 block helpers.

MLAS_FORCEINLINE
float
Q4BlkScale(const std::byte* BlkPtr) {
    float Scale;
    memcpy(&Scale, BlkPtr, sizeof(float));
    return Scale;
}

MLAS_FORCEINLINE
void
Q4BlkSetScale(std::byte* BlkPtr, float Scale)
{
    memcpy(BlkPtr, &Scale, sizeof(float));
}

MLAS_FORCEINLINE
int8_t
Q4BlkZeroPoint(const std::byte* BlkPtr) {
    int8_t zp;
    memcpy(&zp, BlkPtr + sizeof(float), sizeof(int8_t));
    return zp;
}

MLAS_FORCEINLINE
void
Q4BlkSetZeroPoint(std::byte* BlkPtr, int8_t ZeroPoint)
{
    memcpy(BlkPtr + sizeof(float), &ZeroPoint, sizeof(int8_t));
}

MLAS_FORCEINLINE
const std::byte*
Q4BlkData(const std::byte* BlkPtr)
{
    return BlkPtr + sizeof(float) + sizeof(int8_t);
}

MLAS_FORCEINLINE
std::byte*
Q4BlkMutableData(std::byte* BlkPtr)
{
    return BlkPtr + sizeof(float) + sizeof(int8_t);
}

MLAS_FORCEINLINE
constexpr size_t
Q4BlkSize(size_t BlkLen) {
    const size_t BlkSize = sizeof(float) + sizeof(int8_t) + (BlkLen + 1) / 2 * sizeof(std::byte);
    return BlkSize;
}

//
// Kernel dispatch structure.
//

struct MLAS_SQNBIT_GEMM_DISPATCH {
    //
    // CompFp32 kernel function prototypes.
    //

    /**
     * @brief Multiply float matrix A with quantized n-bit integer matrix B.
     *        B is block quantized and column major.
     *        This kernel handles the special case where M, the number of rows of A and C, is 1.
     *
     * @param       BlkLen              Number of values in a block.
     * @param       A                   Supplies the A matrix.
     * @param       QuantB              Supplies the quantized B matrix.
     *                                  Binary data containing block quantized int4 data, zero point and scale values.
     * @param[out]  C                   Supplies the output C matrix.
     * @param       CountN              Number of columns of B and C.
     * @param       CountK              Number of columns of A and rows of B.
     * @param       BlockStrideQuantB   Number of blocks between adjacent columns of the quantized B matrix.
     */
    typedef void(SQNBitGemmM1Kernel_BlkBitWidth4_CompFp32_Fn)(
        size_t BlkLen,
        const float* A,
        const std::byte* QuantB,
        float* C,
        size_t CountN,
        size_t CountK,
        size_t BlockStrideQuantB
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
     * @param       QuantB              Supplies the quantized B matrix.
     *                                  Binary data containing block quantized int4 data, zero point and scale values.
     * @param       CountN              Number of columns of B.
     * @param       CountK              Number of rows of B.
     * @param       BlockStrideQuantB   Number of blocks between adjacent columns of the quantized B matrix.
     */
    typedef void(QNBitBlkDequantBForSgemm_BlkBitWidth4_CompFp32_Fn)(
        size_t BlkLen,
        float* FpData,
        const std::byte* QuantB,
        size_t CountN,
        size_t CountK,
        size_t BlockStrideQuantB
    );

    QNBitBlkDequantBForSgemm_BlkBitWidth4_CompFp32_Fn* QNBitBlkDequantBForSgemm_BlkBitWidth4_CompFp32 = nullptr;

    //
    // CompInt8 kernel function prototypes.
    //

    /**
     * @brief Multiply quantized int8 matrix A with quantized n-bit integer matrix B.
     *        A and B are block quantized and B is column major.
     *        This kernel handles the special case where M, the number of rows of A and C, is 1.
     *
     * @param       BlkLen              Number of values in a block.
     * @param       QuantA              Supplies the quantized A matrix.
                                        Binary data containing block quantized int8 data and scale values.
     * @param       QuantB              Supplies the quantized B matrix.
     *                                  Binary data containing block quantized int4 data, zero point and scale values.
     * @param[out]  C                   Supplies the output C matrix.
     * @param       CountN              Number of columns of B and C.
     * @param       CountK              Number of columns of A and rows of B.
     * @param       BlockStrideQuantB   Number of blocks between adjacent columns of the quantized B matrix.
     */
    typedef void(SQNBitGemmM1Kernel_BlkBitWidth4_CompInt8_Fn)(
        size_t BlkLen,
        const std::byte* QuantA,
        const std::byte* QuantB,
        float* C,
        size_t CountN,
        size_t CountK,
        size_t BlockStrideQuantB
    );

    SQNBitGemmM1Kernel_BlkBitWidth4_CompInt8_Fn* SQNBitGemmM1Kernel_BlkBitWidth4_CompInt8 = nullptr;

    /**
     * @brief Block quantize values from one row of matrix A from float to int8.
     *
     * @param       BlkLen  Number of values in a block.
     * @param       A       Supplies the A matrix.
     * @param       CountK  Number of columns of A.
     * @param[out]  QuantA  Supplies the output quantized A matrix.
     *                      Binary data containing block quantized int8 data and scale values.
     */
    typedef void(QuantizeARow_CompInt8_Fn)(
        size_t BlkLen,
        const float* A,
        size_t CountK,
        std::byte* QuantA
    );

    QuantizeARow_CompInt8_Fn* QuantizeARow_CompInt8 = nullptr;
};
