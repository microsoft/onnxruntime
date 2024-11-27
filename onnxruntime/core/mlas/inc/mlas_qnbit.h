/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_qnbit.h

Abstract:

    This module contains the public data structures and procedure prototypes
    for blocked n-bit quantized GEMM.

    N-bit block quantization is used to compress weight tensors of large
    language models.

--*/

#pragma once

#include "mlas.h"
#include "mlas_gemm_postprocessor.h"

/**
 * @brief Define compute types of block quantization, in order of decreasing accuracy.
 */
typedef enum {
    SQNBIT_CompFp32,      /*!< input fp32, accumulator fp32 */
    HQNBIT_CompFp16,      /*!< input fp16, accumulator fp16 */
    BHQNBIT_CompBf16,     /*!< input bf16, accumulator fp32 */
    SQNBIT_CompInt8,      /*!< input int8, accumulator int32, input fp32 */
    HQNBIT_CompInt8,      /*!< input int8, accumulator int32, input fp16 */
} MLAS_QNBIT_GEMM_COMPUTE_TYPE;

/**
 * @brief Data parameters for float/n-bit quantized int GEMM routine.
 *
 * @tparam  T   data type of input A
 */
template <typename T>
struct MLAS_QNBIT_GEMM_DATA_PARAMS {
    const T* A = nullptr;                       ///< address of A (float32/16 matrix)
    size_t lda = 0;                                 ///< leading dimension of A
    const void* QuantBDataWorkspace;                ///< address of quantized B (quantized n-bit int values)
    const std::byte* PackedQuantBData = nullptr;    /// address of packed quantized B data
    const T* QuantBScale = nullptr;             ///< address of scale values of quantized B, one per block
    const void* QuantBZeroPoint = nullptr;          ///< optional address of zero point values of quantized B, one per block
    const T* QuantBBlkSum = nullptr;            ///< optional address of scale * zp, one per block
    const T* Bias = nullptr;                    ///< optional address of Bias, vector size N
    T* C = nullptr;                             ///< address of result matrix
    size_t ldc = 0;                                 ///< leading dimension of C

    ///< optional post processing to apply to result matrix
    MLAS_GEMM_POSTPROCESSOR<T>* PostProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a float32/16 matrix
 *        B must be a quantized and packed n-bit int matrix
 *
 *        Call MlasIsQNBitGemmAvailable() with the same parameters to determine whether this function may be called.
 *
 *        Call MlasQNBitGemmPackQuantBDataSize() with the same parameters to determine whether
 *          MLAS_QNBIT_GEMM_DATA_PARAMS::QuantBData in `DataParams` should point to a buffer packed with
 *          MlasQNBitGemmPackQuantBData().
 *
 *        Call MlasQNBitGemmBatchWorkspaceSize() with the same parameters to determine whether `Workspace` should
 *          point to an intermediate workspace buffer.
 *
 * @tparam          T               data type of input A
 * @param[in]       M               row size of matrix A and C
 * @param[in]       N               column size of matrix B and C
 * @param[in]       K               column size of matrix A and row size of matrix B
 * @param[in]       BatchN          number of batches
 * @param[in]       BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]       BlkLen          number of quantized values per block
 * @param[in]       ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
 * @param[inout]    DataParams      An array (size BatchN) of parameter blocks
 * @param[in]       Workspace       Address of intermediate workspace buffer.
                                    If MlasQNBitGemmBatchWorkspaceSize() returns a non-zero value, this must be a
                                    buffer with at least that many bytes. Otherwise, it may be nullptr.
 * @param[in]       ThreadPool      optional thread pool to use
 */
template <typename T>
void MLASCALL
MlasQNBitGemmBatch(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<T>* DataParams,
    void* Workspace,
    MLAS_THREADPOOL* ThreadPool = nullptr
);

/**
 * @brief Determines whether a float32/16 quantized n-bit int GEMM implementation is available on the current platform.
 *
 * @param[in]   BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen          number of quantized values per block
 * @param[in]   ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
 */
bool MLASCALL
MlasIsQNBitGemmAvailable(
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
);

/**
 * @brief Gets the size in bytes of the intermediate workspace buffer required by the float32/quantized n-bit int GEMM
 * implementation. If zero, no intermediate workspace is required.
 *
 * @param[in]   M               row size of matrix A and C
 * @param[in]   N               column size of matrix B and C
 * @param[in]   K               column size of matrix A and row size of matrix B
 * @param[in]   BatchN          number of batches
 * @param[in]   BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen          number of quantized values per block
 * @param[in]   ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
 */
size_t MLASCALL
MlasQNBitGemmBatchWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
);

/**
 * @brief Gets the size in bytes of the packed quantized B data.
 * If non-zero, the quantized B data must first be packed by calling MlasQNBitGemmPackQuantBData() with a buffer of
 * this size, and then that packed quantized B data buffer must be passed to MlasQNBitGemmBatch().
 * If zero, MlasQNBitGemmPackQuantBData() must not be called and the quantized B data must be directly passed to
 * MlasQNBitGemmBatch().
 *
 * @param[in]   N               column size of matrix B and C
 * @param[in]   K               column size of matrix A and row size of matrix B
 * @param[in]   BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen          number of quantized values per block
 * @param[in]   ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
 */
size_t MLASCALL
MlasQNBitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
);

/**
 * @brief Packs the quantized B data in a format that the kernel expects.
 *
 * If the function is called without QuantBScale and QuantBZeroPoint,
 * it just packs QuantBData into PackedQuantBDataAndOrBlkSum.
 *
 * If the function is called with QuantBData, QuantBScale, and QuantBZeroPoint
 * additional BlkSum (Scale * zeropoint) is computed and stored at the second part of PackedQuantBDataAndOrBlkSum.
 *
 * Because ORT OpKernel::PrePack is called for each input (in this case, QuantBData,
 * QuantBScale, and QuantBZeroPoint) separately, this function may be called 3 times, first with QuantBData,
 * and then QuantBScale and QuantBZeroPoint. When the function is called with QuantBScale without QuantBZeroPoint,
 * BlkSum is computed with default zero point 8 and stored at the second part of PackedQuantBDataAndOrBlkSum.
 * If there is a third call with QuantBZeroPoint, BlkSum is recomputed/adjusted with provided zeropoint.
 *
 * @param[in]   N                               column size of matrix B and C
 * @param[in]   K                               column size of matrix A and row size of matrix B
 * @param[in]   BlkBitWidth                     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen                          number of quantized values per block
 * @param[in]   ComputeType                     GEMM compute type (e.g., multiplying float or int8 values)
 * @param[in]   QuantBData                      quantized B data
 * @param[in]   PackedQuantBDataAndOrBlkSum     buffer to store packed quantized B data and/or BlkSum
 * @param[in]   QuantBScale                     quantized B scale
 * @param[in]   has_zp_input                    whether QuantBZeroPoint is provided
 * @param[in]   QuantBZeroPoint                 quantized B zero point
 * @param[in]   ThreadPool          thread pool to use (no parallel if nullptr)
 */
void MLASCALL
MlasQNBitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const void* QuantBData,
    void* PackedQuantBDataAndOrBlkSum,
    const void* QuantBScale,
    bool has_zp_input,
    const void* QuantBZeroPoint,
    MLAS_THREADPOOL* ThreadPool
);
