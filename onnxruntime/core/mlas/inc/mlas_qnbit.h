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
    CompUndef = 0, /*!< undef */
    CompFp32,      /*!< input fp32, accumulator fp32 */
    CompFp16,      /*!< input fp16, accumulator fp16 */
    CompBf16,      /*!< input bf16, accumulator fp32 */
    CompInt8,      /*!< input int8, accumulator int32 */

    // special values that should be the first and last actual values

    CompMostAccurate = CompUndef,
    CompLeastAccurate = CompInt8,
} MLAS_SQNBIT_GEMM_COMPUTE_TYPE;

/**
 * @brief Data parameters for float/n-bit quantized int GEMM routine.
 */
struct MLAS_SQNBIT_GEMM_DATA_PARAMS {
    const float* A = nullptr;               ///< address of A (float32 matrix)
    size_t lda = 0;                         ///< leading dimension of A
    const void* QuantBData = nullptr;       ///< address of quantized B (quantized n-bit int values)
    const float* QuantBScale = nullptr;     ///< address of scale values of quantized B, one per block
    const void* QuantBZeroPoint = nullptr;  ///< optional address of zero point values of quantized B, one per block
    const float* Bias = nullptr;            ///< optional address of Bias, vector size N
    float* C = nullptr;                     ///< address of result matrix
    size_t ldc = 0;                         ///< leading dimension of C

    ///< optional post processing to apply to result matrix
    MLAS_GEMM_POSTPROCESSOR<float>* PostProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed n-bit int matrix
 *
 *        Call MlasIsSQNBitGemmAvailable() with the same parameters to determine whether this function may be called.
 *
 *        Call MlasSQNBitGemmPackQuantBDataSize() with the same parameters to determine whether
 *          MLAS_SQNBIT_GEMM_DATA_PARAMS::QuantBData in `DataParams` should point to a buffer packed with
 *          MlasSQNBitGemmPackQuantBData().
 *
 *        Call MlasSQNBitGemmBatchWorkspaceSize() with the same parameters to determine whether `Workspace` should
 *          point to an intermediate workspace buffer.
 *
 * @param[in]       M               row size of matrix A and C
 * @param[in]       N               column size of matrix B and C
 * @param[in]       K               column size of matrix A and row size of matrix B
 * @param[in]       BatchN          number of batches
 * @param[in]       BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]       BlkLen          number of quantized values per block
 * @param[in]       ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
 * @param[inout]    DataParams      An array (size BatchN) of parameter blocks
 * @param[in]       Workspace       Address of intermediate workspace buffer.
                                    If MlasSQNBitGemmBatchWorkspaceSize() returns a non-zero value, this must be a
                                    buffer with at least that many bytes. Otherwise, it may be nullptr.
 * @param[in]       ThreadPool      optional thread pool to use
 */
void MLASCALL
MlasSQNBitGemmBatch(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    void* Workspace,
    MLAS_THREADPOOL* ThreadPool = nullptr
);

/**
 * @brief Determines whether a float32/quantized n-bit int GEMM implementation is available on the current platform.
 *
 * @param[in]   BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen          number of quantized values per block
 * @param[in]   ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
 */
bool MLASCALL
MlasIsSQNBitGemmAvailable(
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
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
MlasSQNBitGemmBatchWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
);

/**
 * @brief Gets the size in bytes of the packed quantized B data.
 * If non-zero, the quantized B data must first be packed by calling MlasSQNBitGemmPackQuantBData() with a buffer of
 * this size, and then that packed quantized B data buffer must be passed to MlasSQNBitGemmBatch().
 * If zero, MlasSQNBitGemmPackQuantBData() must not be called and the quantized B data must be directly passed to
 * MlasSQNBitGemmBatch().
 *
 * @param[in]   N               column size of matrix B and C
 * @param[in]   K               column size of matrix A and row size of matrix B
 * @param[in]   BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen          number of quantized values per block
 * @param[in]   ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
 */
size_t MLASCALL
MlasSQNBitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
);

/**
 * @brief Packs the quantized B data in a format that the kernel expects.
 *
 * @param[in]   N                   column size of matrix B and C
 * @param[in]   K                   column size of matrix A and row size of matrix B
 * @param[in]   BlkBitWidth         quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen              number of quantized values per block
 * @param[in]   ComputeType         GEMM compute type (e.g., multiplying float or int8 values)
 * @param[in]   QuantBData          quantized B data
 * @param[out]  PackedQuantBData    packed quantized B data
 * @param[in]   ThreadPool          optional thread pool to use
 */
void MLASCALL
MlasSQNBitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const void* QuantBData,
    void* PackedQuantBData,
    MLAS_THREADPOOL* ThreadPool = nullptr
);
