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

// TODO add documentation
enum MLAS_SQNBITGEMM_COMPUTE_TYPE {
    CompFp32,  // fp32 A, fp32 accumulator
    CompInt8,  // int8 A, int32 accumulator
};

/**
 * @brief Data parameters for float/n-bit quantized int GEMM routine.
 */
struct MLAS_SQNBIT_GEMM_DATA_PARAMS {
    const float* A = nullptr;                ///< address of A (float32 matrix)
    size_t lda = 0;                          ///< leading dimension of A
    const void* QuantBData = nullptr;        ///< address of quantized B (quantized n-bit int values)
    const float* QuantBScale = nullptr;      ///< address of scale values of quantized B, one per block
    const void* QuantBZeroPoint = nullptr;   ///< optional address of zero point values of quantized B, one per block
    const float* Bias = nullptr;             ///< optional address of Bias, vector size N
    float* C = nullptr;                      ///< address of result matrix
    size_t ldc = 0;                          ///< leading dimension of C

    ///< optional post processing to apply to result matrix
    MLAS_GEMM_POSTPROCESSOR<float>* PostProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed n-bit int matrix
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
                                    If MlasSQNBitGemmWorkspaceSize() returns a non-zero value, this should be a buffer
                                    with at least that many bytes. Otherwise, it can be nullptr.
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
    MLAS_SQNBITGEMM_COMPUTE_TYPE ComputeType,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    void* Workspace,
    MLAS_THREADPOOL* ThreadPool = nullptr
);

/**
 * @brief Determines whether a float32/quantized n-bit int GEMM implementation is available on the current platform.
 * @param[in]   BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen          number of quantized values per block
 * TODO update param doc
 */
bool MLASCALL
MlasIsSQNBitGemmAvailable(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBITGEMM_COMPUTE_TYPE ComputeType
);

/**
 * @brief Gets the size in bytes of the intermediate workspace buffer required by the float32/quantized n-bit int GEMM
 * implementation. If zero, no intermediate workspace is required.
 * // TODO update param doc
 */
size_t MLASCALL
MlasSQNBitGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBITGEMM_COMPUTE_TYPE ComputeType
);
