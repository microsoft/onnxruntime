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
 * @brief Data parameters for float/n-bit quantized int GEMM routine.
 */
struct MLAS_SQNBIT_GEMM_DATA_PARAMS {
    const float* A = nullptr;                ///< address of A (float32 matrix)
    const void* PackedBData = nullptr;           ///< address of B (quantized and packed n-bit int values)
    const float* PackedBScale = nullptr;     ///< address of scale values of quantized B, one per block
    const void* PackedBZeroPoint = nullptr;  ///< optional address of zero point values of quantized B, one per block
    bool IsBPacked = false;                  ///< whether B values are packed in the optimal format for the computation
    const float* Bias = nullptr;             ///< optional address of Bias, vector size N
    float* C = nullptr;                      ///< address of result matrix
    size_t lda = 0;                          ///< leading dimension of A
    size_t ldc = 0;                          ///< leading dimension of C

    ///< optional post processing to apply to result matrix
    MLAS_GEMM_POSTPROCESSOR<float>* PostProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed n-bit int matrix
 *
 * @param[in]  M                row size of matrix A and C
 * @param[in]  N                column size of matrix B and C
 * @param[in]  K                column size of matrix A and row size of matrix B
 * @param[in]  BatchN           number of batches
 * @param[in]  BlkBitWidth      quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]  BlkSize          number of quantized values per block
 * @param[inout]  DataParams    An array (size BatchN) of parameter blocks
 * @param[in]  ThreadPool       optional thread pool to use
 */
void MLASCALL
MlasSQNBitGemmBatch(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const size_t BlkBitWidth,
    const size_t BlkSize,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool = nullptr
);
