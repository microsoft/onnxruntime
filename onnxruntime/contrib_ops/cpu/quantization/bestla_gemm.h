/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    bestla_gemm.h

Abstract:

    Currently only support Q4 gemm.
--*/

#pragma once

#include <stdint.h>
#include <cstddef>

/**
 * @brief Data parameters for NBits GEMM routine
 *        C = A * B
 *        A, C must be a float32 matrix
 *        B must be a packed nbits blob
 *        All except C are [in] parameters
 */
struct NS_SQNBITS_GEMM_DATA_PACKED_PARAMS {
  const float* A = nullptr; /**< address of A (float32 matrix)*/
  const void* B = nullptr;  /**< address of B (packed nbits blob)*/
  float* C = nullptr;       /**< address of result matrix */
  size_t lda = 0;           /**< leading dimension of A */
  size_t ldc = 0;           /**< leading dimension of C*/
};

size_t NSQ4GemmPackBSize(size_t N, size_t K, size_t BlkSize, bool isAsym, int64_t accuracy_level);

bool NSQ4GemmPackB(void* PackedBuf, const uint8_t* QData, const float* Scale, const uint8_t* Zp, size_t N, size_t K,
                   size_t ldb, size_t BlkSize, bool isAsym, bool lastCall, int64_t CompType, void* ThreadPool);

bool NSQ4GemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, void* ThreadPool);

bool NSSQ4GemmBatchDriver(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                          const NS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams, int8_t* WorkSpace, void* ThreadPool);

size_t NSSQ4GemmBatchWorkspaceSize(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                                   const NS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams);
