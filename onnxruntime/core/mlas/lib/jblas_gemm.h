/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    jblas_gemm.h

Abstract:

    Currently only support Q4 gemm.
--*/

#pragma once

#include "mlas_qnbit.h"

size_t
JblasQ4GemmPackBSize(size_t N, size_t K, size_t BlkSize, bool isAsym, MLAS_SQNBIT_COMPUTE_TYPE CompType);

bool
JblasQ4GemmPackB(
    void* PackedBuf,
    const uint8_t* QData,
    const float* Scale,
    const uint8_t* Zp,
    size_t N,
    size_t K,
    size_t ldb,
    size_t BlkSize,
    bool isAsym,
    bool lastCall,
    MLAS_SQNBIT_COMPUTE_TYPE CompType,
    MLAS_THREADPOOL* ThreadPool
);

bool
JblasQ4GemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb
	, MLAS_THREADPOOL* ThreadPool);

bool
JblasSQ4GemmBatchDriver(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams,
    int8_t* WorkSpace,
    MLAS_THREADPOOL* ThreadPool
);

size_t
JblasSQ4GemmBatchWorkspaceSize(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams
);
