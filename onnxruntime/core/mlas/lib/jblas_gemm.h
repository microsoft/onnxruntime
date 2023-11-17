/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    jblas_gemm.h

Abstract:

    int4 block quantization gemm kernel template declarations.

    Int4 block quantization is used to compress weight tensors of large
    language models. It takes a number (must be multiple of 32) of floating
    point values, calculates their quantization parameters, and saves
    the parameters and the quantized data in a blob.
--*/
#include "mlas_qnbit.h"

size_t
JblasQ4GemmPackBSize(size_t N, size_t K, size_t BlkSize, bool isAsym, MLAS_COMPUTE_TYPE CompType);

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
    MLAS_COMPUTE_TYPE CompType,
    MLAS_THREADPOOL* ThreadPool
);

bool
JblasQ4GemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, MLAS_THREADPOOL* ThreadPool);

bool
JblasQ4GemmBatchDriver(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_NBITS_GEMM_DATA_SIMPLE_PARAMS* DataParams,
    int8_t* WorkSpace,
    MLAS_THREADPOOL* ThreadPool
);