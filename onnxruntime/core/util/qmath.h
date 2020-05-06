// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/threadpool.h"

#if defined(_M_AMD64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#define MLAS_SUPPORTS_GEMM_U8X8
#endif

namespace onnxruntime {

void QGemmu8s8_s32(
    int M,
    int N,
    int K,
    const uint8_t* lhs_data,
    int lda,
    const uint8_t lhs_offset,
    const int8_t* rhs_data,
    int ldb,
    const int8_t rhs_offset,
    int32_t* result_data,
    int ldc,
    concurrency::ThreadPool* thread_pool);

void QGemmu8u8_s32(
    int M,
    int N,
    int K,
    const uint8_t* lhs_data,
    int lda,
    const uint8_t lhs_offset,
    const uint8_t* rhs_data,
    int ldb,
    const uint8_t rhs_offset,
    int32_t* result_data,
    int ldc,
    concurrency::ThreadPool* thread_pool);

}  // namespace onnxruntime
