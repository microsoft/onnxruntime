// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/threadpool.h"

#include <cfenv>
#include <cmath>

#if defined(_M_AMD64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#define MLAS_SUPPORTS_GEMM_U8X8
#endif

#if defined(_M_AMD64) || defined(__x86_64__)
#define MLAS_SUPPORTS_PACKED_GEMM_U8X8
#endif

namespace onnxruntime {

void QGemm(
    int M,
    int N,
    int K,
    const uint8_t* lhs_data,
    int lda,
    const uint8_t lhs_offset,
    const uint8_t* rhs_data,
    int ldb,
    const uint8_t rhs_offset,
    bool rhs_signed,
    int32_t* result_data,
    int ldc,
    concurrency::ThreadPool* thread_pool);

void QGemm(
    int M,
    int N,
    int K,
    const uint8_t* lhs_data,
    int lda,
    const uint8_t lhs_offset,
    const uint8_t* rhs_data,
    int ldb,
    const uint8_t rhs_offset,
    bool rhs_signed,
    float* result_data,
    int ldc,
    const float* result_scale,
    const float* bias,
    concurrency::ThreadPool* thread_pool);

inline float RoundHalfToEven(float input) {
  std::fesetround(FE_TONEAREST);
  auto result = std::nearbyintf(input);
  return result;
}

}  // namespace onnxruntime
