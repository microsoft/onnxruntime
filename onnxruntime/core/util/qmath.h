// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/threadpool.h"

#include <cfenv>
#include <cmath>

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
  if (!std::isfinite(input)) {
    return input;
  }
  // std::remainder returns x - n, where n is the integral value nearest to x. When |x - n| = 0.5, n is chosen to be even
  return input - std::remainderf(input, 1.f);
}

}  // namespace onnxruntime
