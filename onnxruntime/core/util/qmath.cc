// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/qmath.h"
#include "core/common/common.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

// fallback to gemmlowp when building for arm devices
#ifndef MLAS_SUPPORTS_GEMM_U8X8
#include "core/util/gemmlowp_common.h"
#endif

namespace onnxruntime {
template <>
void QGemm<uint8_t, int8_t, int32_t>(
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
    concurrency::ThreadPool* thread_pool) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8
  MlasGemm(M, N, K, lhs_data, lda, lhs_offset, rhs_data, ldb, rhs_offset, result_data, ldc, thread_pool);
#else
  ORT_UNUSED_PARAMETER(M);
  ORT_UNUSED_PARAMETER(N);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(lhs_data);
  ORT_UNUSED_PARAMETER(lda);
  ORT_UNUSED_PARAMETER(lhs_offset);
  ORT_UNUSED_PARAMETER(rhs_data);
  ORT_UNUSED_PARAMETER(ldb);
  ORT_UNUSED_PARAMETER(rhs_offset);
  ORT_UNUSED_PARAMETER(result_data);
  ORT_UNUSED_PARAMETER(ldc);
  ORT_UNUSED_PARAMETER(thread_pool);

  ORT_NOT_IMPLEMENTED("MatMulInteger: activation uint8 and weight int8 not supported on ARM");
#endif
}

template <>
void QGemm<uint8_t, uint8_t, int32_t>(
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
    concurrency::ThreadPool* thread_pool) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8
  MlasGemm(M, N, K, lhs_data, lda, lhs_offset, rhs_data, ldb, rhs_offset, result_data, ldc, thread_pool);
#else
  ORT_ENFORCE(lda == K && ldb == N && ldc == N, "For gemmlowp only RowMajor*RowMajor=RowMajor format is supported");

  GemmlowpMultiplyu8u8_s32(lhs_data, rhs_data, result_data, lhs_offset, rhs_offset, M, N, K, thread_pool);
#endif
}

template <typename LeftScalar, typename RightScalar>
void QGemm(
    int M,
    int N,
    int K,
    const LeftScalar* lhs_data,
    int lda,
    const LeftScalar lhs_offset,
    const RightScalar* rhs_data,
    int ldb,
    const RightScalar rhs_offset,
    float* result_data,
    int ldc,
    const float* result_scale,
    const float* bias,
    concurrency::ThreadPool* thread_pool) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8
  MlasGemm(M, N, K, lhs_data, lda, lhs_offset, rhs_data, ldb, rhs_offset, result_data, ldc, result_scale, bias, thread_pool);
#else
  QGemm(M, N, K, lhs_data, lda, lhs_offset, rhs_data, ldb, rhs_offset, reinterpret_cast<int32_t*>(result_data), ldc, thread_pool);
  for (int m = 0; m < M; m++) {
    if (bias != nullptr) {
      for (int n = 0; n < N; n++) {
        result_data[n] = static_cast<float>(reinterpret_cast<int32_t*>(result_data)[n]) * result_scale[0] + bias[n];
      }
    } else {
      for (int n = 0; n < N; n++) {
        result_data[n] = static_cast<float>(reinterpret_cast<int32_t*>(result_data)[n]) * result_scale[0];
      }
    }
    result_data += ldc;
  }
#endif
}

template
void QGemm<uint8_t, int8_t>(
    int M,
    int N,
    int K,
    const uint8_t* lhs_data,
    int lda,
    const uint8_t lhs_offset,
    const int8_t* rhs_data,
    int ldb,
    const int8_t rhs_offset,
    float* result_data,
    int ldc,
    const float* result_scale,
    const float* bias,
    concurrency::ThreadPool* thread_pool);

template
void QGemm<uint8_t, uint8_t>(
    int M,
    int N,
    int K,
    const uint8_t* lhs_data,
    int lda,
    const uint8_t lhs_offset,
    const uint8_t* rhs_data,
    int ldb,
    const uint8_t rhs_offset,
    float* result_data,
    int ldc,
    const float* result_scale,
    const float* bias,
    concurrency::ThreadPool* thread_pool);

}  // namespace onnxruntime
