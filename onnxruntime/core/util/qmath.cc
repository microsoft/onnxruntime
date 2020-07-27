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

template <typename TA, typename TB, typename TY>
void QGemmWithEigen(const TA* A_data, const TB* B_data, TY* Y_data, int M, int N, int K, TA a_offset, TB b_offset) {
  auto A = ConstEigenMatrixMapRowMajor<TA>(A_data, M, K);
  auto B = ConstEigenMatrixMapRowMajor<TB>(B_data, K, N);

  auto A_row_sum = (A.template cast<TY>().rowwise().sum()) * static_cast<TY>(b_offset);
  auto B_col_sum = (B.template cast<TY>().colwise().sum()) * static_cast<TY>(a_offset);
  EigenMatrixMapRowMajor<TY>(Y_data, M, N) = A.template cast<TY>() * B.template cast<TY>() + static_cast<TY>(K * a_offset * b_offset) * ConstEigenMatrixMapRowMajor<TY>::Ones(M, N);
  EigenMatrixMapRowMajor<TY>(Y_data, M, N).colwise() -= A_row_sum;
  EigenMatrixMapRowMajor<TY>(Y_data, M, N).rowwise() -= B_col_sum;
}

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
    concurrency::ThreadPool* thread_pool) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8
  MlasGemm(M, N, K, lhs_data, lda, lhs_offset, rhs_data, ldb, rhs_offset, rhs_signed, result_data, ldc, thread_pool);
#else
  ORT_ENFORCE(lda == K && ldb == N && ldc == N, "Only RowMajor*RowMajor=RowMajor format is supported");

  if (rhs_signed) {
    QGemmWithEigen<uint8_t, int8_t, int32_t>(lhs_data, reinterpret_cast<const int8_t*>(rhs_data), result_data, M, N, K, lhs_offset, static_cast<int8_t>(rhs_offset));
  } else {
    GemmlowpMultiplyu8u8_s32(lhs_data, rhs_data, result_data, lhs_offset, rhs_offset, M, N, K, thread_pool);
  }
#endif
}

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
    concurrency::ThreadPool* thread_pool) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8
  MlasGemm(M, N, K, lhs_data, lda, lhs_offset, rhs_data, ldb, rhs_offset, rhs_signed, result_data, ldc, result_scale, bias, thread_pool);
#else
  QGemm(M, N, K, lhs_data, lda, lhs_offset, rhs_data, ldb, rhs_offset, rhs_signed, reinterpret_cast<int32_t*>(result_data), ldc, thread_pool);
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

}  // namespace onnxruntime
