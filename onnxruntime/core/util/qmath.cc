// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/util/qmath.h"
#include "core/common/common.h"
#include "core/util/math_cpuonly.h"

// fallback to gemmlowp when building for arm devices
#ifndef MLAS_SUPPORTS_GEMM_U8X8
#include "core/util/gemmlowp_common.h"
#endif

namespace onnxruntime {

template <typename TA, typename TB, typename TY>
void QGemmWithEigen(
    const TA* A_data,
    const TB* B_data,
    TY* Y_data,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    TA a_offset,
    TB b_offset) {
  auto A = ConstEigenMatrixMapRowMajorOuterStride<TA>(A_data, M, K, Eigen::OuterStride<>(lda));
  auto B = ConstEigenMatrixMapRowMajorOuterStride<TB>(B_data, K, N, Eigen::OuterStride<>(ldb));

  auto A_row_sum = (A.template cast<TY>().rowwise().sum()) * static_cast<TY>(b_offset);
  auto B_col_sum = (B.template cast<TY>().colwise().sum()) * static_cast<TY>(a_offset);
  EigenMatrixMapRowMajorOuterStride<TY>(Y_data, M, N, Eigen::OuterStride<>(ldc)) =
      A.template cast<TY>() * B.template cast<TY>() +
      static_cast<TY>(K * a_offset * b_offset) * ConstEigenMatrixMapRowMajor<TY>::Ones(M, N);

  EigenMatrixMapRowMajorOuterStride<TY>(Y_data, M, N, Eigen::OuterStride<>(ldc)).colwise() -= A_row_sum;
  EigenMatrixMapRowMajorOuterStride<TY>(Y_data, M, N, Eigen::OuterStride<>(ldc)).rowwise() -= B_col_sum;
}

void QGemm(
    int M,
    int N,
    int K,
    const uint8_t* A,
    int lda,
    const uint8_t a_offset,
    const uint8_t* B,
    int ldb,
    const uint8_t b_offset,
    bool b_signed,
    int32_t* C,
    int ldc,
    concurrency::ThreadPool* thread_pool,
    const MLAS_QGEMM_OUTPUT_PROCESSOR* output_processor) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8
  MlasGemm(M, N, K, A, lda, a_offset, B, ldb, b_offset, b_signed, C, ldc, thread_pool, output_processor);
#else
  if (b_signed) {
    QGemmWithEigen<uint8_t, int8_t, int32_t>(A, reinterpret_cast<const int8_t*>(B), C,
                                             M, N, K, lda, ldb, ldc,
                                             a_offset, static_cast<int8_t>(b_offset));
  } else {
    GemmlowpMultiplyu8u8_s32(A, B, C,
                             a_offset, b_offset,
                             M, N, K, lda, ldb, ldc, thread_pool);
  }

  if (output_processor) {
    output_processor->Process(C, 0, 0, M, N, ldc);
  }
#endif
}

}  // namespace onnxruntime
