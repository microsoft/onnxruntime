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
  if (rhs_signed) {
    QGemmWithEigen<uint8_t, int8_t, int32_t>(lhs_data, reinterpret_cast<const int8_t*>(rhs_data), result_data,
                                             M, N, K, lda, ldb, ldc,
                                             lhs_offset, static_cast<int8_t>(rhs_offset));
  } else {
    GemmlowpMultiplyu8u8_s32(lhs_data, rhs_data, result_data,
                             lhs_offset, rhs_offset,
                             M, N, K, lda, ldb, ldc, thread_pool);
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
  MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR scale_bias_processor(result_data, ldc, result_scale, bias);
  MlasGemm(M, N, K,
           lhs_data, lda, lhs_offset,
           rhs_data, ldb, rhs_offset, rhs_signed,
           reinterpret_cast<int32_t*>(result_data), ldc,
           thread_pool,
           &scale_bias_processor);
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

void GetQuantizationParameter(const float* data, int64_t num_of_elements, float& scale, uint8_t& zp) {
  // find input range min and max
  float min, max;
  MlasFindMinMaxElement(data, &min, &max, num_of_elements);

  // ensure the input range includes zero
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);

  // find scale and zero point
  uint8_t qmin = 0;
  uint8_t qmax = 255;
  scale = max == min ? 1.0f : (max - min) / (qmax - qmin);

  float initial_zero_point = qmin - min / scale;
  zp = static_cast<uint8_t>(RoundHalfToEven(std::max(float(qmin), std::min(float(qmax), initial_zero_point))));
}

}  // namespace onnxruntime
