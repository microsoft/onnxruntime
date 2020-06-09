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

void QGemm(
    const MLAS_GEMM_U8X8_PARAMETERS& gemm_parameters,
    concurrency::ThreadPool* thread_pool) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8
  MlasGemm(&gemm_parameters, thread_pool);
#else
  if (gemm_parameters.BTypeIsSigned) {
    ORT_NOT_IMPLEMENTED("MatMulInteger: activation uint8 and weight int8 not supported");
  }

  const size_t M = gemm_parameters.M;
  const size_t N = gemm_parameters.N;
  const size_t K = gemm_parameters.K;

  ORT_ENFORCE(gemm_parameters.lda == K && gemm_parameters.ldb == N && gemm_parameters.ldc == N,
              "For gemmlowp only RowMajor*RowMajor=RowMajor format is supported");

  GemmlowpMultiplyu8u8_s32(gemm_parameters.A,
                           gemm_parameters.B,
                           gemm_parameters.C,
                           gemm_parameters.offa,
                           gemm_parameters.offb,
                           static_cast<int>(M),
                           static_cast<int>(N),
                           static_cast<int>(K),
                           thread_pool);
#endif
}

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
  MLAS_GEMM_U8X8_PARAMETERS gemm_parameters = {};
  gemm_parameters.M = M;
  gemm_parameters.N = N;
  gemm_parameters.K = K;
  gemm_parameters.A = lhs_data;
  gemm_parameters.lda = lda;
  gemm_parameters.B = (const uint8_t*)rhs_data;
  gemm_parameters.ldb = ldb;
  gemm_parameters.C = result_data;
  gemm_parameters.ldc = ldc;
  gemm_parameters.offa = uint8_t(lhs_offset);
  gemm_parameters.offb = uint8_t(rhs_offset);
  gemm_parameters.BTypeIsSigned = true;

  QGemm(gemm_parameters, thread_pool);
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
  MLAS_GEMM_U8X8_PARAMETERS gemm_parameters = {};
  gemm_parameters.M = M;
  gemm_parameters.N = N;
  gemm_parameters.K = K;
  gemm_parameters.A = lhs_data;
  gemm_parameters.lda = lda;
  gemm_parameters.B = (const uint8_t*)rhs_data;
  gemm_parameters.ldb = ldb;
  gemm_parameters.C = result_data;
  gemm_parameters.ldc = ldc;
  gemm_parameters.offa = uint8_t(lhs_offset);
  gemm_parameters.offb = uint8_t(rhs_offset);
  gemm_parameters.BTypeIsSigned = false;

  QGemm(gemm_parameters, thread_pool);
}

}  // namespace onnxruntime
