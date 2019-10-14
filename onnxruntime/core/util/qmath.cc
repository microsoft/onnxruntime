// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/qmath.h"
#include "core/common/common.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

#if defined(_M_AMD64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#define MLAS_SUPPORTS_GEMM_U8X8
#else
// default to gemmlowp when building for arm devices
#ifndef USE_GEMMLOWP
#define USE_GEMMLOWP
#endif
#endif

#ifdef USE_GEMMLOWP
#include "core/util/gemmlowp_common.h"
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
    concurrency::ThreadPool* thread_pool) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8

  MlasGemm(M, N, K, lhs_data, lda, lhs_offset, rhs_data, ldb, rhs_offset, result_data, ldc, thread_pool);

#else
  ORT_UNUSED_PARAMETER(thread_pool);

  ORT_ENFORCE(lhs_offset == 0 && rhs_offset == 0, "For Eigen, zero point must be zero");
  ORT_ENFORCE(lda == K && ldb == N && ldc == N, "For Eigen only RowMajor*RowMajor=RowMajor format is supported");

  EigenCastGEMM<uint8_t, int8_t, int32_t>(lhs_data, rhs_data, result_data, M, N, K);

#endif
}

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
    concurrency::ThreadPool* thread_pool) {
#ifdef USE_GEMMLOWP

  ORT_ENFORCE(lda == K && ldb == N && ldc == N, "For gemmlowp only RowMajor*RowMajor=RowMajor format is supported");

  GemmlowpMultiplyu8u8_s32(lhs_data, rhs_data, result_data, lhs_offset, rhs_offset, M, N, K, thread_pool);

#else
  MlasGemm(M, N, K, lhs_data, lda, lhs_offset, rhs_data, ldb, rhs_offset, result_data, ldc, thread_pool);

#endif
}
}  // namespace onnxruntime
