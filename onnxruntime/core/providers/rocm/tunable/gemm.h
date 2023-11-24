// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/rocm/tunable/gemm_common.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {

#define GEMM(T, ScalarT)                                                         \
  common::Status Gemm(                                                           \
      RocmTuningContext* tuning_ctx, Stream* stream, rocblas_handle handle,      \
      BlasOp opa, BlasOp opb,                                                    \
      std::int64_t m, std::int64_t n, std::int64_t k,                            \
      ScalarT alpha, const T* a, std::int64_t lda, const T* b, std::int64_t ldb, \
      ScalarT beta, T* c, std::int64_t ldc)

#define BATCHED_GEMM(T, ScalarT)                                            \
  common::Status BatchedGemm(                                               \
      RocmTuningContext* tuning_ctx, Stream* stream, rocblas_handle handle, \
      BlasOp opa, BlasOp opb,                                               \
      std::int64_t m, std::int64_t n, std::int64_t k,                       \
      ScalarT alpha,                                                        \
      const T** as, std::int64_t lda,                                       \
      const T** bs, std::int64_t ldb,                                       \
      ScalarT beta,                                                         \
      T** cs, std::int64_t ldc, std::int64_t batch)

#define STRIDED_BATCHED_GEMM(T, ScalarT)                                    \
  common::Status StridedBatchedGemm(                                        \
      RocmTuningContext* tuning_ctx, Stream* stream, rocblas_handle handle, \
      BlasOp opa, BlasOp opb,                                               \
      std::int64_t m, std::int64_t n, std::int64_t k,                       \
      ScalarT alpha,                                                        \
      const T* a, std::int64_t lda, std::int64_t stride_a,                  \
      const T* b, std::int64_t ldb, std::int64_t stride_b,                  \
      ScalarT beta,                                                         \
      T* c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch)

namespace row_major {

GEMM(double, double);
GEMM(float, float);
GEMM(half, half);
GEMM(BFloat16, BFloat16);
GEMM(double, float);
GEMM(half, float);
GEMM(BFloat16, float);

BATCHED_GEMM(double, double);
BATCHED_GEMM(float, float);
BATCHED_GEMM(half, half);
BATCHED_GEMM(BFloat16, BFloat16);
BATCHED_GEMM(double, float);
BATCHED_GEMM(half, float);
BATCHED_GEMM(BFloat16, float);

STRIDED_BATCHED_GEMM(double, double);
STRIDED_BATCHED_GEMM(float, float);
STRIDED_BATCHED_GEMM(half, half);
STRIDED_BATCHED_GEMM(BFloat16, BFloat16);
STRIDED_BATCHED_GEMM(double, float);
STRIDED_BATCHED_GEMM(half, float);
STRIDED_BATCHED_GEMM(BFloat16, float);

}  // namespace row_major

// TODO(anyone): the caller should not need to swap the params a and b manually, but all the current callsites are
// doing so. It is cumbersome and unintuitive. At the moment, this namespace only ease the porting from old direct
// rocblas_gemm* calls to tunable gemm calls. After all porting of all callsites, if there is no column_major usecase
// left, then we shall remove this namespace, finally.
namespace column_major {

GEMM(double, double);
GEMM(float, float);
GEMM(half, half);
GEMM(BFloat16, BFloat16);
GEMM(double, float);
GEMM(half, float);
GEMM(BFloat16, float);

BATCHED_GEMM(double, double);
BATCHED_GEMM(float, float);
BATCHED_GEMM(half, half);
BATCHED_GEMM(BFloat16, BFloat16);
BATCHED_GEMM(double, float);
BATCHED_GEMM(half, float);
BATCHED_GEMM(BFloat16, float);

STRIDED_BATCHED_GEMM(double, double);
STRIDED_BATCHED_GEMM(float, float);
STRIDED_BATCHED_GEMM(half, half);
STRIDED_BATCHED_GEMM(BFloat16, BFloat16);
STRIDED_BATCHED_GEMM(double, float);
STRIDED_BATCHED_GEMM(half, float);
STRIDED_BATCHED_GEMM(BFloat16, float);

}  // namespace column_major

}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime

#ifndef _GEMM_H_KEEP_SIGNATURE_DEFINES
#undef GEMM
#endif
