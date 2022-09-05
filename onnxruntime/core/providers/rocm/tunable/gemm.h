// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/float16.h"
// #include "core/providers/rocm/rocm_pch.h"
#include "core/providers/rocm/tunable/gemm_common.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {

#define GEMM(T, ScalarT)                                                         \
  common::Status Gemm(                                                           \
      bool tunable, hipStream_t stream, rocblas_handle handle,                   \
      BlasOp opa, BlasOp opb,                                                    \
      std::int64_t m, std::int64_t n, std::int64_t k,                            \
      ScalarT alpha, const T* a, std::int64_t lda, const T* b, std::int64_t ldb, \
      ScalarT beta, T* c, std::int64_t ldc)

namespace row_major {

GEMM(double, double);
GEMM(float, float);
GEMM(half, half);
GEMM(BFloat16, BFloat16);
GEMM(double, float);
GEMM(half, float);
GEMM(BFloat16, float);

}  // namespace row_major

// TODO(anyone): the caller should not need to swap params a and b manaully, but all the current call site is doing so.
// It is cumbersome and unintuitive. We shall correct all callsite and remove this namespace in the furture.
namespace column_major {

GEMM(double, double);
GEMM(float, float);
GEMM(half, half);
GEMM(BFloat16, BFloat16);
GEMM(double, float);
GEMM(half, float);
GEMM(BFloat16, float);

}  // namespace column_major

}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime

#ifndef _GEMM_H_KEEP_SIGNATURE_DEFINES
#undef GEMM
#endif
