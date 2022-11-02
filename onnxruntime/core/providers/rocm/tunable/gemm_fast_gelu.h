// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/rocm/tunable/gemm_fast_gelu_common.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {

#define GEMMFASTGELU(T, ScalarT)                                                 \
  common::Status GemmFastGelu(                                                   \
      bool tunable, hipStream_t stream, rocblas_handle handle,                   \
      BlasOp opa, BlasOp opb,                                                    \
      std::int64_t m, std::int64_t n, std::int64_t k,                            \
      ScalarT alpha, const T* a, std::int64_t lda, const T* b, std::int64_t ldb, \
      const T* bias, ScalarT beta, T* c, std::int64_t ldc)

namespace row_major {

GEMMFASTGELU(double, double);
GEMMFASTGELU(float, float);
GEMMFASTGELU(half, half);
GEMMFASTGELU(BFloat16, BFloat16);
GEMMFASTGELU(double, float);
GEMMFASTGELU(half, float);
GEMMFASTGELU(BFloat16, float);

}  // namespace row_major

// TODO(anyone): the caller should not need to swap the params a and b manually, but all the current callsites are
// doing so. It is cumbersome and unintuitive. At the moment, this namespace only ease the porting from old direct
// rocblas_gemm* calls to tunable gemm calls. After all porting of all callsites, if there is no column_major usecase
// left, then we shall remove this namespace, finally.
namespace column_major {

GEMMFASTGELU(double, double);
GEMMFASTGELU(float, float);
GEMMFASTGELU(half, half);
GEMMFASTGELU(BFloat16, BFloat16);
GEMMFASTGELU(double, float);
GEMMFASTGELU(half, float);
GEMMFASTGELU(BFloat16, float);

}  // namespace column_major

}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime

#ifndef _GEMM_FASTGELU_H_KEEP_SIGNATURE_DEFINES
#undef GEMMFASTGELU
#endif
