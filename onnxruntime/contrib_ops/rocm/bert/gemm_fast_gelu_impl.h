// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/rocm/bert/gemm_fast_gelu_common.h"
#include "core/common/status.h"
#include "core/framework/float16.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace blas {

#define GEMMFASTGELU(T, ScalarT)                                                 \
  common::Status GemmFastGelu(                                                   \
      RocmTuningContext* tuning_ctx, Stream* stream, rocblas_handle handle,      \
      BlasOp opa, BlasOp opb,                                                    \
      std::int64_t m, std::int64_t n, std::int64_t k,                            \
      ScalarT alpha, const T* a, std::int64_t lda, const T* b, std::int64_t ldb, \
      const T* bias, ScalarT beta, T* c, std::int64_t ldc)

namespace row_major {

GEMMFASTGELU(float, float);
GEMMFASTGELU(half, half);
GEMMFASTGELU(BFloat16, BFloat16);
GEMMFASTGELU(half, float);
GEMMFASTGELU(BFloat16, float);

}  // namespace row_major

}  // namespace blas
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime

#ifndef _GEMM_FASTGELU_H_KEEP_SIGNATURE_DEFINES
#undef GEMMFASTGELU
#endif
