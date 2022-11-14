// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/gemm_fast_gelu_impl.h"

#include <hip/hip_fp16.h>

#include "contrib_ops/rocm/bert/gemm_fast_gelu_tunable_op.h"
#include "core/providers/rocm/tunable/gemm_common.h"

using onnxruntime::rocm::tunable::blas::BlasOp;

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
Status LaunchGemmFastGeluKernel(bool tuning,
                                hipStream_t stream,
                                rocblas_handle handle,
                                bool transa,
                                bool transb,
                                int64_t m,
                                int64_t n,
                                int64_t k,
                                const T alpha,
                                const T* a,
                                int64_t lda,
                                const T* b,
                                int64_t ldb,
                                const T* bias,
                                T* c,
                                int64_t ldc,
                                const T beta) {
  GemmFastGeluParams<T> params;
  params.tuning = tuning;
  params.stream = stream;
  params.handle = handle;
  params.opa = transa ? BlasOp::Trans : BlasOp::NonTrans;
  params.opb = transb ? BlasOp::Trans : BlasOp::NonTrans;

  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = alpha;
  params.a = a;
  params.lda = lda;
  params.b = b;
  params.ldb = ldb;
  params.beta = beta;
  params.bias = bias;
  params.c = c;
  params.ldc = ldc;
  params.has_bias = (bias != nullptr) ? true : false;

  static GemmFastGeluTunableOp<T> op;

  if (tuning) {
    op.EnableTuning();
  }

  return op(&params);
}

#define SPECIALIZED_IMPL(T)                                                                     \
  template Status LaunchGemmFastGeluKernel<T>(bool tuning,                                      \
                                              hipStream_t stream, rocblas_handle handle,        \
                                              bool transa, bool transb,                         \
                                              int64_t m, int64_t n, int64_t k, const T alpha,   \
                                              const T* a, int64_t lda, const T* b, int64_t ldb, \
                                              const T* bias, T* c, int64_t ldc, const T beta);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(BFloat16)

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
