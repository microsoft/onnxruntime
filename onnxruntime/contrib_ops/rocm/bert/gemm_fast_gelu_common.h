// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

using onnxruntime::rocm::tunable::blas::BlasOp;
using onnxruntime::rocm::tunable::blas::BlasOpToString;

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace blas {

template <typename T>
struct GemmFastGeluParams : OpParams {
  std::string Signature() const override {
    bool has_bias = (nullptr != bias) ? 0 : 1;
    return MakeString(BlasOpToString(opa), BlasOpToString(opb), "_", m, "_", n, "_", k, '_', has_bias);
  }
  rocblas_handle handle;
  BlasOp opa;
  BlasOp opb;
  int64_t m;
  int64_t n;
  int64_t k;
  T alpha;
  const T* a;
  int64_t lda;
  const T* b;
  int64_t ldb;
  const T* bias;
  T beta;
  T* c;
  int64_t ldc;
};

}  // namespace blas
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
