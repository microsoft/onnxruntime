// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/framework/float8.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {

enum class BlasOp {
  N = 0,
  T = 1,
  NonTrans = 0,
  Trans = 1,
};

inline std::string BlasOpToString(BlasOp op) {
  switch (op) {
    case BlasOp::N:
      return "N";
    case BlasOp::T:
      return "T";
    // following is unreachable, compiler is producing false-positive warning, unfortunately.
    default:
      ORT_THROW("unreachable");
  }
}

// We don't assume the implementation is row-majored or column-majored. But for testing convenience, we assume all
// our wrappers have row-majored convention, since it is the native layout to numpy and pytorch.
template <typename T>
struct GemmParams : tunable::OpParams {
  std::string Signature() const override {
    return MakeString(BlasOpToString(opa), BlasOpToString(opb), "_", m, "_", n, "_", k);
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
  T beta;
  T* c;
  int64_t ldc;
};

template <typename T>
struct BatchedGemmParams : tunable::OpParams {
  std::string Signature() const override {
    return MakeString(BlasOpToString(opa), BlasOpToString(opb), "_", m, "_", n, "_", k, "_B", batch);
  }

  rocblas_handle handle;
  BlasOp opa;
  BlasOp opb;
  int64_t m;
  int64_t n;
  int64_t k;
  T alpha;
  const T** as;
  int64_t lda;
  const T** bs;
  int64_t ldb;
  T beta;
  T** cs;
  int64_t ldc;
  int64_t batch;
};

template <typename T>
struct StridedBatchedGemmParams : tunable::OpParams {
  std::string Signature() const override {
    return MakeString(BlasOpToString(opa), BlasOpToString(opb), "_", m, "_", n, "_", k, "_B", batch);
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
  int64_t stride_a;
  const T* b;
  int64_t ldb;
  int64_t stride_b;
  T beta;
  T* c;
  int64_t ldc;
  int64_t stride_c;
  int64_t batch;
};

}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
