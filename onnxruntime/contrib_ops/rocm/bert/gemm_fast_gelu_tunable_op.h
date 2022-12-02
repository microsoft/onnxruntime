// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <string>

#include "contrib_ops/rocm/bert/fast_gelu_impl.h"
#include "core/providers/rocm/tunable/gemm.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

using onnxruntime::rocm::tunable::blas::BlasOp;
using onnxruntime::rocm::tunable::blas::BlasOpToString;

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct GemmFastGeluParams : onnxruntime::rocm::tunable::OpParams {
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
  const T* bias;
  T beta;
  T* c;
  int64_t ldc;
  bool tuning{false};
};

template <typename T>
Status GemmFastGeluUnfused(const GemmFastGeluParams<T>* params) {
  namespace column_major = onnxruntime::rocm::tunable::blas::column_major;
  if (column_major::Gemm(params->tuning, params->stream, params->handle,
                         params->opb, params->opa,
                         params->n, params->m, params->k,
                         params->alpha, params->b, params->ldb, params->a, params->lda,
                         params->beta, params->c, params->ldc) != Status::OK()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "GemmFastGelu call column_major::Gemm failed");
  }

  int64_t fast_gelu_input_length = params->m * params->n;
  int64_t bias_length = (params->bias != nullptr) ? params->n : 0;

  // inplace computation
  return LaunchFastGeluKernel<T>(params->tuning,
                                 params->stream,
                                 static_cast<int>(fast_gelu_input_length),
                                 static_cast<int>(bias_length),
                                 params->c,
                                 params->bias,
                                 params->c);
}

template <typename T>
class GemmFastGeluTunableOp : public onnxruntime::rocm::tunable::TunableOp<GemmFastGeluParams<T>> {
 public:
  GemmFastGeluTunableOp() {
    this->ops_.emplace_back(GemmFastGeluUnfused<T>);

    this->SetDefaultId(0);
  }
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
