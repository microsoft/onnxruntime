// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <string>

#include "contrib_ops/rocm/bert/fast_gelu_impl.h"
#include "core/providers/rocm/tunable/gemm.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "core/providers/rocm/tunable/gemm_fast_gelu_ck.cuh"
#include "core/providers/rocm/tunable/gemm_fast_gelu_common.h"

using onnxruntime::rocm::tunable::blas::BlasOp;
using onnxruntime::rocm::tunable::blas::BlasOpToString;

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

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
  return onnxruntime::contrib::rocm::LaunchFastGeluKernel<T>(params->stream,
                                 static_cast<int>(fast_gelu_input_length),
                                 static_cast<int>(bias_length),
                                 params->c,
                                 params->bias,
                                 params->c,
                                 params->tuning);
}

template <typename T, typename ALayout, typename BLayout>
class GemmFastGeluTunableOp : public onnxruntime::rocm::tunable::TunableOp<GemmFastGeluParams<T>> {
 public:
  GemmFastGeluTunableOp() {
    this->ops_.emplace_back(GemmFastGeluUnfused<T>);
    for (auto&& [_, op] : GetCKGemmAddFastGeluTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->ops_.emplace_back(std::move(op));
    }
    for (auto&& [_, op] : GetCKGemmFastGeluTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->ops_.emplace_back(std::move(op));
    }

    this->SetDefaultId(0);
  }
};

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
