// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <memory>

#include "contrib_ops/rocm/bert/elementwise.h"
#include "contrib_ops/rocm/bert/gemm_fast_gelu_ck.cuh"
#include "contrib_ops/rocm/bert/gemm_fast_gelu_common.h"
#include "core/providers/rocm/tunable/gemm.h"
#include "core/providers/rocm/tunable/gemm_hipblaslt.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace blas {
namespace internal {

using namespace onnxruntime::rocm::tunable::blas::internal;

template <typename T>
Status GemmFastGeluUnfused(const GemmFastGeluParams<T>* params) {
  namespace column_major = onnxruntime::rocm::tunable::blas::column_major;
  ORT_RETURN_IF_ERROR(column_major::Gemm(params->tuning_ctx, params->stream, params->handle,
                                         params->opb, params->opa,
                                         params->n, params->m, params->k,
                                         params->alpha, params->b, params->ldb, params->a, params->lda,
                                         params->beta, params->c, params->ldc));

  int64_t fast_gelu_input_length = params->m * params->n;
  int64_t bias_length = (params->bias != nullptr) ? params->n : 0;

  // Because of GemmFastGeluUnfused is a combination of GemmOp and FastGeluOp, FastGeluOp in this combination is
  // an inplace computation.
  // 1. If we call GemmFastGeluUnfused directly with enabled tuning, it may cause the input buffer of FastGelu been
  // updated accumulatedly and result in incorrect result finally. This only happens if the tuning's FindFastest is invoked.
  // 2. It's safe to call GemmFastGeluUnfused with disabled tuning, FastGelu only run once and produce correct result.
  // 3. It's safe to call GemmFastGeluUnfused as part of GemmFastGeluTunableOp with enable tuning, GemmTunableOp and
  // FastGeluTunableOp will do tune in first warmup step separately during GemmFastGeluUnfused profiling process.
  // After that, the call to GemmFastGeluUnfused not invoke tuning's FindFastest of FastGelu.
  //
  // Note: If any change cause directly usage of GemmFastGeluUnfused, add PreTuning() and PostTuning() in FastGeluTunableOp
  // to protect original input value.
  return onnxruntime::contrib::rocm::LaunchElementwiseKernel<functor::FastGeLU, T>(
      params->tuning_ctx, params->stream,
      params->c, static_cast<int>(fast_gelu_input_length),
      params->bias, static_cast<int>(bias_length),
      params->c);
}

template <typename T, typename ALayout, typename BLayout>
class GemmFastGeluTunableOp : public TunableOp<GemmFastGeluParams<T>> {
 public:
  GemmFastGeluTunableOp() {
    this->RegisterOp(GemmFastGeluUnfused<T>);
#ifdef USE_COMPOSABLE_KERNEL
    for (auto&& [_, op] : GetCKGemmAddFastGeluTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
    for (auto&& [_, op] : GetCKGemmFastGeluTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif

#ifdef USE_HIPBLASLT
    for (auto&& [_, op] : GetHipBlasLtGemmFastGeluTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif
  }
};

}  // namespace internal
}  // namespace blas
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
