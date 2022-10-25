// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>

#include "core/providers/rocm/tunable/gemm_ck.cuh"
#include "core/providers/rocm/tunable/gemm_rocblas.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/tunable.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

template <typename T>
bool IsZero(T v) {
  return v == 0.0f;
}

template <>
bool IsZero(BFloat16 v) {
  return v.val == 0;
}

template <>
bool IsZero(half v) {
  return __half2float(v) == 0.0f;
}


template <typename T, typename ALayout, typename BLayout>
class GemmTunableOp : public tunable::TunableOp<GemmParams<T>> {
 public:
  GemmTunableOp() {
    this->ops_.emplace_back(RocBlasGemmOp<T>);
    for (auto&& [_, op] : GetCKGemmTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->ops_.emplace_back(std::move(op));
    }
  }

  const GemmParams<T>* PreTuning(const GemmParams<T>* params) override {
    if (!IsZero(params->beta)) {
      // When beta != 0, C buffer is used as an input as well as an output. We need to create a proxy params for the
      // tuning process. Otherwise, tuning will cause the C buffer been updated accumulatedly, say, we tune it for n
      // iterations, then during tuning C^(1) = alpha A B + beta C^(0), ..., C^(n) = alpha A B + beta C^(n-1). And for
      // the actual run after tuning, the result will be C^(n+1), whereas what we want is C^(1). This only happens if
      // the tuning's FindFastest is invoked.
      //
      // Note, C^(i) is the C at i-th iteration.
      GemmParams<T>* proxy = new GemmParams<T>();
      *proxy = *params;
      HIP_CALL_THROW(hipMalloc(&(proxy->c), proxy->m * proxy->ldc * sizeof(T)));
      return proxy;
    }

    return params;
  }

  void PostTuning(const GemmParams<T>* params) override {
    if (!IsZero(params->beta)) {
      HIP_CALL_THROW(hipFree(params->c));
      delete params;
    }
  }
};

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
