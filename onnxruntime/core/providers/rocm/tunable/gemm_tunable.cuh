// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>

#include "core/providers/rocm/tunable/gemm_ck.cuh"
#include "core/providers/rocm/tunable/gemm_rocblas.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

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
    this->RegisterOp(RocBlasGemmOp<T>);

#ifdef USE_ROCBLAS_EXTENSION_API
    this->RegisterNestedTunableOp(&rocblas_gemm_tunable_op_);
#endif /* #ifdef USE_ROCBLAS_EXTENSION_API */

    for (auto&& [_, op] : GetCKGemmTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
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

 private:
#ifdef USE_ROCBLAS_EXTENSION_API
  RocBlasGemmTunableOp<T> rocblas_gemm_tunable_op_;
#endif
};

template <typename T, typename ALayout, typename BLayout>
class BatchedGemmTunableOp : public tunable::TunableOp<BatchedGemmParams<T>> {
 public:
  BatchedGemmTunableOp() {
    this->ops_.emplace_back(RocBlasBatchedGemmOp<T>);
  }

  const BatchedGemmParams<T>* PreTuning(const BatchedGemmParams<T>* params) override {
    if (!IsZero(params->beta)) {
      // See GemmTunableOp<T>::PreTuning for more details
      BatchedGemmParams<T>* proxy = new BatchedGemmParams<T>();
      *proxy = *params;
      using PointerOfT = T*;
      proxy->cs = new PointerOfT[params->batch];
      for (int i = 0; i < params->batch; i++) {
        HIP_CALL_THROW(hipMalloc(&(proxy->cs[i]), proxy->m * proxy->ldc * sizeof(T)));
      }
      return proxy;
    }

    return params;
  }

  void PostTuning(const BatchedGemmParams<T>* params) override {
    if (!IsZero(params->beta)) {
      for (int i = 0; i < params->batch; i++) {
        HIP_CALL_THROW(hipFree(params->cs[i]));
      }
      delete[] params->cs;
      delete params;
    }
  }
};

template <typename T, typename ALayout, typename BLayout>
class StridedBatchedGemmTunableOp : public tunable::TunableOp<StridedBatchedGemmParams<T>> {
 public:
  StridedBatchedGemmTunableOp() {
    this->ops_.emplace_back(RocBlasStridedBatchedGemmOp<T>);
    for (auto&& [_, op] : GetCKStridedBatchedGemmTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->ops_.emplace_back(std::move(op));
    }
  }

  const StridedBatchedGemmParams<T>* PreTuning(const StridedBatchedGemmParams<T>* params) override {
    if (!IsZero(params->beta)) {
      // See GemmTunableOp<T>::PreTuning for more details
      StridedBatchedGemmParams<T>* proxy = new StridedBatchedGemmParams<T>();
      *proxy = *params;
      HIP_CALL_THROW(hipMalloc(&(proxy->c), proxy->m * proxy->stride_c * sizeof(T)));
      return proxy;
    }

    return params;
  }

  void PostTuning(const StridedBatchedGemmParams<T>* params) override {
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
