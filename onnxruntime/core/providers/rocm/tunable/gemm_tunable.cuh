// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/tunable/gemm_ck.cuh"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/gemm_hipblaslt.h"
#include "core/providers/rocm/tunable/gemm_rocblas.h"
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
class GemmTunableOp : public TunableOp<GemmParams<T>> {
 public:
  GemmTunableOp() {
    this->RegisterOp(RocBlasGemmOp<T>);

#ifdef USE_HIPBLASLT
    for (auto&& [_, op] : GetHipBlasLtGemmTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif

#ifdef USE_ROCBLAS_EXTENSION_API
    for (auto&& [_, op] : GetRocBlasGemmTypeStringAndOps<T>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif

#ifdef USE_COMPOSABLE_KERNEL
    for (auto&& [_, op] : GetCKGemmTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }

    for (auto&& [_, op] : GetCKStreamKGemmTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
    for (auto&& [_, op] : GetCKSplitKGemmTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif
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

template <typename T, typename ALayout, typename BLayout>
class BatchedGemmTunableOp : public TunableOp<BatchedGemmParams<T>> {
 public:
  BatchedGemmTunableOp() {
    this->RegisterOp(RocBlasBatchedGemmOp<T>);

#ifdef USE_ROCBLAS_EXTENSION_API
    for (auto&& [_, op] : GetRocBlasBatchedGemmTypeStringAndOps<T>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif
  }

  const BatchedGemmParams<T>* PreTuning(const BatchedGemmParams<T>* params) override {
    if (!IsZero(params->beta)) {
      // See GemmTunableOp<T>::PreTuning for more details
      BatchedGemmParams<T>* proxy = new BatchedGemmParams<T>();
      *proxy = *params;

      // malloc a large buffer and then slice it
      const int single_buffer_bytes = CeilDiv(proxy->m * proxy->ldc * sizeof(T), 128) * 128;
      T* buffer;
      HIP_CALL_THROW(hipMalloc(&buffer, proxy->batch * single_buffer_bytes));
      std::vector<T*> buffer_ptrs(proxy->batch, nullptr);
      for (int i = 0; i < proxy->batch; i++) {
        // note the following is offseted by bytes
        buffer_ptrs[i] = reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) + i * single_buffer_bytes);
      }

      // copy all ptrs to device
      HIP_CALL_THROW(hipMalloc(&(proxy->cs), proxy->batch * sizeof(T*)));
      HIP_CALL_THROW(hipMemcpy(proxy->cs, buffer_ptrs.data(), buffer_ptrs.size() * sizeof(T*), hipMemcpyHostToDevice));
      return proxy;
    }

    return params;
  }

  void PostTuning(const BatchedGemmParams<T>* params) override {
    if (!IsZero(params->beta)) {
      T* buffer;
      HIP_CALL_THROW(hipMemcpy(&buffer, params->cs, sizeof(T*), hipMemcpyDeviceToHost));
      HIP_CALL_THROW(hipFree(buffer));
      HIP_CALL_THROW(hipFree(params->cs));
      delete params;
    }
  }
};

template <typename T, typename ALayout, typename BLayout>
class StridedBatchedGemmTunableOp : public TunableOp<StridedBatchedGemmParams<T>> {
 public:
  StridedBatchedGemmTunableOp() {
    this->RegisterOp(RocBlasStridedBatchedGemmOp<T>);

#ifdef USE_HIPBLASLT
    for (auto&& [_, op] : GetHipBlasLtStridedBatchedGemmTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif

#ifdef USE_ROCBLAS_EXTENSION_API
    for (auto&& [_, op] : GetRocBlasStridedBatchedGemmTypeStringAndOps<T>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif

#ifdef USE_COMPOSABLE_KERNEL
    for (auto&& [_, op] : GetCKStridedBatchedGemmTypeStringAndOps<T, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif
  }

  const StridedBatchedGemmParams<T>* PreTuning(const StridedBatchedGemmParams<T>* params) override {
    if (!IsZero(params->beta)) {
      // See GemmTunableOp<T>::PreTuning for more details
      StridedBatchedGemmParams<T>* proxy = new StridedBatchedGemmParams<T>();
      *proxy = *params;
      HIP_CALL_THROW(hipMalloc(&(proxy->c), proxy->batch * proxy->stride_c * sizeof(T)));
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
