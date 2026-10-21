// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/quantization/quantized_decode_l2_prefetch.h"

namespace onnxruntime::contrib::cuda {

inline bool QuantizedDecodeL2PrefetchEnabled() {
  static const bool enabled =
      ParseEnvironmentVariableWithDefault<bool>("ORT_QUANTIZED_DECODE_L2_PREFETCH", false);
  if (!enabled) {
    return false;
  }

  int device;
  CUDA_CALL_THROW(cudaGetDevice(&device));
  thread_local int cached_device = -1;
  thread_local bool supported = false;
  if (cached_device != device) {
    int major, minor;
    CUDA_CALL_THROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CALL_THROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    supported = ShouldPrefetchQuantizedDecodeL2(enabled, major, minor);
    cached_device = device;
  }
  return supported;
}

__device__ __forceinline__ void PrefetchQuantizedDecodeL2Address(const void* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  asm volatile("prefetch.global.L2 [%0];" : : "l"(address) : "memory");
#else
  (void)address;
#endif
}

template <typename T>
__device__ __forceinline__ void PrefetchQuantizedDecodeL2(
    const T* base, int64_t current, int64_t step, int64_t extent, bool enabled) {
  if (enabled) {
    const int64_t offset = QuantizedDecodeL2PrefetchOffset(current, step, extent);
    if (offset >= 0) {
      PrefetchQuantizedDecodeL2Address(base + offset);
    }
  }
}

}  // namespace onnxruntime::contrib::cuda
