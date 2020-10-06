// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

class CudaScratchBufferAllocator {
 public:
  explicit CudaScratchBufferAllocator(const CudaKernel& kernel) : kernel_{kernel} {
  }

  template <typename T>
  IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes) const {
    return kernel_.GetScratchBuffer<T>(count_or_bytes);
  }

 private:
  const CudaKernel& kernel_;
};

enum GatherGradImplementation {
  ThreadPerIndex,
  FancyIterator,
};

template <typename T, typename Tin>
void GatherGradImpl(
    const CudaScratchBufferAllocator& allocator,
    const T* grad_data,
    const Tin* indices_data,
    const int64_t num_indices,
    const int64_t num_weights,
    const int64_t stride,
    T* output_data,
    const int64_t num_inputs,
    const int64_t param_itrs,
    GatherGradImplementation impl);

}  // namespace cuda
}  // namespace onnxruntime
