// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/framework/stream_handles.h"

namespace onnxruntime {
namespace cuda {

class CudaScratchBufferAllocator {
 public:
  explicit CudaScratchBufferAllocator(const CudaKernel& kernel, Stream* stream) : kernel_{kernel}, stream_{stream} {
  }

  template <typename T>
  IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes) const {
    return kernel_.GetScratchBuffer<T>(count_or_bytes, stream_);
  }

 private:
  const CudaKernel& kernel_;
  Stream* stream_;
};

// unit for handling indexing and counting of gathered indices
using GatheredIndexIndex_t = int32_t;

template <typename T, typename TIndex>
void GatherGradImpl(
    cudaStream_t stream,
    const cudaDeviceProp& prop,
    const CudaScratchBufferAllocator& allocator,
    const T* dY_data,
    const TIndex* dX_indices,
    const GatheredIndexIndex_t num_gathered_indices,
    const int64_t gather_dimension_size,
    const int64_t num_gathered_per_index,
    const int64_t num_batches,
    T* dX_data);

}  // namespace cuda
}  // namespace onnxruntime
