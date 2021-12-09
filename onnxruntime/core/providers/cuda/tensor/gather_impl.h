// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

void GatherImpl(
    cudaStream_t stream,
    const int64_t input_block_size,
    const int64_t indices_max,
    const fast_divmod& output_block_size,
    const fast_divmod& block_size,
    const void* indices_data,
    size_t index_element_size,
    const void* input_data,
    size_t element_size,
    void* output_data,
    const size_t N);

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

// unit for handling indexing and counting of gathered indices
using GatheredIndexIndex_t = int32_t;
using SegmentIndex_t = GatheredIndexIndex_t;

// GatherGradPrepareGetNumSegments and GatherGradPrepare precomputes
// information needed in GatherGrad thereby removing the need
// to compute this information within GatherGrad and moving all the
// memcpys to GatherGradPrepareGetNumSegments and GatherGradPrepare.
template <typename TIndex>
void GatherGradPrepareGetNumSegments(cudaStream_t stream,
    const CudaScratchBufferAllocator& allocator,
    const TIndex* dX_indices,
    const GatheredIndexIndex_t num_gathered_indices,
    int64_t gather_dimension_size,
    int64_t num_gathered_per_index,
    SegmentIndex_t& host_num_segments,
    IAllocatorUniquePtr<SegmentIndex_t>& segment_counts_out,
    TIndex& dX_indices_sorted,
    TIndex& dY_indices_sorted);

template <typename TIndex>
void GatherGradPrepare(
    cudaStream_t stream,
    const CudaScratchBufferAllocator& allocator,
    const TIndex* dX_indices,
    const GatheredIndexIndex_t num_gathered_indices,
    int64_t gather_dimension_size,
    int64_t num_gathered_per_index,
    SegmentIndex_t& host_num_segments,
    IAllocatorUniquePtr<SegmentIndex_t>& segment_counts,
    SegmentIndex_t& segment_offsets,
    SegmentIndex_t& last_segment_partial_segment_count,
    SegmentIndex_t& last_segment_partial_segment_offset,
    SegmentIndex_t& per_segment_partial_segment_counts,
    SegmentIndex_t& per_segment_partial_segment_offsets,
    TIndex& dX_indices_sorted,
    TIndex& dY_indices_sorted);

}  // namespace cuda
}  // namespace onnxruntime
