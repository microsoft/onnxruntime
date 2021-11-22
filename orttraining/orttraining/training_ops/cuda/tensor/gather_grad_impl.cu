// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "orttraining/training_ops/cuda/tensor/gather_grad_impl.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {
namespace cuda {
namespace gather_grad_internal {

// Note:
// For these implementations, first we generate sorted lists of dX and dY
// indices, ordered by dX indices. Then, we can consider segments of the sorted
// lists.
//
// Each continuous run of indices with the same dX value in dX_indices_sorted
// forms a segment.
//
// For example, given:
//   dX_indices_sorted = [1, 1, 2, 2, 2, 3]
//   dY_indices_sorted = [1, 4, 0, 3, 5, 2]
// The segments will be:  '1 1'  '2 2 2'  '3'
//
// The segments are further divided into partial segments
// of kMaxPartialSegmentSize size for increased parallelism.
//
// In the above example,
// num_segments = 3 (Number of distinct segments)
// segment_offsets = [0, 2, 5] (Index in the sorted indices where a new segment begins)
// per_segment_partial_segment_counts = [1, 1, 1] (Number of partial segments for each segment)
// per_segment_partial_segment_offsets = [0, 1, 2] (Index of a partial segment where a new partial
//                                                  segment begins assuming the index is referring to
//                                                  an array comprising of elements where each element
//                                                  is a partial segment.)

// unit for handling indexing and counting of segments or partial segments
using SegmentIndex_t = GatheredIndexIndex_t;

constexpr GatheredIndexIndex_t kMaxPartialSegmentSize = 10;

template <typename TInputIterator, typename TOutputIterator>
__global__ void CopyKernel(TOutputIterator dst, TInputIterator src, int64_t length) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, length);
  dst[id] = src[id];
}

// get sorted dX and dY indices, ordered by dX indices
template <typename TIndex>
void GetSortedIndices(
    cudaStream_t stream,
    const CudaScratchBufferAllocator& allocator,
    const TIndex* dX_indices,
    GatheredIndexIndex_t num_gathered_indices,
    IAllocatorUniquePtr<TIndex>& dX_indices_sorted_out,
    IAllocatorUniquePtr<TIndex>& dY_indices_sorted_out) {
  auto dY_indices = allocator.GetScratchBuffer<TIndex>(num_gathered_indices);
  CopyKernel<<<CeilDiv(num_gathered_indices, GridDim::maxThreadsPerBlock),
               GridDim::maxThreadsPerBlock, 0, stream>>>(
      dY_indices.get(), cub::CountingInputIterator<TIndex>{0}, num_gathered_indices);

  auto dX_indices_sorted = allocator.GetScratchBuffer<TIndex>(num_gathered_indices);
  auto dY_indices_sorted = allocator.GetScratchBuffer<TIndex>(num_gathered_indices);

  size_t temp_storage_size_bytes = 0;
  CUDA_CALL_THROW(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_size_bytes,
      dX_indices, dX_indices_sorted.get(),
      dY_indices.get(), dY_indices_sorted.get(),
      num_gathered_indices, 0, sizeof(TIndex) * 8, stream));

  auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);
  CUDA_CALL_THROW(cub::DeviceRadixSort::SortPairs(
      temp_storage.get(), temp_storage_size_bytes,
      dX_indices, dX_indices_sorted.get(),
      dY_indices.get(), dY_indices_sorted.get(),
      num_gathered_indices, 0, sizeof(TIndex) * 8, stream));

  dX_indices_sorted_out = std::move(dX_indices_sorted);
  dY_indices_sorted_out = std::move(dY_indices_sorted);
}

template <typename T>
IAllocatorUniquePtr<T> GetOffsetsFromCounts(
    cudaStream_t stream,
    const CudaScratchBufferAllocator& allocator,
    const T* counts, int32_t num_counts) {
  auto offsets = allocator.GetScratchBuffer<T>(num_counts);
  size_t temp_storage_size_bytes = 0;
  CUDA_CALL_THROW(cub::DeviceScan::ExclusiveSum(
      nullptr, temp_storage_size_bytes,
      counts, offsets.get(), num_counts, stream));

  auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);
  CUDA_CALL_THROW(cub::DeviceScan::ExclusiveSum(
      temp_storage.get(), temp_storage_size_bytes,
      counts, offsets.get(), num_counts, stream));

  return offsets;
}

// partial sums implementation adapted from here:
// https://github.com/pytorch/pytorch/blob/b186831c08e0e4e447eedb8a5cfab582995d37f9/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu

__global__ void ComputePartialSegmentOffsetsKernel(
    GatheredIndexIndex_t* ret,
    const SegmentIndex_t* partials_per_segment,
    const SegmentIndex_t* partials_per_segment_offset,
    const GatheredIndexIndex_t* segment_offsets,
    SegmentIndex_t num_of_segments) {
  const auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < num_of_segments) {
    auto idx = partials_per_segment_offset[id];
    const auto num_partials = partials_per_segment[id];
    const auto segment_offset = segment_offsets[id];
    for (SegmentIndex_t i = 0; i < num_partials; ++i) {
      ret[idx++] = segment_offset + i * kMaxPartialSegmentSize;
    }
  }
}

template <typename T, typename TIndex>
__global__ void ComputePartialSegmentSumsKernel(
    const TIndex* dY_indices_sorted,
    const T* dY_data,
    GatheredIndexIndex_t num_gathered_indices,
    int64_t num_gathered_per_index,
    const GatheredIndexIndex_t* partial_segment_offsets,
    SegmentIndex_t num_partial_segments,
    AccumulationType_t<T>* partial_segment_sums,
    const int64_t num_gathered_per_index_warp_size_multiple) {
  const auto id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto partial_segment_id = id / num_gathered_per_index_warp_size_multiple;
  const auto gathered_element_id = id % num_gathered_per_index_warp_size_multiple;
  const auto batch_id = blockIdx.y;

  if (gathered_element_id >= num_gathered_per_index) {
    return;
  }
  if (partial_segment_id >= num_partial_segments) {
    return;
  }

  const auto idx_begin = partial_segment_offsets[partial_segment_id];
  const auto idx_end =
      (partial_segment_id == num_partial_segments - 1) ? num_gathered_indices : partial_segment_offsets[partial_segment_id + 1];

  AccumulationType_t<T> partial_segment_sum = 0;
  for (auto idx = idx_begin; idx < idx_end; ++idx) {
    const auto target_row = dY_indices_sorted[idx];
    partial_segment_sum += static_cast<AccumulationType_t<T>>(
        dY_data[batch_id * num_gathered_indices * num_gathered_per_index +
                target_row * num_gathered_per_index +
                gathered_element_id]);
  }

  partial_segment_sums[batch_id * num_partial_segments * num_gathered_per_index +
                       partial_segment_id * num_gathered_per_index +
                       gathered_element_id] =
      partial_segment_sum;
}

template <typename T, typename TIndex>
__global__ void ComputeSegmentSumsAndScatterKernel(
    const TIndex* dX_indices_sorted,
    T* dX_data,
    int64_t num_gathered_per_index,
    const GatheredIndexIndex_t* segment_offsets,
    SegmentIndex_t num_segments,
    const AccumulationType_t<T>* partial_segment_sums,
    const SegmentIndex_t* per_segment_partial_segment_offsets,
    SegmentIndex_t num_partial_segments,
    const int64_t num_gathered_per_index_warp_size_multiple,
    const int64_t gather_dimension_size) {
  const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto segment_id = gid / num_gathered_per_index_warp_size_multiple;
  const auto gathered_element_id = gid % num_gathered_per_index_warp_size_multiple;
  const auto batch_id = blockIdx.y;

  if (gathered_element_id >= num_gathered_per_index) {
    return;
  }
  if (segment_id >= num_segments) {
    return;
  }

  const auto idx_begin = per_segment_partial_segment_offsets[segment_id];
  const auto idx_end =
      (segment_id == num_segments - 1) ? num_partial_segments : per_segment_partial_segment_offsets[segment_id + 1];

  AccumulationType_t<T> segment_sum = 0;
  for (auto idx = idx_begin; idx < idx_end; ++idx) {
    segment_sum +=
        partial_segment_sums[batch_id * num_partial_segments * num_gathered_per_index +
                             idx * num_gathered_per_index +
                             gathered_element_id];
  }

  auto target_row = dX_indices_sorted[segment_offsets[segment_id]];
  // All index values are expected to be within bounds [-s, s-1] along axis of size s.
  if (target_row < 0) target_row += gather_dimension_size;
  dX_data[batch_id * gather_dimension_size * num_gathered_per_index +
          target_row * num_gathered_per_index +
          gathered_element_id] =
      segment_sum;
}

// get partial sums of gathered dY values first, then sum the partial sums into
// the corresponding dX value
template <typename T, typename TIndex>
void PartialSumsImpl(
    cudaStream_t stream,
    const CudaScratchBufferAllocator& allocator,
    const TIndex* dX_indices_sorted,
    const TIndex* dY_indices_sorted,
    const T* dY_data,
    T* dX_data,
    GatheredIndexIndex_t num_gathered_indices,
    int64_t num_gathered_per_index,
    int64_t gather_dimension_size,
    int64_t num_batches,
    SegmentIndex_t num_segments,
    const GatheredIndexIndex_t* segment_offsets,
    const int32_t last_segment_partial_segment_count_in,
    const int32_t last_segment_partial_segment_offset_in,
    const int32_t* per_segment_partial_segment_counts_in,
    const int32_t* per_segment_partial_segment_offsets_in) {

  SegmentIndex_t host_num_partial_segments = last_segment_partial_segment_offset_in + last_segment_partial_segment_count_in;

  // compute index offsets per partial segment
  auto partial_segment_offsets = allocator.GetScratchBuffer<GatheredIndexIndex_t>(host_num_partial_segments);
  {
    const auto blocks_per_grid = CeilDiv(num_segments, GridDim::maxThreadsPerBlock);
    ComputePartialSegmentOffsetsKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        partial_segment_offsets.get(),
        per_segment_partial_segment_counts_in,
        per_segment_partial_segment_offsets_in,
        segment_offsets,
        num_segments);
  }

  {
    const auto num_gathered_per_index_warp_size_multiple =
        CeilDiv(num_gathered_per_index, GPU_WARP_SIZE) * GPU_WARP_SIZE;
    const auto threads_per_block =
        std::min<int64_t>(num_gathered_per_index_warp_size_multiple, GridDim::maxThreadsPerBlock);

    // compute partial segment sums
    auto partial_segment_sums = allocator.GetScratchBuffer<AccumulationType_t<T>>(
        num_batches * host_num_partial_segments * num_gathered_per_index);
    {
      const dim3 blocks_per_grid(
          CeilDiv(host_num_partial_segments * num_gathered_per_index_warp_size_multiple, threads_per_block),
          num_batches);
      ComputePartialSegmentSumsKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
          dY_indices_sorted,
          dY_data,
          num_gathered_indices,
          num_gathered_per_index,
          partial_segment_offsets.get(),
          host_num_partial_segments,
          partial_segment_sums.get(),
          num_gathered_per_index_warp_size_multiple);
    }

    // compute segment sums from partial segment sums
    {
      const dim3 blocks_per_grid(
          CeilDiv(num_segments * num_gathered_per_index_warp_size_multiple, threads_per_block),
          num_batches);
      ComputeSegmentSumsAndScatterKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
          dX_indices_sorted,
          dX_data,
          num_gathered_per_index,
          segment_offsets,
          num_segments,
          partial_segment_sums.get(),
          per_segment_partial_segment_offsets_in,
          host_num_partial_segments,
          num_gathered_per_index_warp_size_multiple,
          gather_dimension_size);
    }
  }
}

template <typename T, typename TIndex>
void Impl(
    cudaStream_t stream,
    const CudaScratchBufferAllocator& allocator,
    const T* dY_data,
    const TIndex* dX_indices,
    const GatheredIndexIndex_t num_gathered_indices,
    const int64_t gather_dimension_size,
    const int64_t num_gathered_per_index,
    const int64_t num_batches,
    const int32_t num_segments_in,
    const int32_t* segment_offsets,
    const int32_t last_segment_partial_segment_count,
    const int32_t last_segment_partial_segment_offset,
    const int32_t* per_segment_partial_segment_counts,
    const int32_t* per_segment_partial_segment_offsets,
    const TIndex* dX_indices_sorted,
    const TIndex* dY_indices_sorted,
    T* dX_data) {

  PartialSumsImpl(
      stream,
      allocator,
      dX_indices_sorted, dY_indices_sorted,
      dY_data, dX_data,
      num_gathered_indices, num_gathered_per_index, gather_dimension_size, num_batches,
      num_segments_in, segment_offsets,
      last_segment_partial_segment_count,
      last_segment_partial_segment_offset,
      per_segment_partial_segment_counts,
      per_segment_partial_segment_offsets);
}

}  // namespace gather_grad_internal

template <typename T, typename TIndex>
void GatherGradImpl(
    cudaStream_t stream,
    const CudaScratchBufferAllocator& allocator,
    const T* dY_data,
    const TIndex* dX_indices,
    const GatheredIndexIndex_t num_gathered_indices,
    const int64_t gather_dimension_size,
    const int64_t num_gathered_per_index,
    const int64_t num_batches,
    const int32_t num_segments,
    const int32_t* segment_offsets,
    const int32_t last_segment_partial_segment_count,
    const int32_t last_segment_partial_segment_offset,
    const int32_t* per_segment_partial_segment_counts,
    const int32_t* per_segment_partial_segment_offsets,
    const TIndex* dX_indices_sorted,
    const TIndex* dY_indices_sorted,
    T* dX_data) {
  gather_grad_internal::Impl(
      stream,
      allocator,
      dY_data, dX_indices,
      num_gathered_indices, gather_dimension_size, num_gathered_per_index, num_batches,
      num_segments,
      segment_offsets,
      last_segment_partial_segment_count,
      last_segment_partial_segment_offset,
      per_segment_partial_segment_counts,
      per_segment_partial_segment_offsets,
      dX_indices_sorted,
      dY_indices_sorted,
      dX_data);
}

#define SPECIALIZED(T, TIndex)                            \
  template void GatherGradImpl<T, TIndex>(                \
      cudaStream_t stream,                                \
      const CudaScratchBufferAllocator& allocator,        \
      const T* dY_data,                                   \
      const TIndex* dX_indices,                           \
      const GatheredIndexIndex_t num_gathered_indices,    \
      const int64_t gather_dimension_size,                \
      const int64_t num_gathered_per_index,               \
      const int64_t num_batches,                          \
      const int32_t num_segments,                         \
      const int32_t* segment_offsets,                     \
      const int32_t last_segment_partial_segment_count,   \
      const int32_t last_segment_partial_segment_offset,  \
      const int32_t* per_segment_partial_segment_counts,  \
      const int32_t* per_segment_partial_segment_offsets, \
      const TIndex* dX_indices_sorted,                    \
      const TIndex* dY_indices_sorted,                    \
      T* dX_data);

#define SPECIALIZED_WITH_IDX(T) \
  SPECIALIZED(T, int32_t)       \
  SPECIALIZED(T, int64_t)

SPECIALIZED_WITH_IDX(float)
SPECIALIZED_WITH_IDX(half)
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
SPECIALIZED_WITH_IDX(nv_bfloat16)
#endif

#undef SPECIALIZED_WITH_IDX
#undef SPECIALIZED

}  // namespace cuda
}  // namespace onnxruntime
