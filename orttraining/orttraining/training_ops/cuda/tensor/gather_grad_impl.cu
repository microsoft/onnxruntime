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
// The segments will be:  '--'  '-----'  '
//
// The segments can be processed in parallel, or further divided into partial
// segments for increased parallelism.

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
      num_gathered_indices, 0, sizeof(TIndex)*8, stream));

  auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);
  CUDA_CALL_THROW(cub::DeviceRadixSort::SortPairs(
      temp_storage.get(), temp_storage_size_bytes,
      dX_indices, dX_indices_sorted.get(),
      dY_indices.get(), dY_indices_sorted.get(),
      num_gathered_indices, 0, sizeof(TIndex)*8, stream));

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

// adapted from here:
// https://github.com/pytorch/pytorch/blob/b186831c08e0e4e447eedb8a5cfab582995d37f9/aten/src/ATen/native/cuda/Embedding.cu#L121
template <typename T, typename TIndex, int NumElementsPerThread>
__global__ void DirectSumKernel(
    const TIndex* dX_indices_sorted,
    const TIndex* dY_indices_sorted,
    const T* dY_data,
    T* dX_data,
    GatheredIndexIndex_t num_gathered_indices,
    int64_t num_gathered_per_index,
    int64_t gather_dimension_size,
    int64_t num_batches) {
  GatheredIndexIndex_t idx = blockIdx.x * 4 + threadIdx.y;

  if (idx < num_gathered_indices && (idx == 0 || dX_indices_sorted[idx] != dX_indices_sorted[idx - 1])) {
    do {
      // All index values are expected to be within bounds [-s, s-1] along axis of size s.
      auto target_row = dX_indices_sorted[idx];
      if (target_row < 0) target_row += gather_dimension_size;
      for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const auto gathered_element_idx_start = threadIdx.x + blockIdx.y * blockDim.x * NumElementsPerThread;
        const auto dX_row_offset =
            (batch_idx * gather_dimension_size + target_row) * num_gathered_per_index;
        const auto dY_row_offset =
            (batch_idx * num_gathered_indices + dY_indices_sorted[idx]) * num_gathered_per_index;

        AccumulationType_t<T> dY_value[NumElementsPerThread];
        AccumulationType_t<T> dX_value[NumElementsPerThread];

#pragma unroll
        for (int ii = 0; ii < NumElementsPerThread; ii++) {
          const auto gathered_element_idx = gathered_element_idx_start + ii * GPU_WARP_SIZE;
          if (gathered_element_idx < num_gathered_per_index) {
            dY_value[ii] = static_cast<AccumulationType_t<T>>(dY_data[dY_row_offset + gathered_element_idx]);
            dX_value[ii] = static_cast<AccumulationType_t<T>>(dX_data[dX_row_offset + gathered_element_idx]);
          }
        }

#pragma unroll
        for (int ii = 0; ii < NumElementsPerThread; ii++) {
          dX_value[ii] += dY_value[ii];
        }

#pragma unroll
        for (int ii = 0; ii < NumElementsPerThread; ii++) {
          const auto gathered_element_idx = gathered_element_idx_start + ii * GPU_WARP_SIZE;
          if (gathered_element_idx < num_gathered_per_index) {
            dX_data[dX_row_offset + gathered_element_idx] = static_cast<T>(dX_value[ii]);
          }
        }
      }
      idx++;
    } while (idx < num_gathered_indices && dX_indices_sorted[idx] == dX_indices_sorted[idx - 1]);
  }
}

// directly sum gathered dY values into the corresponding dX value
template <typename T, typename TIndex>
void DirectSumImpl(
    cudaStream_t stream,
    const TIndex* dX_indices_sorted,
    const TIndex* dY_indices_sorted,
    const T* dY_data,
    T* dX_data,
    GatheredIndexIndex_t num_gathered_indices,
    int64_t num_gathered_per_index,
    int64_t gather_dimension_size,
    int64_t num_batches) {
  dim3 block(GPU_WARP_SIZE, 4);
  dim3 grid(CeilDiv(num_gathered_indices, 4), CeilDiv(num_gathered_per_index, GridDim::maxElementsPerThread * GPU_WARP_SIZE));

  DirectSumKernel<T, TIndex, GridDim::maxElementsPerThread><<<grid, block, 0, stream>>>(
      dX_indices_sorted,
      dY_indices_sorted,
      dY_data,
      dX_data,
      num_gathered_indices,
      num_gathered_per_index,
      gather_dimension_size,
      num_batches);
}

// partial sums implementation adapted from here:
// https://github.com/pytorch/pytorch/blob/b186831c08e0e4e447eedb8a5cfab582995d37f9/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu

__global__ void ComputePerSegmentPartialSegmentCountsKernel(
    SegmentIndex_t* ret, const GatheredIndexIndex_t* segment_offsets,
    SegmentIndex_t num_of_segments, GatheredIndexIndex_t num_gathered_indices) {
  const auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < num_of_segments) {
    const auto idx_start = segment_offsets[id];
    const auto idx_end = (id == num_of_segments - 1) ? num_gathered_indices : segment_offsets[id + 1];
    const auto size = idx_end - idx_start;
    ret[id] = CeilDiv(size, kMaxPartialSegmentSize);
  }
}

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
    const GatheredIndexIndex_t* segment_offsets,
    SegmentIndex_t num_segments) {
  // each segment is split into partial segments of at most
  // kMaxPartialSegmentSize index pairs.

  // compute the number of partial segments per segment
  auto per_segment_partial_segment_counts = allocator.GetScratchBuffer<SegmentIndex_t>(num_segments);
  {
    const auto blocks_per_grid = CeilDiv(num_gathered_indices, GridDim::maxThreadsPerBlock);
    ComputePerSegmentPartialSegmentCountsKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        per_segment_partial_segment_counts.get(),
        segment_offsets, num_segments, num_gathered_indices);
  }

  // compute partial segment offsets per segment
  auto per_segment_partial_segment_offsets = GetOffsetsFromCounts(
      stream, allocator, per_segment_partial_segment_counts.get(), num_segments);

  SegmentIndex_t host_num_partial_segments = 0;
  {
    SegmentIndex_t last_segment_partial_segment_offset = 0,
                   last_segment_partial_segment_count = 0;
    // CPU/GPU sync!
    CUDA_CALL_THROW(cudaMemcpyAsync(
        &last_segment_partial_segment_offset,
        &per_segment_partial_segment_offsets.get()[num_segments - 1],
        sizeof(SegmentIndex_t), cudaMemcpyDeviceToHost, stream));
    // CPU/GPU sync!
    CUDA_CALL_THROW(cudaMemcpyAsync(
        &last_segment_partial_segment_count,
        &per_segment_partial_segment_counts.get()[num_segments - 1],
        sizeof(SegmentIndex_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
    host_num_partial_segments =
        last_segment_partial_segment_offset + last_segment_partial_segment_count;
  }

  // compute index offsets per partial segment
  auto partial_segment_offsets = allocator.GetScratchBuffer<GatheredIndexIndex_t>(host_num_partial_segments);
  {
    const auto blocks_per_grid = CeilDiv(num_segments, GridDim::maxThreadsPerBlock);
    ComputePartialSegmentOffsetsKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        partial_segment_offsets.get(),
        per_segment_partial_segment_counts.get(),
        per_segment_partial_segment_offsets.get(),
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
          per_segment_partial_segment_offsets.get(),
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
    T* dX_data) {
  IAllocatorUniquePtr<TIndex> dX_indices_sorted, dY_indices_sorted;
  GetSortedIndices(
      stream,
      allocator,
      dX_indices, num_gathered_indices,
      dX_indices_sorted, dY_indices_sorted);

  // get number of segments and segment counts
  SegmentIndex_t host_num_segments = 0;
  auto segment_counts = allocator.GetScratchBuffer<GatheredIndexIndex_t>(num_gathered_indices);
  {
    auto num_segments = allocator.GetScratchBuffer<SegmentIndex_t>(1);
    size_t temp_storage_size_bytes = 0;
    CUDA_CALL_THROW(cub::DeviceRunLengthEncode::Encode(
        nullptr, temp_storage_size_bytes,
        dX_indices_sorted.get(), cub::DiscardOutputIterator<TIndex>{}, segment_counts.get(),
        num_segments.get(), num_gathered_indices, stream));

    auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);
    CUDA_CALL_THROW(cub::DeviceRunLengthEncode::Encode(
        temp_storage.get(), temp_storage_size_bytes,
        dX_indices_sorted.get(), cub::DiscardOutputIterator<TIndex>{}, segment_counts.get(),
        num_segments.get(), num_gathered_indices, stream));

    // CPU/GPU sync!
    CUDA_CALL_THROW(cudaMemcpyAsync(
        &host_num_segments, num_segments.get(), sizeof(SegmentIndex_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  }

  // get largest segment size and use that to select implementation
  GatheredIndexIndex_t host_max_segment_count = 0;
  {
    auto max_segment_count = allocator.GetScratchBuffer<GatheredIndexIndex_t>(1);

    size_t temp_storage_size_bytes = 0;
    CUDA_CALL_THROW(cub::DeviceReduce::Max(
        nullptr, temp_storage_size_bytes,
        segment_counts.get(), max_segment_count.get(), host_num_segments, stream));

    auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);
    CUDA_CALL_THROW(cub::DeviceReduce::Max(
        temp_storage.get(), temp_storage_size_bytes,
        segment_counts.get(), max_segment_count.get(), host_num_segments, stream));

    // CPU/GPU sync!
    CUDA_CALL_THROW(cudaMemcpyAsync(
        &host_max_segment_count, max_segment_count.get(), sizeof(GatheredIndexIndex_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  }

  constexpr GatheredIndexIndex_t kMaxSegmentSizeThreshold = 32;
  if (host_max_segment_count <= kMaxSegmentSizeThreshold) {
    DirectSumImpl(
        stream, dX_indices_sorted.get(), dY_indices_sorted.get(),
        dY_data, dX_data,
        num_gathered_indices, num_gathered_per_index, gather_dimension_size, num_batches);
  } else {
    auto segment_offsets = GetOffsetsFromCounts(
        stream, allocator, segment_counts.get(), host_num_segments);
    segment_counts.reset();

    PartialSumsImpl(
        stream,
        allocator,
        dX_indices_sorted.get(), dY_indices_sorted.get(),
        dY_data, dX_data,
        num_gathered_indices, num_gathered_per_index, gather_dimension_size, num_batches,
        segment_offsets.get(), host_num_segments);
  }
}

// this is a backup implementation that doesn't incur GPU/CPU syncs, but
// doesn't perform well if there are many duplicate values in dX_indices
template <typename T, typename TIndex>
void Impl_Simplified(
    cudaStream_t stream,
    const CudaScratchBufferAllocator& allocator,
    const T* dY_data,
    const TIndex* dX_indices,
    const GatheredIndexIndex_t num_gathered_indices,
    const int64_t gather_dimension_size,
    const int64_t num_gathered_per_index,
    const int64_t num_batches,
    T* dX_data) {
  IAllocatorUniquePtr<TIndex> dX_indices_sorted, dY_indices_sorted;
  GetSortedIndices(
      stream,
      allocator,
      dX_indices, num_gathered_indices,
      dX_indices_sorted, dY_indices_sorted);

  dim3 block(GPU_WARP_SIZE, 4);
  dim3 grid(CeilDiv(num_gathered_indices, 4), CeilDiv(num_gathered_per_index, GridDim::maxElementsPerThread * GPU_WARP_SIZE));

  DirectSumKernel<T, TIndex, GridDim::maxElementsPerThread><<<grid, block, 0, stream>>>(
      dX_indices_sorted.get(),
      dY_indices_sorted.get(),
      dY_data,
      dX_data,
      num_gathered_indices,
      num_gathered_per_index,
      gather_dimension_size,
      num_batches);
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
    T* dX_data) {
  gather_grad_internal::Impl(
      stream,
      allocator,
      dY_data, dX_indices,
      num_gathered_indices, gather_dimension_size, num_gathered_per_index, num_batches,
      dX_data);
}

#define SPECIALIZED(T, TIndex)                         \
  template void GatherGradImpl<T, TIndex>(             \
      cudaStream_t stream,                             \
      const CudaScratchBufferAllocator& allocator,     \
      const T* dY_data,                                \
      const TIndex* dX_indices,                        \
      const GatheredIndexIndex_t num_gathered_indices, \
      const int64_t gather_dimension_size,             \
      const int64_t num_gathered_per_index,            \
      const int64_t num_batches,                       \
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
