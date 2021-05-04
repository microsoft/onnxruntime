// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/gather_nd_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/atomic/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename TIndex>
__global__ void _ComputeSliceOffsetsKernel(
    const int64_t batch_dims,
    const TArray<int64_t> input_dims,
    const size_t num_slices,
    const size_t num_slices_per_batch,
    const size_t input_batch_stride,
    const size_t num_slice_dims,
    const int64_t* const sizes_from_slice_dims_data,  // num_slice_dims elements
    const TIndex* const indices_data,                 // num_slices * num_slice_dims elements
    int64_t* const input_slice_offsets_data) {        // num_slices elements
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(slice_idx, num_slices)

  const size_t batch_idx = slice_idx / num_slices_per_batch;
  const size_t base_offset = batch_idx * input_batch_stride;

  const TIndex* const slice_indices = indices_data + slice_idx * num_slice_dims;
  size_t relative_slice_offset = 0;
  for (size_t dim_idx = 0; dim_idx < num_slice_dims; ++dim_idx) {
    int64_t index = static_cast<int64_t>(slice_indices[dim_idx]);
    const size_t input_dim_idx = batch_dims + dim_idx;
    CUDA_KERNEL_ASSERT(index >= -input_dims[input_dim_idx] && index < input_dims[input_dim_idx]);
    if (index < 0) index += input_dims[input_dim_idx];

    relative_slice_offset += index * sizes_from_slice_dims_data[dim_idx];
  }

  input_slice_offsets_data[slice_idx] = base_offset + relative_slice_offset;
}

template <typename T>
__global__ void _GatherNDKernel(
    const size_t num_slices,
    const T* input_data,
    T* output_data,
    const size_t slice_size,
    const int64_t* slice_offsets) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, num_slices * slice_size)
  uint64_t slice_offset = slice_offsets[i / slice_size];
  output_data[i] = input_data[slice_offset + i % slice_size];
};

template <typename TIndex>
void ComputeSliceOffsetsImpl(
    cudaStream_t stream,
    const int64_t batch_dims,
    const TArray<int64_t> input_dims,
    const size_t num_slices,
    const size_t num_slices_per_batch,
    const size_t input_batch_stride,
    const size_t num_slice_dims,
    const int64_t* const sizes_from_slice_dims_data,  // num_slice_dims elements
    const TIndex* const indices_data,                 // num_slices * num_slice_dims elements
    int64_t* const input_slice_offsets_data) {        // num_slices elements
  const unsigned int blocks_per_grid = static_cast<unsigned int>(CeilDiv(num_slices, GridDim::maxThreadsPerBlock));
  _ComputeSliceOffsetsKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      batch_dims,
      input_dims,
      num_slices,
      num_slices_per_batch,
      input_batch_stride,
      num_slice_dims,
      sizes_from_slice_dims_data,
      indices_data,
      input_slice_offsets_data);
}

template <typename T>
void GatherNDImpl(
    cudaStream_t stream,
    const size_t num_slices,
    const void* input_data,
    void* output_data,
    const size_t slice_size,
    const int64_t* input_slice_offsets_data) {
  const unsigned int blocks_per_grid = static_cast<unsigned int>(CeilDiv(num_slices * slice_size, GridDim::maxThreadsPerBlock));
  _GatherNDKernel<T><<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      num_slices, static_cast<const T*>(input_data), static_cast<T*>(output_data), slice_size, input_slice_offsets_data);
}

#define SPECIALIZED_COMPUTE_SLICE_OFFSETS_IMPL(TIndex) \
  template void ComputeSliceOffsetsImpl<TIndex>(       \
      cudaStream_t stream,                             \
      const int64_t batch_dims,                        \
      const TArray<int64_t> input_dims,                \
      const size_t num_slices,                         \
      const size_t num_slices_per_batch,               \
      const size_t input_batch_stride,                 \
      const size_t num_slice_dims,                     \
      const int64_t* const sizes_from_slice_dims_data, \
      const TIndex* const indices_data,                \
      int64_t* const input_slice_offsets_data);

#define SPECIALIZED_IMPL(T) \
  template void GatherNDImpl<T>(cudaStream_t stream, const size_t num_slices, const void* input_data, void* output_data, const size_t slice_size, const int64_t* input_slice_offsets_data);

SPECIALIZED_COMPUTE_SLICE_OFFSETS_IMPL(int32_t)
SPECIALIZED_COMPUTE_SLICE_OFFSETS_IMPL(int64_t)

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(int64_t)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(double)
#endif
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
SPECIALIZED_IMPL(nv_bfloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime
