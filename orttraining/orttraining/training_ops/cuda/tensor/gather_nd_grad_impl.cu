// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
#include "orttraining/training_ops/cuda/tensor/gather_nd_grad_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/atomic/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _GatherNDGradKernel(
    const size_t num_slices,
    const T* update_data,
    T* output_data,
    const size_t slice_size,
    const int64_t* slice_offsets) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, num_slices * slice_size);
  uint64_t slice_offset = slice_offsets[i / slice_size];
  size_t j = i % slice_size;
  atomic_add(output_data + slice_offset + j, update_data[i]);
};

template <typename T>
void GatherNDGradImpl(
    cudaStream_t stream,
    const size_t num_slices,
    const void* update_data,
    void* output_data,
    const size_t slice_size,
    const int64_t* input_slice_offsets_data) {
  const unsigned int blocks_per_grid = static_cast<unsigned int>(CeilDiv(num_slices * slice_size, GridDim::maxThreadsPerBlock));
  _GatherNDGradKernel<T><<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      num_slices, static_cast<const T*>(update_data), static_cast<T*>(output_data), slice_size, input_slice_offsets_data);
}

#define SPECIALIZED_GRAD_IMPL(T) \
  template void GatherNDGradImpl<T>(cudaStream_t stream, const size_t num_slices, const void* update_data, void* output_data, const size_t slice_size, const int64_t* input_slice_offsets_data)

SPECIALIZED_GRAD_IMPL(float);
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
SPECIALIZED_GRAD_IMPL(half);
SPECIALIZED_GRAD_IMPL(double);
#endif
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
SPECIALIZED_GRAD_IMPL(nv_bfloat16);
#endif

}  // namespace cuda
}  // namespace onnxruntime
