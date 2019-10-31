// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "gather_elements_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename Tin>
__global__ void _GatherElementsKernel(
    const int64_t rank,
    const T* input_data,
    const int64_t input_dim_along_axis,
    const int64_t* input_strides,
    const Tin* indices_data,
    const int64_t indices_size,
    const fast_divmod* indices_strides,
    const int axis,
    T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(indices_index, indices_size);
  int dim, remain = indices_index;
  size_t data_idx = 0;
  for (int i = 0; i < rank; ++i) {
    indices_strides[i].divmod(remain, dim, remain);
    if (i == axis) {
      dim = static_cast<int>(indices_data[indices_index]);
      if (dim < -input_dim_along_axis || dim >= input_dim_along_axis) {
        return; // Invalid index
      }
      if (dim < 0) {
        dim += input_dim_along_axis;
      }
    }
    data_idx += input_strides[i] * dim;
  }
  output_data[indices_index] = input_data[data_idx];
}

template <typename T, typename Tin>
void GatherElementsImpl(
    const int64_t rank,
    const T* input_data,
    const int64_t input_size,
    const int64_t input_dim_along_axis,
    const int64_t* input_strides,
    const Tin* indices_data,
    const int64_t indices_size,
    const fast_divmod* indices_strides,
    const int axis,
    T* output_data) {

  if (input_data != output_data) {
    cudaMemcpyAsync(output_data, input_data, input_size * sizeof(T), cudaMemcpyDeviceToDevice, 0);
  }

  if (indices_size > 0) {
    int blocksPerGrid = static_cast<int>((indices_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);
    _GatherElementsKernel<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        rank, input_data, input_dim_along_axis, input_strides,
        indices_data, indices_size, indices_strides,
        axis, output_data);
  }
}

#define SPECIALIZED_IMPL(T)                           \
  template void GatherElementsImpl<T, int32_t>(       \
      const int64_t rank,                             \
      const T* input_data,                            \
      const int64_t input_size,                       \
      const int64_t input_dim_along_axis,             \
      const int64_t* input_strides,                   \
      const int32_t* indices_data,                    \
      const int64_t indices_size,                     \
      const fast_divmod* indices_strides,             \
      const int axis,                                 \
      T* output_data);                                \
  template void GatherElementsImpl<T, int64_t>(       \
      const int64_t rank,                             \
      const T* input_data,                            \
      const int64_t input_size,                       \
      const int64_t input_dim_along_axis,             \
      const int64_t* input_strides,                   \
      const int64_t* indices_data,                    \
      const int64_t indices_size,                     \
      const fast_divmod* indices_strides,             \
      const int axis,                                 \
      T* output_data);                                \

SPECIALIZED_IMPL(int8_t)
SPECIALIZED_IMPL(int16_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(uint8_t)
SPECIALIZED_IMPL(uint16_t)
SPECIALIZED_IMPL(uint32_t)
SPECIALIZED_IMPL(uint64_t)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(bool)

}  // namespace cuda
}  // namespace onnxruntime

