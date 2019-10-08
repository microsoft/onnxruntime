// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "scatter_elements_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename Tin>
__global__ void _ScatterElementsKernel(
    const int rank,
    const T* input_data,
    const int64_t* input_dims,
    const int64_t* input_strides,
    const Tin* indices_data,
    const int64_t indices_size,
    const int64_t* indices_dims,
    const fast_divmod* indices_strides,
    const T* updates,
    const int axis,
    T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(indices_index, indices_size);
  int dim, remain = indices_index;
  size_t data_idx = 0;
  for (int i = 0; i < rank; ++i) {
    indices_strides[i].divmod(remain, dim, remain);
    if (i == axis) {
      dim = (int)(indices_data[indices_index]);
      if (dim < -input_dims[i] || dim >= input_dims[i]) {
        return; // Invalid index
      }
      if (dim < 0) dim += input_dims[i];
    }
    data_idx += input_strides[i] * dim;
  }
  output_data[data_idx] = updates[indices_index];
}

template <typename T, typename Tin>
void ScatterElementsImpl(
    const int rank,
    const T* input_data,
    const int64_t input_size,
    const int64_t* input_dims,
    const int64_t* input_strides,
    const Tin* indices_data,
    const int64_t indices_size,
    const int64_t* indices_dims,
    const fast_divmod* indices_strides,
    const T* updates,
    const int axis,
    T* output_data) {

  if (input_data != output_data) {
    cudaMemcpyAsync(output_data, input_data, input_size * sizeof(T), cudaMemcpyDeviceToDevice, 0);
  }

  int blocksPerGrid = (int)((indices_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);
  _ScatterElementsKernel<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      rank, input_data, input_dims, input_strides,
      indices_data, indices_size, indices_dims, indices_strides,
      updates, axis, output_data);
}

#define SPECIALIZED_IMPL(T)                           \
  template void ScatterElementsImpl<T, int32_t>(      \
      const int rank,                                 \
      const T* input_data,                            \
      const int64_t input_size,                       \
      const int64_t* input_dims,                      \
      const int64_t* input_strides,                   \
      const int32_t* indices_data,                    \
      const int64_t indices_size,                     \
      const int64_t* indices_dims,                    \
      const fast_divmod* indices_strides,             \
      const T* updates,                               \
      const int axis,                                 \
      T* output_data);                                \
  template void ScatterElementsImpl<T, int64_t>(      \
      const int rank,                                 \
      const T* input_data,                            \
      const int64_t input_size,                       \
      const int64_t* input_dims,                      \
      const int64_t* input_strides,                   \
      const int64_t* indices_data,                    \
      const int64_t indices_size,                     \
      const int64_t* indices_dims,                    \
      const fast_divmod* indices_strides,             \
      const T* updates,                               \
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

