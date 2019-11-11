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
    const int64_t axis,
    T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(indices_index, indices_size);
  int dim = 0; 
  int remain = indices_index;
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

template <typename Tin>
void GatherElementsImpl(
    const int64_t rank,
    const void* input_data,
    const int64_t input_size,
    const int64_t input_dim_along_axis,
    const int64_t* input_strides,
    const Tin* indices_data,
    const int64_t indices_size,
    const fast_divmod* indices_strides,
    const int64_t axis,
    void* output_data,
    size_t element_size) {

  if (indices_size > 0) {

    int blocksPerGrid = static_cast<int>((indices_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);

    switch (element_size) {
      case sizeof(int8_t):
         _GatherElementsKernel<int8_t, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            rank, reinterpret_cast<const ToCudaType<int8_t>::MappedType*>(input_data), input_dim_along_axis, input_strides,
            indices_data, indices_size, indices_strides,
            axis, reinterpret_cast<ToCudaType<int8_t>::MappedType*>(output_data));
        break;

      case sizeof(int16_t):
        _GatherElementsKernel<int16_t, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            rank, reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data), input_dim_along_axis, input_strides,
            indices_data, indices_size, indices_strides,
            axis, reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data));
        break;

      case sizeof(int32_t):
        _GatherElementsKernel<int32_t, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            rank, reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data), input_dim_along_axis, input_strides,
            indices_data, indices_size, indices_strides,
            axis, reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data));
        break;

      case sizeof(int64_t):
        _GatherElementsKernel<int64_t, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            rank, reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data), input_dim_along_axis, input_strides,
            indices_data, indices_size, indices_strides,
            axis, reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data));
        break;

      // should not reach here as we validate if the all relevant types are supported in the Compute method 
      default:
        ORT_THROW("Unsupported element size by the GatherElements CUDA kernel");
    }
  }
}

template void GatherElementsImpl<int32_t>(
    const int64_t rank,
    const void* input_data,
    const int64_t input_size,
    const int64_t input_dim_along_axis,
    const int64_t* input_strides,
    const int32_t* indices_data,
    const int64_t indices_size,
    const fast_divmod* indices_strides,
    const int64_t axis,
    void* output_data,
    size_t element_size);

template void GatherElementsImpl<int64_t>(
    const int64_t rank,
    const void* input_data,
    const int64_t input_size,
    const int64_t input_dim_along_axis,
    const int64_t* input_strides,
    const int64_t* indices_data,
    const int64_t indices_size,
    const fast_divmod* indices_strides,
    const int64_t axis,
    void* output_data,
    size_t element_size);

}  // namespace cuda
}  // namespace onnxruntime

