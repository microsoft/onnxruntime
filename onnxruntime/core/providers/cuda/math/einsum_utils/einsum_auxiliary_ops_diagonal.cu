// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "einsum_auxiliary_ops_diagonal.h"

namespace onnxruntime {

namespace cuda {

template <typename T>
__global__ void _DiagonalKernel(
    const T* input_data,
    const int64_t input_rank,
    const int64_t dim_1,
    const int64_t dim_2,
    const TArray<int64_t> input_strides,
    T* output_data,
    const TArray<fast_divmod> output_strides,
    const size_t output_size) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(output_idx, output_size);
  int dim = 0;
  int remain = output_idx;
  size_t input_idx = 0;
  int64_t current_input_axis = 0;

  // Output's rank is always 1 less than the input's rank
  for (int i = 0; i < input_rank - 1; ++i) {
    output_strides[i].divmod(remain, dim, remain);
    if (i == dim_1) {
      // Process dim_2 as dim_2 needs to have the same dim value as dim_1
      // For example: given a tensor of shape [2, 3, 3] and parsing the diagonal along axes `1` and `2`
      // we need to parse elements in input[j, i, i] (j -> 0 to 1; and i -> 0 to 2)
      // and place them in output[j, i] and by definition of diagonal parsing dim_1 has to be equal to
      // dim_2
      input_idx += input_strides[dim_2] * dim;
    }
    input_idx += input_strides[current_input_axis] * dim;

    // Update current_input_axis
    // If it is dim_2, skip it
    if (++current_input_axis == dim_2) {
      ++current_input_axis;
    }
  }
  output_data[output_idx] = input_data[input_idx];
}

void DiagonalImpl(
    cudaStream_t stream,
    const void* input_data,
    const int64_t input_rank,
    const int64_t dim_1,
    const int64_t dim_2,
    const TArray<int64_t> input_strides,
    void* output_data,
    const TArray<fast_divmod> output_strides,
    const size_t output_size,
    size_t element_size) {
  if (output_size > 0) {
    int blocksPerGrid = static_cast<int>((output_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);

    switch (element_size) {
      case sizeof(int32_t):
        _DiagonalKernel<int32_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data), input_rank, dim_1, dim_2,
            input_strides, reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data), output_strides,
            output_size);
        break;

      case sizeof(int64_t):
        _DiagonalKernel<int64_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data), input_rank, dim_1, dim_2,
            input_strides, reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data), output_strides,
            output_size);
        break;

      case sizeof(int16_t):
        _DiagonalKernel<half><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const half*>(input_data), input_rank, dim_1, dim_2,
            input_strides, reinterpret_cast<half*>(output_data), output_strides,
            output_size);
        break;

      // Should not hit this as we do not register kernel support for types that will run into this
      default:
        ORT_THROW("Einsum Op: Diagonal parsing unsupported");
    }
  }
}

}  // namespace cuda

}  // namespace onnxruntime
