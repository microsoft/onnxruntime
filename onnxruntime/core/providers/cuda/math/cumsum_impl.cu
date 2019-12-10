// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "cumsum_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _CumSumKernel(
    const int64_t input_rank,
    const T* input_data,
    const int64_t axis,
    const int64_t input_dim_along_axis,
    const int64_t* input_strides,
    T* output_data,
    const int64_t output_size,
    size_t element_size,
    bool exclusive,
    bool reverse) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(indices_index, output_size);
  int dim = 0; 
  int remain = indices_index;
  size_t data_idx = 0;
  for (int i = 0; i < input_rank; ++i) {
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


void CumSumImpl(
    const int64_t input_rank,
    const void* input_data,
    const int64_t axis,
    const int64_t input_dim_along_axis,
    const int64_t* input_strides,
    void* output_data,
    const int64_t output_size,
    size_t element_size,
    bool exclusive,
    bool reverse) {

  if (output_size > 0) {

    int blocksPerGrid = static_cast<int>((output_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);

    switch (element_size) {
      case sizeof(int32_t):
        break;

      case sizeof(int64_t):
        break;

      // should not reach here as we validate if the all relevant types are supported in the Compute method 
      default:
        ORT_THROW("Unsupported element size by the CumSum CUDA kernel");
    }
  }
}

}  // namespace cuda
}  // namespace onnxruntime

