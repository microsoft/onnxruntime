// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/fast_divmod.h"

#include "cumsum_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _CumSumKernel(
    const T* input_data,
    const fast_divmod fast_divmod_input_dim_along_axis,
    const fast_divmod fast_divmod_input_stride_along_axis,
    T* output_data,
    const int64_t output_size,
    const bool exclusive,
    const bool reverse) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(indices_index, output_size);

  int input_dim_along_axis = fast_divmod_input_dim_along_axis.d_;
  int input_stride_along_axis = fast_divmod_input_stride_along_axis.d_;

  int axis_dim = 0;
  int div = fast_divmod_input_stride_along_axis.div(static_cast<int>(indices_index));
  fast_divmod_input_dim_along_axis.divmod(div, div, axis_dim);

  int start = 0;
  int end = 0;

  if (!reverse && !exclusive) {
    start = 0;
    end = axis_dim;
  
  } else if (reverse && !exclusive) {
    start = axis_dim;
    end = input_dim_along_axis - 1;

  } else if (!reverse && exclusive) {
    start = 0;
    end = axis_dim - 1;

  } else { // reverse && exclusive
    start = axis_dim + 1;
    end = input_dim_along_axis - 1;

  }

  // count the number of elements to accumulate the sum
  int count = end - start + 1;
  if (count <= 0) {
    output_data[indices_index] = 0;
    return;  
  }

  // adjust start index based on the above identified start dim value along the axis of interest
  int data_index = static_cast<int>(indices_index) + (start - axis_dim) * input_stride_along_axis;
  T sum = 0;

  // keep accumulating values from the start index for 'count' times and skip appropriately 
  while (count != 0) {
    sum += input_data[data_index];
    data_index += input_stride_along_axis;
    --count;
  }

  output_data[indices_index] = sum;
}

template <typename T>
void CumSumImpl(
    cudaStream_t stream,
    const T* input_data,
    const fast_divmod& input_dim_along_axis,
    const fast_divmod& input_stride_along_axis,
    T* output_data,
    int64_t output_size,
    bool exclusive,
    bool reverse) {
  if (output_size > 0) {
    int blocksPerGrid = static_cast<int>((output_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);

    _CumSumKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data,
                                                                        input_dim_along_axis,
                                                                        input_stride_along_axis,
                                                                        output_data,
                                                                        output_size,
                                                                        exclusive,
                                                                        reverse);
  }
}

template void CumSumImpl<int32_t>(
    cudaStream_t stream,
    const int32_t* input_data,
    const fast_divmod& input_dim_along_axis,
    const fast_divmod& input_stride_along_axis,
    int32_t* output_data,
    int64_t output_size,
    bool exclusive,
    bool reverse);

template void CumSumImpl<int64_t>(
    cudaStream_t stream,
    const int64_t* input_data,
    const fast_divmod& input_dim_along_axis,
    const fast_divmod& input_stride_along_axis,
    int64_t* output_data,
    int64_t output_size,
    bool exclusive,
    bool reverse);

template void CumSumImpl<uint32_t>(
    cudaStream_t stream,
    const uint32_t* input_data,
    const fast_divmod& input_dim_along_axis,
    const fast_divmod& input_stride_along_axis,
    uint32_t* output_data,
    int64_t output_size,
    bool exclusive,
    bool reverse);

template void CumSumImpl<uint64_t>(
    cudaStream_t stream,
    const uint64_t* input_data,
    const fast_divmod& input_dim_along_axis,
    const fast_divmod& input_stride_along_axis,
    uint64_t* output_data,
    int64_t output_size,
    bool exclusive,
    bool reverse);

template void CumSumImpl<float>(
    cudaStream_t stream,
    const float* input_data,
    const fast_divmod& input_dim_along_axis,
    const fast_divmod& input_stride_along_axis,
    float* output_data,
    int64_t output_size,
    bool exclusive,
    bool reverse);

template void CumSumImpl<double>(
    cudaStream_t stream,
    const double* input_data,
    const fast_divmod& input_dim_along_axis,
    const fast_divmod& input_stride_along_axis,
    double* output_data,
    int64_t output_size,
    bool exclusive,
    bool reverse);

template void CumSumImpl<half>(
    cudaStream_t stream,
    const half* input_data,
    const fast_divmod& input_dim_along_axis,
    const fast_divmod& input_stride_along_axis,
    half* output_data,
    int64_t output_size,
    bool exclusive,
    bool reverse);

}  // namespace cuda
}  // namespace onnxruntime

