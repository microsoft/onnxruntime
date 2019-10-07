// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "transpose_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _TransposeKernel(int32_t rank, CUDA_LONG N,
                                 const TArray<int64_t> input_strides, const T* input_data,
                                 const TArray<fast_divmod> output_strides, T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;

  #pragma unroll
  for (auto dim = 0; dim < MAX_ARRAY_SIZE; ++dim) {
    if (dim >= rank) {
      break;
    }
    int out_coord, r;
    output_strides.data_[dim].divmod(output_index, out_coord, r);
    output_index = r;
    input_index += input_strides.data_[dim] * out_coord;
  }
  output_data[id] = input_data[input_index];
}

template <typename T>
void TransposeImpl(int32_t rank, int64_t N,
                   const TArray<int64_t>& input_strides, const T* input_data,
                   const TArray<fast_divmod>& output_strides, T* output_data) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _TransposeKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      rank, (CUDA_LONG)N, input_strides, input_data, output_strides, output_data);
}

#define SPECIALIZED_IMPL(T)                                                           \
  template void TransposeImpl<T>(int32_t rank, int64_t N,                             \
                      const TArray<int64_t>& input_strides, const T* input_data,      \
                      const TArray<fast_divmod>& output_strides, T* output_data);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime
