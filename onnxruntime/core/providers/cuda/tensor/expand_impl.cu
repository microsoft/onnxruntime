// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "expand_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int pad_mode>
__global__ void _ExpandKernel(
    const size_t rank,
    const size_t N,
    const size_t N_input,
    const T* input_data,
    T* output_data,
    const int64_t* input_dims,
    const int64_t* output_dims) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  // initialize
  int64_t output_index = id;
  int64_t input_index = 0;

  // use striding when tensor is larger than grid
  int stride = blockDim.x * gridDim.x;
  auto sizeTillDimensionOutput = N;
  auto sizeTillDimensionInput = N_input;
  auto out_coord = output_index;

  // translate indices to coordinates. copy expanded dims from source
  while (output_index < N) {
    for (int64_t i = 0; i < rank; i++) {
      sizeTillDimensionOutput /= output_dims[i];
      sizeTillDimensionInput /= input_dims[i];
      auto new_out_coord = out_coord / sizeTillDimensionOutput;
      auto in_coord = (new_out_coord > (input_dims[i] - 1)) ? input_dims[i] - 1 : new_out_coord;
      input_index += sizeTillDimensionInput * in_coord;
      out_coord -= new_out_coord * sizeTillDimensionOutput;
    }
    output_data[output_index] = input_data[input_index];
    output_index += stride;
    out_coord = output_index;
    sizeTillDimensionOutput = N;
    sizeTillDimensionInput = N_input;  
    input_index = 0;
  }
}

template <typename T>
void ExpandImpl(
    const size_t rank,
    const size_t N,
    const size_t N_input,
    const T* input_data,
    T* output_data,
    const int64_t* input_dims,
    const int64_t* output_dims) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _ExpandKernel<T, 0><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      rank, N, N_input, input_data, output_data, input_dims, output_dims);
}

#define SPECIALIZED_IMPL(T) \
  template void ExpandImpl<T>(const size_t rank, const size_t N, const size_t N_input, const T* input_data, T* output_data, const int64_t* input_dims, const int64_t* output_dims);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(int8_t)
SPECIALIZED_IMPL(int16_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(uint8_t)
SPECIALIZED_IMPL(uint16_t)
SPECIALIZED_IMPL(uint32_t)
SPECIALIZED_IMPL(uint64_t)
SPECIALIZED_IMPL(bool)

}  // namespace cuda
}  // namespace onnxruntime
