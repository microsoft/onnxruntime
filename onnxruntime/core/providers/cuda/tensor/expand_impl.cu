// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "expand_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _ExpandKernel(
    const size_t rank,
    const size_t N,
    const size_t N_input,
    const T* input_data,
    T* output_data,
    const fast_divmod* fdm_input_dims,
    const fast_divmod* fdm_output_dims) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  // initialize
  int64_t output_index = id;
  int64_t input_index = 0;

  // use striding when tensor is larger than grid
  int stride = blockDim.x * gridDim.x;
  auto outputSubDimSize = N;
  auto inputSubDimSize = N_input;
  auto out_coord = output_index;

  // translate indices to coordinates. copy expanded dims from source
  while (output_index < N) {
    for (int64_t i = 0; i < rank; i++) {
      outputSubDimSize = fdm_output_dims[i].div(outputSubDimSize);
      inputSubDimSize = fdm_input_dims[i].div(inputSubDimSize);
      auto new_out_coord = out_coord / outputSubDimSize;
      auto in_coord = (new_out_coord > (fdm_input_dims[i].d_ - 1)) ? fdm_input_dims[i].d_ - 1 : new_out_coord;
      input_index += inputSubDimSize * in_coord;
      out_coord -= new_out_coord * outputSubDimSize;
    }
    output_data[output_index] = input_data[input_index];
    output_index += stride;
    out_coord = output_index;
    outputSubDimSize = N;
    inputSubDimSize = N_input;
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
    const fast_divmod* fdm_input_dims,
    const fast_divmod* fdm_output_dims) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _ExpandKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      rank, N, N_input, input_data, output_data, fdm_input_dims, fdm_output_dims);
}

#define SPECIALIZED_IMPL(T) \
  template void ExpandImpl<T>(const size_t rank, const size_t N, const size_t N_input, const T* input_data, T* output_data, const fast_divmod* fdm_input_dims, const fast_divmod* fdm_output_dims);

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
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime
