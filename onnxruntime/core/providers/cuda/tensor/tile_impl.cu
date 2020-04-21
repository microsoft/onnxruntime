// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "tile_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _TileKernel(
    const size_t shape_rank,
    const TArray<fast_divmod> fdm_input_shape,
    const TArray<int64_t> input_strides,
    const T* input_data,
    const TArray<fast_divmod> fdm_output_strides,
    T* output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;
  for (int dim = 0; dim < shape_rank; ++dim) {
    int out_coord, r;
    fdm_output_strides[dim].divmod(output_index, out_coord, r);
    output_index = r;
    int q, in_coord;
    fdm_input_shape[dim].divmod(out_coord, q, in_coord);
    input_index += input_strides[dim] * in_coord;
  }
  output_data[id] = input_data[input_index];
}

template <typename T>
void TileImpl(
    const size_t shape_rank,
    const TArray<fast_divmod>& fdm_input_shape,
    const TArray<int64_t>& input_stride,
    const T* input_data,
    const TArray<fast_divmod>& fdm_output_strides,
    T* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _TileKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      shape_rank, fdm_input_shape, input_stride, input_data,
      fdm_output_strides, output_data, (CUDA_LONG)N);
}

#define SPECIALIZED_IMPL(T) \
  template void TileImpl<T>(const size_t shape_rank, const TArray<fast_divmod>& fdm_input_shape, const TArray<int64_t>& input_stride, const T* input_data, const TArray<fast_divmod>& fdm_output_strides, T* output_data, const size_t N);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime
