// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "pad_impl.h"

namespace onnxruntime {
namespace cuda {

// PadMode enum from core/providers/cpu/tensor/pad.h, cannot use that header because of nvcc/onnxruntime incompatibility
enum class PadMode : int {
  Constant = 0,
  Reflect,
  Edge
};

template <typename T, int pad_mode>
__global__ void _PadKernel(
    const size_t shape_rank,
    const TArray<int64_t> input_dims,
    const TArray<int64_t> input_strides,
    const TArray<int64_t> lower_pads,
    const TArray<int64_t> upper_pads,
    const T pad_value,
    const T* input_data,
    const TArray<fast_divmod> fdm_output_strides,
    T* output_data,
    const size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;
  bool use_pad_value = false;
  for (int dim = 0; dim < shape_rank && !use_pad_value; ++dim) {
    int out_coord, r;
    fdm_output_strides[dim].divmod(output_index, out_coord, r);
    output_index = r;
    int in_coord = 0;
    if (out_coord < lower_pads[dim]) {
      switch ((PadMode)pad_mode) {
        case PadMode::Constant:
          use_pad_value = true;
          break;
        case PadMode::Edge:
          in_coord = 0;
          break;
        case PadMode::Reflect:
          in_coord = lower_pads[dim] - out_coord;
          break;
      }
    } else if (out_coord >= lower_pads[dim] + input_dims[dim]) {
      switch ((PadMode)pad_mode) {
        case PadMode::Constant:
          use_pad_value = true;
          break;
        case PadMode::Edge:
          in_coord = input_dims[dim] - 1;
          break;
        case PadMode::Reflect:
          in_coord = input_dims[dim] - 2 - (out_coord - (lower_pads[dim] + input_dims[dim]));
          break;
      }
    } else {
      in_coord = out_coord - lower_pads[dim];
    }
    input_index += input_strides[dim] * in_coord;
  }
  output_data[id] = use_pad_value ? (T)pad_value : input_data[input_index];
}

template <typename T>
void PadImpl(
    cudaStream_t stream,
    const size_t shape_rank,
    const TArray<int64_t>& input_dims,
    const TArray<int64_t>& input_strides,
    const TArray<int64_t>& lower_pads,
    const TArray<int64_t>& upper_pads,
    const T pad_value,
    const int pad_mode,
    const T* input_data,
    const TArray<fast_divmod>& fdm_output_strides,
    T* output_data,
    const size_t N) {
  if (N == 0) // special case where there's a dim value of 0 in the output shape
    return;

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (pad_mode) {
    case 0:
      _PadKernel<T, 0><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_dims, input_strides, lower_pads, upper_pads,
          pad_value, input_data, fdm_output_strides, output_data, N);
      break;
    case 1:
      _PadKernel<T, 1><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_dims, input_strides, lower_pads, upper_pads,
          pad_value, input_data, fdm_output_strides, output_data, N);
      break;
    case 2:
      _PadKernel<T, 2><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_dims, input_strides, lower_pads, upper_pads,
          pad_value, input_data, fdm_output_strides, output_data, N);
      break;
  }
}

#define SPECIALIZED_IMPL(T) \
  template void PadImpl<T>(cudaStream_t stream, const size_t shape_rank, const TArray<int64_t>& input_dims, const TArray<int64_t>& input_strides, const TArray<int64_t>& lower_pads, const TArray<int64_t>& upper_pads, const T pad_value, const int pad_mode, const T* input_data, const TArray<fast_divmod>& fdm_output_strides, T* output_data, const size_t N);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime
