// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "expand_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void ExpandKernel(
    const size_t rank,
    const size_t N,
    const size_t N_input,
    const T* input_data,
    T* output_data,
    const fast_divmod* fdm_input_dims,
    const fast_divmod* fdm_output_dims,
    const fast_divmod* fdm_output_subdim_size) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // initialize
  auto output_index = id;
  auto input_index = 0;
  auto input_subdim_size = N_input;
  auto out_coord = output_index;
  // use striding when tensor is larger than grid
  int stride = blockDim.x * gridDim.x;

  // translate indices to coordinates. copy expanded dims from source
  while (output_index < N) {
    for (int64_t i = 0; i < rank; i++) {
      input_subdim_size = fdm_input_dims[i].div(input_subdim_size);
      auto new_out_coord = fdm_output_subdim_size[i].div(out_coord);
      auto in_coord = (new_out_coord > (fdm_input_dims[i].d_ - 1)) ? fdm_input_dims[i].d_ - 1 : new_out_coord;
      input_index += input_subdim_size * in_coord;
      out_coord -= new_out_coord * fdm_output_subdim_size[i].d_;
    }
    output_data[output_index] = input_data[input_index];
    output_index += stride;
    out_coord = output_index;
    input_subdim_size = N_input;
    input_index = 0;
  }
}

Status ExpandImpl(
    const size_t element_size,
    const size_t rank,
    const size_t N,
    const size_t N_input,
    const void* input_data,
    void* output_data,
    const fast_divmod* fdm_input_dims,
    const fast_divmod* fdm_output_dims,
    const fast_divmod* fdm_output_subdim_size) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  switch (element_size) {
    case sizeof(uint8_t):
      ExpandKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          rank, N, N_input,
          reinterpret_cast<const ToCudaType<uint8_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<uint8_t>::MappedType*>(output_data),
          fdm_input_dims, fdm_output_dims, fdm_output_subdim_size);
      break;
    case sizeof(uint16_t):
      ExpandKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          rank, N, N_input,
          reinterpret_cast<const ToCudaType<uint16_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<uint16_t>::MappedType*>(output_data),
          fdm_input_dims, fdm_output_dims, fdm_output_subdim_size);
      break;
    case sizeof(uint32_t):
      ExpandKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          rank, N, N_input,
          reinterpret_cast<const ToCudaType<uint32_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<uint32_t>::MappedType*>(output_data),
          fdm_input_dims, fdm_output_dims, fdm_output_subdim_size);
      break;
    case sizeof(uint64_t):
      ExpandKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          rank, N, N_input,
          reinterpret_cast<const ToCudaType<uint64_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<uint64_t>::MappedType*>(output_data),
          fdm_input_dims, fdm_output_dims, fdm_output_subdim_size);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Expand operator");
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
