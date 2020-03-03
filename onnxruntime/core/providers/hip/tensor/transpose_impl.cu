// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/cu_inc/common.cuh"
#include "core/providers/hip/tensor/transpose_impl.h"

namespace onnxruntime {
namespace hip {

template <typename T>
__global__ void _TransposeKernel(int32_t shape_rank, const TArray<int64_t> input_strides,
                                 const T* input_data, const TArray<fast_divmod> output_strides, T* output_data, HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  HIP_LONG input_index = 0;
  HIP_LONG output_index = id;

  #pragma unroll
  for (auto dim = 0; dim < input_strides.GetCapacity(); ++dim) {
    if (dim >= shape_rank) {
      break;
    }
    int out_coord, r;
    output_strides.data_[dim].divmod(output_index, out_coord, r);
    output_index = r;
    input_index += input_strides.data_[dim] * out_coord;
  }
  output_data[id] = input_data[input_index];
}

Status TransposeImpl(size_t element_size, int32_t shape_rank, const TArray<int64_t>& input_strides,
                     const void* input_data, const TArray<fast_divmod>& fdm_output_strides, void* output_data, int64_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (element_size) {
    case sizeof(int8_t):
      hipLaunchKernelGGL(_TransposeKernel<int8_t>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          shape_rank, input_strides,
          reinterpret_cast<const ToHipType<int8_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToHipType<int8_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int16_t):
      hipLaunchKernelGGL(_TransposeKernel<int16_t>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          shape_rank, input_strides,
          reinterpret_cast<const ToHipType<int16_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToHipType<int16_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int32_t):
      hipLaunchKernelGGL(_TransposeKernel<int32_t>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          shape_rank, input_strides,
          reinterpret_cast<const ToHipType<int32_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToHipType<int32_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int64_t):
      hipLaunchKernelGGL(_TransposeKernel<int64_t>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          shape_rank, input_strides,
          reinterpret_cast<const ToHipType<int64_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToHipType<int64_t>::MappedType*>(output_data),
          N);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on HIP. Element size was ",
                             element_size);
  }

  return Status::OK();
}

}  // namespace hip
}  // namespace onnxruntime
