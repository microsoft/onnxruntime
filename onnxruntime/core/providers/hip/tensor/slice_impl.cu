// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/cu_inc/common.cuh"
#include "slice_impl.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

template <typename T >
__global__ void _SliceKernel(const int32_t dimension_count,
                             const TArray<int64_t> starts,
                             const TArray<int64_t> steps,
                             const TArray<int64_t> input_strides,
                             const TArray<fast_divmod> output_strides,
                             const T* input_data,
                             T* output_data,
                             const HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  HIP_LONG input_index = 0;
  int div;
  int mod = id;
  int value = id;
  int dim = 0;
  #pragma unroll
  for (; dim < starts.GetCapacity(); ++dim) {
    if (dim >= dimension_count - 1) {
      break;
    }

    output_strides.data_[dim].divmod(value, div, mod);
    input_index += (starts.data_[dim] + div * steps.data_[dim]) * input_strides.data_[dim];
    value = mod;
  }
  input_index += starts.data_[dim] + mod * steps.data_[dim];
  output_data[id] = input_data[input_index];
}

Status SliceImpl(const size_t element_size,
                const int32_t dimension_count,
                const TArray<int64_t>* starts,
                const TArray<int64_t>* steps,
                const TArray<int64_t>* input_strides,
                const TArray<fast_divmod>* output_strides,
                const void* input_data,
                void* output_data,
                const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  switch (element_size) {
    case sizeof(int8_t):
      hipLaunchKernelGGL(_SliceKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          dimension_count, *starts, *steps, *input_strides, *output_strides,
          reinterpret_cast<const ToHipType<int8_t>::MappedType*>(input_data),
          reinterpret_cast<ToHipType<int8_t>::MappedType*>(output_data),
          (HIP_LONG)N);
      break;
    case sizeof(int16_t):
      hipLaunchKernelGGL(_SliceKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          dimension_count, *starts, *steps, *input_strides, *output_strides,
          reinterpret_cast<const ToHipType<int16_t>::MappedType*>(input_data),
          reinterpret_cast<ToHipType<int16_t>::MappedType*>(output_data),
          (HIP_LONG)N);
      break;
    case sizeof(int32_t):
      hipLaunchKernelGGL(_SliceKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          dimension_count, *starts, *steps, *input_strides, *output_strides,
          reinterpret_cast<const ToHipType<int32_t>::MappedType*>(input_data),
          reinterpret_cast<ToHipType<int32_t>::MappedType*>(output_data),
          (HIP_LONG)N);
      break;
    case sizeof(int64_t):
      hipLaunchKernelGGL(_SliceKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          dimension_count, *starts, *steps, *input_strides, *output_strides,
          reinterpret_cast<const ToHipType<int64_t>::MappedType*>(input_data),
          reinterpret_cast<ToHipType<int64_t>::MappedType*>(output_data),
          (HIP_LONG)N);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Slice operator");
  }

  return Status::OK();
}

}  // namespace hip
}  // namespace onnxruntime
