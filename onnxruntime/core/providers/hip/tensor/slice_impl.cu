// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/cu_inc/common.cuh"
#include "core/providers/hip/hip_common.h"
#include "core/providers/hip/tensor/slice_impl.h"

namespace onnxruntime {
namespace hip {

template <bool is_grad, typename T>
__global__ void _SliceKernel(const int32_t dimension_count,
                             const int64_t* starts,
                             const int64_t* steps,
                             const int64_t* input_strides,
                             const fast_divmod* output_strides,
                             const T* input_data,
                             T* output_data,
                             const HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  HIP_LONG input_index = 0;
  int div;
  int mod = id;
  int value = id;
  int dim = 0;
// #pragma unroll
//   for (; dim < starts.GetCapacity(); ++dim) {
//     if (dim >= dimension_count - 1) {
//       break;
//     }
  for (; dim < dimension_count - 1; ++dim) {
    output_strides[dim].divmod(value, div, mod);
    input_index += (starts[dim] + div * steps[dim]) * input_strides[dim];
    value = mod;
  }
  input_index += starts[dim] + mod * steps[dim];
  if (is_grad)
    output_data[input_index] = input_data[id];
  else
    output_data[id] = input_data[input_index];
}

Status SliceImpl(const size_t element_size,
                 const int32_t dimension_count,
                 const int64_t* starts,
                 const int64_t* steps,
                 const int64_t* input_strides,
                 const fast_divmod* output_strides,
                 const void* input_data,
                 void* output_data,
                 const size_t N) {
  return SliceImplEx<false>(element_size, dimension_count, starts, steps, input_strides, output_strides, input_data,
                            output_data, N);
}

Status SliceImplGrad(const size_t element_size,
                     const int32_t dimension_count,
                     const int64_t* starts,
                     const int64_t* steps,
                     const int64_t* input_strides,
                     const fast_divmod* output_strides,
                     const void* input_data,
                     void* output_data,
                     const size_t N) {
  return SliceImplEx<true>(element_size, dimension_count, starts, steps, input_strides, output_strides, input_data,
                           output_data, N);
}

template <bool is_grad>
Status SliceImplEx(const size_t element_size,
                   const int32_t dimension_count,
                   const int64_t* starts,
                   const int64_t* steps,
                   const int64_t* input_strides,
                   const fast_divmod* output_strides,
                   const void* input_data,
                   void* output_data,
                   const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  switch (element_size) {
    case sizeof(int8_t):
      hipLaunchKernelGGL(HIP_KERNEL_NAME(_SliceKernel<is_grad>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          dimension_count, starts, steps, input_strides, output_strides,
          reinterpret_cast<const ToHipType<int8_t>::MappedType*>(input_data),
          reinterpret_cast<ToHipType<int8_t>::MappedType*>(output_data),
          (HIP_LONG)N);
      break;
    case sizeof(int16_t):
      hipLaunchKernelGGL(HIP_KERNEL_NAME(_SliceKernel<is_grad>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          dimension_count, starts, steps, input_strides, output_strides,
          reinterpret_cast<const ToHipType<int16_t>::MappedType*>(input_data),
          reinterpret_cast<ToHipType<int16_t>::MappedType*>(output_data),
          (HIP_LONG)N);
      break;
    case sizeof(int32_t):
      hipLaunchKernelGGL(HIP_KERNEL_NAME(_SliceKernel<is_grad>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          dimension_count, starts, steps, input_strides, output_strides,
          reinterpret_cast<const ToHipType<int32_t>::MappedType*>(input_data),
          reinterpret_cast<ToHipType<int32_t>::MappedType*>(output_data),
          (HIP_LONG)N);
      break;
    case sizeof(int64_t):
      hipLaunchKernelGGL(HIP_KERNEL_NAME(_SliceKernel<is_grad>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          dimension_count, starts, steps, input_strides, output_strides,
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
