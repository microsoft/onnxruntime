// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "slice_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _SliceKernel(const int32_t dimension_count,
                             const int64_t* starts,
                             const int64_t* input_strides,
                             const fast_divmod* div_strides,
                             const T* input_data,
                             T* output_data,
                             const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  int div;
  int mod = id;
  int value = id;
  int dim_idx = 0;
  for (; dim_idx < dimension_count - 1; ++dim_idx) {
    div_strides[dim_idx].divmod(value, div, mod);
    input_index += (starts[dim_idx] + div) * input_strides[dim_idx];
    value = mod;
  }
  input_index += starts[dim_idx] + mod;
  output_data[id] = input_data[input_index];
}

Status SliceImpl(const size_t element_size,
               const int32_t dimension_count,
               const int64_t* starts,
               const int64_t* input_strides,
               const fast_divmod* output_div_strides,
               const void* input_data,
               void* output_data,
               const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  switch (element_size) {
    case sizeof(int8_t):
      _SliceKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, 0>>>(
          dimension_count, starts, input_strides, output_div_strides,
          reinterpret_cast<const ToCudaType<int8_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int8_t>::MappedType*>(output_data),
          (CUDA_LONG)N);
      break;
    case sizeof(int16_t):
      _SliceKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, 0>>>(
          dimension_count, starts, input_strides, output_div_strides,
          reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data),
          (CUDA_LONG)N);
      break;
    case sizeof(int32_t):
      _SliceKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, 0>>>(
          dimension_count, starts, input_strides, output_div_strides,
          reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data),
          (CUDA_LONG)N);
      break;
    case sizeof(int64_t):
      _SliceKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, 0>>>(
          dimension_count, starts, input_strides, output_div_strides,
          reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data),
          (CUDA_LONG)N);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Slice operator");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
