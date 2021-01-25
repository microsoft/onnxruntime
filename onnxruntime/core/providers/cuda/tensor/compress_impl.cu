// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "compress_impl.h"
//TODO:fix the warnings
#ifdef _MSC_VER
#pragma warning(disable : 4244)
#endif
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace onnxruntime {
namespace cuda {

void PrefixSumImpl(const int8_t* condition_data,
                   int32_t* condition_cumulative_sum,
                   const size_t length) {
  thrust::inclusive_scan(thrust::device, condition_data, condition_data + length, condition_cumulative_sum);
}

template <typename T>
__global__ void _CompressKernel(const int32_t valid_condition_length,
                                const fast_divmod axis_right_stride_div,
                                const fast_divmod input_axis_included_stride_div,
                                const int32_t output_axis_included_stride,
                                const int32_t* condition_cumulative_sum,
                                const bool* condition_data,
                                const T* input_data,
                                T* output_data,
                                const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG output_index = 0;

  int div, mod;
  input_axis_included_stride_div.divmod(id, div, mod);
  output_index = output_axis_included_stride * div;
  axis_right_stride_div.divmod(mod, div, mod);

  if (div < valid_condition_length && condition_data[div]) {
    output_index += (condition_cumulative_sum[div] - 1) * axis_right_stride_div.d_ + mod;
    output_data[output_index] = input_data[id];
  }
}

Status CompressImpl(const size_t element_bytes,
                    const int32_t valid_condition_length,
                    const int32_t axis_right_stride,
                    const int32_t input_axis_dim_length,
                    const int32_t output_axis_dim_length,
                    const int32_t* condition_cumulative_sum,
                    const bool* condition_data,
                    const void* input_data,
                    void* output_data,
                    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  fast_divmod axis_right_stride_div(axis_right_stride);
  fast_divmod input_axis_included_stride_div(axis_right_stride * input_axis_dim_length);
  int output_axis_included_stride = axis_right_stride * output_axis_dim_length;

  switch (element_bytes) {
    case sizeof(int8_t):
      _CompressKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          valid_condition_length,
          axis_right_stride_div,
          input_axis_included_stride_div,
          output_axis_included_stride,
          condition_cumulative_sum,
          condition_data,
          reinterpret_cast<const ToCudaType<int8_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int8_t>::MappedType*>(output_data),
          (CUDA_LONG)N);
      break;
    case sizeof(int16_t):
      _CompressKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          valid_condition_length,
          axis_right_stride_div,
          input_axis_included_stride_div,
          output_axis_included_stride,
          condition_cumulative_sum,
          condition_data,
          reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data),
          (CUDA_LONG)N);
      break;
    case sizeof(int32_t):
      _CompressKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          valid_condition_length,
          axis_right_stride_div,
          input_axis_included_stride_div,
          output_axis_included_stride,
          condition_cumulative_sum,
          condition_data,
          reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data),
          (CUDA_LONG)N);
      break;
    case sizeof(int64_t):
      _CompressKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          valid_condition_length,
          axis_right_stride_div,
          input_axis_included_stride_div,
          output_axis_included_stride,
          condition_cumulative_sum,
          condition_data,
          reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data),
          (CUDA_LONG)N);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Compress operator");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
