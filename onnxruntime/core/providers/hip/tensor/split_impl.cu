#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/cu_inc/common.cuh"
#include "core/providers/hip/hip_common.h"
#include "split_impl.h"

namespace onnxruntime {
namespace hip {

template <typename T>
__global__ void _SplitKernel(const fast_divmod block_size_including_axis_dim_div,
                             const fast_divmod block_size_inside_axis_dim_div,
                             const int64_t* split_sizes,
                             const int64_t* split_sizes_range,
                             const int64_t* axis_dimension_input_output_mapping,
                             const int num_outputs,
                             const T* input_data,
                             void** output_ptr,
                             const HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  HIP_LONG output_pos = 0;

  int outter_block_index = 0;
  int block_index = 0;
  int offset = 0;

  block_size_including_axis_dim_div.divmod(id, outter_block_index, offset);
  block_size_inside_axis_dim_div.divmod(offset, block_index, offset);

  int output_index = axis_dimension_input_output_mapping[block_index];
  int64_t range_left = (output_index == 0) ? 0 : split_sizes_range[output_index - 1];
  int block_offset = block_index - range_left;

  output_pos = (outter_block_index * split_sizes[output_index] + block_offset) * 
               block_size_inside_axis_dim_div.d_ +
               offset;

  reinterpret_cast<T*>(output_ptr[output_index])[output_pos] = input_data[id];
}

Status SplitImpl(const size_t element_size,
                 const int block_size_including_axis_dim,
                 const int block_size_inside_axis_dim,
                 const int64_t* split_sizes,
                 const int64_t* split_sizes_range,
                 const int64_t* axis_dimension_input_output_mapping,
                 const int num_outputs,
                 const void* input_data,
                 void** output_ptr,
                 const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  fast_divmod block_size_including_axis_dim_div = fast_divmod(block_size_including_axis_dim);
  fast_divmod block_size_inside_axis_dim_div = fast_divmod(block_size_inside_axis_dim);

  switch (element_size) {
    case sizeof(int8_t):
      hipLaunchKernelGGL(_SplitKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          block_size_including_axis_dim_div, block_size_inside_axis_dim_div,
          split_sizes, split_sizes_range, axis_dimension_input_output_mapping, num_outputs,
          reinterpret_cast<const ToHipType<int8_t>::MappedType*>(input_data),
          output_ptr,
          (HIP_LONG)N);
      break;
    case sizeof(int16_t):
      hipLaunchKernelGGL(_SplitKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          block_size_including_axis_dim_div, block_size_inside_axis_dim_div,
          split_sizes, split_sizes_range, axis_dimension_input_output_mapping, num_outputs,
          reinterpret_cast<const ToHipType<int16_t>::MappedType*>(input_data),
          output_ptr,
          (HIP_LONG)N);
      break;
    case sizeof(int32_t):
      hipLaunchKernelGGL(_SplitKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          block_size_including_axis_dim_div, block_size_inside_axis_dim_div,
          split_sizes, split_sizes_range, axis_dimension_input_output_mapping, num_outputs,
          reinterpret_cast<const ToHipType<int32_t>::MappedType*>(input_data),
          output_ptr,
          (HIP_LONG)N);
      break;
    case sizeof(int64_t):
      hipLaunchKernelGGL(_SplitKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          block_size_including_axis_dim_div, block_size_inside_axis_dim_div,
          split_sizes, split_sizes_range, axis_dimension_input_output_mapping, num_outputs,
          reinterpret_cast<const ToHipType<int64_t>::MappedType*>(input_data),
          output_ptr,
          (HIP_LONG)N);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Slice operator");
  }

  return Status::OK();
}

}  // namespace hip
}  // namespace onnxruntime