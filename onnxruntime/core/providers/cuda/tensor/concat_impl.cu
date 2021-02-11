// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "concat_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _ConcatKernel(const fast_divmod block_size_including_axis_dim_div,
                              const fast_divmod block_size_inside_axis_dim_div,
                              const int64_t* concat_sizes,
                              const int64_t* concat_sizes_range,
                              const int64_t* axis_dimension_input_output_mapping,
                              T* output_data,
                              const void** input_ptr,
                              const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_pos = 0;

  int outter_block_index = 0;
  int block_index = 0;
  int offset = 0;

  block_size_including_axis_dim_div.divmod(id, outter_block_index, offset);
  block_size_inside_axis_dim_div.divmod(offset, block_index, offset);

  int input_index = axis_dimension_input_output_mapping[block_index];
  int64_t range_left = (input_index == 0) ? 0 : concat_sizes_range[input_index - 1];
  int block_offset = block_index - range_left;

  input_pos = (outter_block_index * concat_sizes[input_index] + block_offset) *
               block_size_inside_axis_dim_div.d_ +
               offset;

  output_data[id] = reinterpret_cast<const T*>(input_ptr[input_index])[input_pos];
}

Status ConcatImpl(cudaStream_t stream,
                  const size_t element_bytes,
                  const int block_size_including_axis_dim,
                  const int block_size_inside_axis_dim,
                  const int64_t* concat_sizes,
                  const int64_t* concat_sizes_range,
                  const int64_t* axis_dimension_input_output_mapping,
                  void* output_data,
                  const void** input_ptr,
                  const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  fast_divmod block_size_including_axis_dim_div = fast_divmod(block_size_including_axis_dim);
  fast_divmod block_size_inside_axis_dim_div = fast_divmod(block_size_inside_axis_dim);

  switch (element_bytes) {
    case sizeof(int8_t):
      _ConcatKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          block_size_including_axis_dim_div, block_size_inside_axis_dim_div,
          concat_sizes, concat_sizes_range, axis_dimension_input_output_mapping,
          reinterpret_cast<int8_t*>(output_data),
          input_ptr,
          (CUDA_LONG)N);
      break;
    case sizeof(int16_t):
      _ConcatKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          block_size_including_axis_dim_div, block_size_inside_axis_dim_div,
          concat_sizes, concat_sizes_range, axis_dimension_input_output_mapping,
          reinterpret_cast<int16_t*>(output_data),
          input_ptr,
          (CUDA_LONG)N);
      break;
    case sizeof(int32_t):
      _ConcatKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          block_size_including_axis_dim_div, block_size_inside_axis_dim_div,
          concat_sizes, concat_sizes_range, axis_dimension_input_output_mapping,
          reinterpret_cast<int32_t*>(output_data),
          input_ptr,
          (CUDA_LONG)N);
      break;
    case sizeof(int64_t):
      _ConcatKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          block_size_including_axis_dim_div, block_size_inside_axis_dim_div,
          concat_sizes, concat_sizes_range, axis_dimension_input_output_mapping,
          reinterpret_cast<int64_t*>(output_data),
          input_ptr,
          (CUDA_LONG)N);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Concat operator");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
