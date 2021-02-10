// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/scatter_nd_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/atomic/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _ScatterNDKernel(
    T* output_data,
    const size_t num_indices,
    const int64_t* indices_data,
    const int64_t last_index_dimension,
    const int64_t* element_counts_and_input_dims,
    const T* updates_data,
    const size_t num_updates_elements) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, num_indices);

  // Compute the base offset into the output data
  int64_t data_offset = 0;

  size_t indices_start = last_index_dimension * id;
  size_t indices_end = indices_start + last_index_dimension;
  for (size_t i = indices_start; i < indices_end; ++i) {
    int64_t index = indices_data[i];

    int64_t element_count_dim = element_counts_and_input_dims[i - indices_start];
    int64_t dim_value = element_counts_and_input_dims[i - indices_start + last_index_dimension];

    // Clamp the index if out of range
    // This would have been an error in the CPU kernel, but throwing in the CUDA EP
    // is hard. This is the approach taken by other frameworks for out of bound indices
    // in their corresponding GPU backends as well.
    if (index < 0)
      index = 0;

    else if (index >= dim_value)
      index = dim_value - 1;

    data_offset += (index * element_count_dim);
  }

  const T* updates_data_base = updates_data + num_updates_elements * id;
  T* output_data_base = output_data + data_offset;

  for (size_t i = 0; i < num_updates_elements; ++i) {
    output_data_base[i] = updates_data_base[i];
  }
}

Status ScatterNDImpl(
    cudaStream_t stream,
    void* output_data,
    const size_t element_size,
    const size_t num_indices,
    const int64_t* indices_data,
    const int64_t last_index_dimension,
    const int64_t* element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements) {
  if (num_indices == 0)
    return Status::OK();

  // Parallelize on number of indices
  int blocksPerGrid = static_cast<int>(ceil(static_cast<float>(num_indices) / GridDim::maxThreadsPerBlock));

  switch (element_size) {
    case sizeof(int8_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int8_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int8_t*>(updates_data),
          num_updates_elements);
      break;

    case sizeof(int16_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int16_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int16_t*>(updates_data),
          num_updates_elements);
      break;

    case sizeof(int32_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int32_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int32_t*>(updates_data),
          num_updates_elements);
      break;

    case sizeof(int64_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int64_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int64_t*>(updates_data),
          num_updates_elements);
      break;

    default:
      // Shouldn't hit this
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for ScatterND operator");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
