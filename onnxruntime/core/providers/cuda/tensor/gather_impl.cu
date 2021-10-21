// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "gather_impl.h"

namespace onnxruntime {
namespace cuda {

__host__ __device__ inline int64_t GetIndexValue(const void* index_data, size_t index_element_size, size_t offset) {
  switch (index_element_size) {
    case sizeof(int32_t):
      return *(reinterpret_cast<const int32_t*>(index_data) + offset);
      break;
    case sizeof(int64_t):
      return *(reinterpret_cast<const int64_t*>(index_data) + offset);
      break;
    default:
      break;
  }
  // What is a sensible thing to do here?
  assert(false);
  return std::numeric_limits<int64_t>::max();
}

template <typename T>
__global__ void _GatherKernel(
    const int64_t input_block_size,
    const int64_t indices_max,
    const fast_divmod output_block_size,
    const fast_divmod block_size,
    const void* indices_data,
    const size_t index_element_size,
    const T* input_data,
    T* output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  int input_block_index, block_offset;
  output_block_size.divmod(id, input_block_index, block_offset);
  int indices_index, offset;
  block_size.divmod(block_offset, indices_index, offset);
  int64_t idx = GetIndexValue(indices_data, index_element_size, indices_index);
  idx = idx < 0 ? idx + indices_max : idx;
  if (idx < 0 || idx >= indices_max) {
    output_data[id] = 0;
    return;
  }

  input_index = input_block_index * input_block_size + idx * block_size.d_ + offset;
  output_data[id] = input_data[input_index];
}

void GatherImpl(
    cudaStream_t stream,
    const int64_t input_block_size,
    const int64_t indices_max,
    const fast_divmod& output_block_size,
    const fast_divmod& block_size,
    const void* indices_data,
    size_t index_element_size,
    const void* input_data,
    size_t element_size,
    void* output_data,
    const size_t N) {

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  switch (element_size) {
    case sizeof(int8_t): {
      using CudaType = typename ToCudaType<int8_t>::MappedType;
      _GatherKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_block_size, indices_max, output_block_size, block_size, indices_data, index_element_size,
          reinterpret_cast<const CudaType*>(input_data), reinterpret_cast<CudaType*>(output_data), (CUDA_LONG)N);

    } break;
    case sizeof(int16_t): {
      using CudaType = typename ToCudaType<int16_t>::MappedType;
      _GatherKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_block_size, indices_max, output_block_size, block_size, indices_data, index_element_size,
          reinterpret_cast<const CudaType*>(input_data), reinterpret_cast<CudaType*>(output_data), (CUDA_LONG)N);

    } break;
    case sizeof(int32_t): {
      using CudaType = typename ToCudaType<int32_t>::MappedType;
      _GatherKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_block_size, indices_max, output_block_size, block_size, indices_data, index_element_size,
          reinterpret_cast<const CudaType*>(input_data), reinterpret_cast<CudaType*>(output_data), (CUDA_LONG)N);

    } break;
    case sizeof(int64_t): {
      using CudaType = typename ToCudaType<int64_t>::MappedType;
      _GatherKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_block_size, indices_max, output_block_size, block_size, indices_data, index_element_size,
          reinterpret_cast<const CudaType*>(input_data), reinterpret_cast<CudaType*>(output_data), (CUDA_LONG)N);

    } break;

    default:
      ORT_THROW("Unsupported element size by the Gather CUDA kernel");
  }
}

}  // namespace cuda
}  // namespace onnxruntime
