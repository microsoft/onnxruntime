#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/cu_inc/common.cuh"
#include "gather_impl.h"

namespace onnxruntime {
namespace hip {

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
    const HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  HIP_LONG input_index = 0;
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
      using HipType = typename ToHipType<int8_t>::MappedType;
      hipLaunchKernelGGL(_GatherKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          input_block_size, indices_max, output_block_size, block_size, indices_data, index_element_size,
          reinterpret_cast<const HipType*>(input_data), reinterpret_cast<HipType*>(output_data), (HIP_LONG)N);

    } break;
    case sizeof(int16_t): {
      using HipType = typename ToHipType<int16_t>::MappedType;
      hipLaunchKernelGGL(_GatherKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          input_block_size, indices_max, output_block_size, block_size, indices_data, index_element_size,
          reinterpret_cast<const HipType*>(input_data), reinterpret_cast<HipType*>(output_data), (HIP_LONG)N);

    } break;
    case sizeof(int32_t): {
      using HipType = typename ToHipType<int32_t>::MappedType;
      hipLaunchKernelGGL(_GatherKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          input_block_size, indices_max, output_block_size, block_size, indices_data, index_element_size,
          reinterpret_cast<const HipType*>(input_data), reinterpret_cast<HipType*>(output_data), (HIP_LONG)N);

    } break;
    case sizeof(int64_t): {
      using HipType = typename ToHipType<int64_t>::MappedType;
      hipLaunchKernelGGL(_GatherKernel, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          input_block_size, indices_max, output_block_size, block_size, indices_data, index_element_size,
          reinterpret_cast<const HipType*>(input_data), reinterpret_cast<HipType*>(output_data), (HIP_LONG)N);

    } break;

    default:
      ORT_THROW("Unsupported element size by the Gather HIP kernel");
  }
}

}  // namespace hip
}  // namespace onnxruntime
