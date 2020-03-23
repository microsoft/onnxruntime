// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/cu_inc/common.cuh"
#include "gather_impl.h"

namespace onnxruntime {
namespace hip {

template <typename T, typename Tin>
__global__ void _GatherKernel(
    const int64_t input_block_size,
    const int64_t indices_max,
    const fast_divmod output_block_size,
    const fast_divmod block_size,
    const Tin* indices_data,
    const T* input_data,
    T* output_data,
    const HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  HIP_LONG input_index = 0;
  int input_block_index, block_offset;
  output_block_size.divmod(id, input_block_index, block_offset);
  int indices_index, offset;
  block_size.divmod(block_offset, indices_index, offset);
  int64_t idx = indices_data[indices_index];
  idx = idx < 0 ? idx + indices_max : idx;
  if (idx < 0 || idx >= indices_max) {
    output_data[id] = 0;
    return;
  }

  input_index = input_block_index * input_block_size + idx * block_size.d_ + offset;
  output_data[id] = input_data[input_index];
}

template <typename T, typename Tin>
void GatherImpl(
    const int64_t input_block_size,
    const int64_t indices_max,
    const fast_divmod* output_block_size,
    const fast_divmod* block_size,
    const Tin* indices_data,
    const T* input_data,
    T* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  hipLaunchKernelGGL(_GatherKernel<T, Tin>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
      input_block_size, indices_max, *output_block_size, *block_size, indices_data, input_data, output_data, (HIP_LONG)N);
}

#define SPECIALIZED_IMPL(T)                                                                                               \
  template void GatherImpl<T, int32_t>(const int64_t input_block_size, const int64_t indices_max,                         \
                                       const fast_divmod* output_block_size, const fast_divmod* block_size,               \
                                       const int32_t* indices_data, const T* input_data, T* output_data, const size_t N); \
  template void GatherImpl<T, int64_t>(const int64_t input_block_size, const int64_t indices_max,                         \
                                       const fast_divmod* output_block_size, const fast_divmod* block_size,               \
                                       const int64_t* indices_data, const T* input_data, T* output_data, const size_t N);

SPECIALIZED_IMPL(int8_t)
SPECIALIZED_IMPL(int16_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(uint8_t)
SPECIALIZED_IMPL(uint16_t)
SPECIALIZED_IMPL(uint32_t)
SPECIALIZED_IMPL(uint64_t)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(bool)

}  // namespace hip
}  // namespace onnxruntime
