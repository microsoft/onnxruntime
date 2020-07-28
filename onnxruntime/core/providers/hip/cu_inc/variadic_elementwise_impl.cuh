#include "hip/hip_runtime.h"
#pragma once

#include "core/providers/hip/cu_inc/common.cuh"
#include "core/providers/hip/shared_inc/hip_utils.h"

namespace onnxruntime {
namespace hip {

template <
    typename T, typename Func,
    int32_t max_input_batch_size, int32_t num_elements_per_thread>
__global__ void VariadicElementWiseNoBroadcastInputBatchKernel(
    Func func,
    size_t N,
    TArray<const T*, max_input_batch_size> inputs,
    T* output) {
  const size_t base_idx = num_elements_per_thread * blockDim.x * blockIdx.x + threadIdx.x;

  T inputs_buffer[num_elements_per_thread][max_input_batch_size];

  int32_t element_count;
  size_t element_idx;

#pragma unroll
  for (element_count = 0, element_idx = base_idx;
       element_count < num_elements_per_thread;
       ++element_count, element_idx += blockDim.x) {
    if (element_idx < N) {
#pragma unroll
      for (int32_t input_batch_idx = 0; input_batch_idx < max_input_batch_size; ++input_batch_idx) {
        if (input_batch_idx < inputs.Size()) {
          inputs_buffer[element_count][input_batch_idx] = inputs[input_batch_idx][element_idx];
        }
      }
    }
  }

#pragma unroll
  for (element_count = 0, element_idx = base_idx;
       element_count < num_elements_per_thread;
       ++element_count, element_idx += blockDim.x) {
    if (element_idx < N) {
      // first and second inputs
      T output_value = func(
          inputs_buffer[element_count][0], inputs_buffer[element_count][1]);

      // remaining inputs
#pragma unroll
      for (int32_t input_batch_idx = 2; input_batch_idx < max_input_batch_size; ++input_batch_idx) {
        if (input_batch_idx < inputs.Size()) {
          output_value = func(output_value, inputs_buffer[element_count][input_batch_idx]);
        }
      }

      output[element_idx] = output_value;
    }
  }
}

// assumptions:
// - inputs.Size() > 1 && inputs.Size() <= max_input_batch_size
// - inputs and output have N elements
template <typename T, typename Func, int32_t max_input_batch_size>
void VariadicElementWiseNoBroadcastInputBatchImpl(
    Func func,
    size_t N,
    TArray<const T*, max_input_batch_size> inputs,
    T* output) {
  constexpr int32_t elements_per_thread = GridDim::maxElementsPerThread;
  constexpr int32_t threads_per_block = GridDim::maxThreadsPerBlock;
  const int32_t blocks_per_grid = CeilDiv(N, elements_per_thread * threads_per_block);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(VariadicElementWiseNoBroadcastInputBatchKernel<T, Func, max_input_batch_size, elements_per_thread>), dim3(blocks_per_grid), dim3(threads_per_block), 0, 0, func, N, inputs, output);
}

}  // namespace hip
}  // namespace onnxruntime
