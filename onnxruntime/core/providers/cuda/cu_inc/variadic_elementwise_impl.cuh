#pragma once

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <
    typename T, typename Func,
    size_t max_input_batch_size, size_t num_elements_per_thread>
__global__ void VariadicElementWiseNoBroadcastInputBatchKernel(
    Func func,
    size_t N,
    TArray<const T*, max_input_batch_size> inputs,
    T* output) {
  const size_t base_idx = num_elements_per_thread * blockDim.x * blockIdx.x + threadIdx.x;

  T inputs_buffer[num_elements_per_thread][max_input_batch_size];

#pragma unroll
  for (size_t element_count = 0, element_idx = base_idx;
       element_count < num_elements_per_thread;
       ++element_count, element_idx += blockDim.x) {
    if (element_idx < N) {
#pragma unroll
      for (size_t input_batch_idx = 0; input_batch_idx < max_input_batch_size; ++input_batch_idx) {
        if (input_batch_idx < inputs.Size()) {
          inputs_buffer[element_count][input_batch_idx] = inputs[input_batch_idx][element_idx];
        }
      }
    }
  }

#pragma unroll
  for (size_t element_count = 0, element_idx = base_idx;
       element_count < num_elements_per_thread;
       ++element_count, element_idx += blockDim.x) {
    if (element_idx < N) {
      // first and second inputs
      output[element_idx] = func(
          inputs_buffer[element_count][0], inputs_buffer[element_count][1]);

      // remaining inputs
#pragma unroll
      for (size_t input_batch_idx = 2; input_batch_idx < max_input_batch_size; ++input_batch_idx) {
        if (input_batch_idx < inputs.Size()) {
          output[element_idx] = func(output[element_idx], inputs_buffer[element_count][input_batch_idx]);
        }
      }
    }
  }
}

// assumptions:
// - inputs.Size() > 1 && inputs.Size() <= max_input_batch_size
// - inputs and output have N elements
template <typename T, typename Func, size_t max_input_batch_size>
void VariadicElementWiseNoBroadcastInputBatchImpl(
    Func func,
    size_t N,
    TArray<const T*, max_input_batch_size> inputs,
    T* output) {
  const size_t elements_per_thread = 4;
  const size_t threads_per_block = GridDim::maxThreadsPerBlock;
  const size_t blocks_per_grid = CeilDiv(N, elements_per_thread * threads_per_block);
  VariadicElementWiseNoBroadcastInputBatchKernel<T, Func, max_input_batch_size, elements_per_thread>
      <<<blocks_per_grid, threads_per_block>>>(func, N, inputs, output);
}

}  // namespace cuda
}  // namespace onnxruntime
