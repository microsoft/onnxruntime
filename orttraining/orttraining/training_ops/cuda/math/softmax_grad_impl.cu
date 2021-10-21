/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/* Modifications Copyright (c) Microsoft. */

// The code below is mostly copied from Pytorch PersistentSoftmax.cuh

#include "orttraining/training_ops/cuda/math/softmax_grad.h"

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/math/softmax_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_backward(output_t* gradInput, const input_t* grad, const input_t* output, int batch_size, int stride, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_backward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the batch
  int local_idx = threadIdx.x % WARP_SIZE;

  // the first element to process by the current thread
  int thread_offset = first_batch * stride + local_idx;
  grad += thread_offset;
  output += thread_offset;
  gradInput += thread_offset;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
  // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
  // the nested loops.
  // This should have no impact on performance because the loops are unrolled anyway.

  // load data from global memory
  acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
  acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
  acc_t grad_output_reg[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        grad_reg[i][it] = grad[i * element_count + it * WARP_SIZE];
        output_reg[i][it] = output[i * element_count + it * WARP_SIZE];
        grad_output_reg[i][it] = grad_reg[i][it] * output_reg[i][it];
      } else {
        grad_reg[i][it] = acc_t(0);
        output_reg[i][it] = acc_t(0);
        grad_output_reg[i][it] = acc_t(0);
      }
    }
  }

  acc_t sum[WARP_BATCH];
  if (!is_log_softmax) {
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] = grad_output_reg[i][0];
      #pragma unroll
      for (int it = 1; it < WARP_ITERATIONS; ++it) {
        sum[i] += grad_output_reg[i][it];
      }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);
  }
  else {
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] = grad_reg[i][0];
      #pragma unroll
      for (int it = 1; it < WARP_ITERATIONS; ++it) {
        sum[i] += grad_reg[i][it];
      }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);
  }

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // compute gradients
        if (is_log_softmax) {
          gradInput[i * element_count + it * WARP_SIZE] = (grad_reg[i][it] - std::exp(output_reg[i][it]) * sum[i]);
        } else {
          gradInput[i * element_count + it * WARP_SIZE] = (grad_reg[i][it] - sum[i] ) * output_reg[i][it];
        }
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_backward(cudaStream_t stream, output_t* grad_input, const input_t* grad, const input_t* output, int softmax_elements, int softmax_elements_stride, int batch_count) {
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
    int warp_size = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        softmax_warp_backward<input_t, output_t, acc_t, 0, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 1:  // 2
        softmax_warp_backward<input_t, output_t, acc_t, 1, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 2:  // 4
        softmax_warp_backward<input_t, output_t, acc_t, 2, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 3:  // 8
        softmax_warp_backward<input_t, output_t, acc_t, 3, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 4:  // 16
        softmax_warp_backward<input_t, output_t, acc_t, 4, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 5:  // 32
        softmax_warp_backward<input_t, output_t, acc_t, 5, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 6:  // 64
        softmax_warp_backward<input_t, output_t, acc_t, 6, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 7:  // 128
        softmax_warp_backward<input_t, output_t, acc_t, 7, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 8:  // 256
        softmax_warp_backward<input_t, output_t, acc_t, 8, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 9:  // 512
        softmax_warp_backward<input_t, output_t, acc_t, 9, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 10:  // 1024
        softmax_warp_backward<input_t, output_t, acc_t, 10, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        break;
      default:
        break;
    }
  }
}

#define SPECIALIZED_SOFTMAX_GRAD_IMPL(input_t, output_t, acc_t) \
template void dispatch_softmax_backward<input_t, output_t, acc_t, false>(cudaStream_t stream, input_t * grad_input, const output_t* grad, const output_t* output, int softmax_elements, int softmax_elements_stride, int batch_count); \
template void dispatch_softmax_backward<input_t, output_t, acc_t, true>(cudaStream_t stream, input_t * grad_input, const output_t* grad, const output_t* output, int softmax_elements, int softmax_elements_stride, int batch_count);

SPECIALIZED_SOFTMAX_GRAD_IMPL(float, float, float)
SPECIALIZED_SOFTMAX_GRAD_IMPL(half, half, float)
SPECIALIZED_SOFTMAX_GRAD_IMPL(double, double, double)

#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
SPECIALIZED_SOFTMAX_GRAD_IMPL(nv_bfloat16, nv_bfloat16, float)
#endif

}
}