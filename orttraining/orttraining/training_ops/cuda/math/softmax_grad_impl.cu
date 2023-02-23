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

#include "orttraining/training_ops/cuda/math/softmax_grad_impl.h"

#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/math/softmax_common.h"
#include "core/providers/cuda/math/softmax_warpwise_impl.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_backward(output_t* gradInput, const input_t* grad, const input_t* output,
                                      int element_count, int batch_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method
  // warp_softmax_backward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
#ifdef USE_ROCM
  constexpr int WARP_BATCH = 1;
#else
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
#endif

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_count might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_count - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the batch
  int local_idx = threadIdx.x % WARP_SIZE;

  // the first element to process by the current thread
  int thread_offset = first_batch * element_count + local_idx;
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

template <typename T>
Status SoftmaxGradImpl(cudaStream_t stream, cudnnHandle_t cudnn_handle, T* input_grad, const T* output_grad,
                       const T* softmax_output, int element_count, int batch_count, bool is_log_softmax) {
  if (element_count == 0) return Status::OK();
  if (element_count <= 1024 && element_count * sizeof(T) <= 4096) {
    typedef AccumulationType_t<T> AccT;
    int log2_elements = log2_ceil(element_count);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
    int warp_size = std::min(next_power_of_two, GPU_WARP_SIZE_HOST);

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
#ifdef USE_ROCM
    int batches_per_warp = 1;
    constexpr int threads_per_block = 256;
#else
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int threads_per_block = 128;
#endif

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
#define CASE_LOG2_ELEMENTS(log2_elements_value)                                                                  \
  case log2_elements_value: {                                                                                    \
    if (is_log_softmax) {                                                                                        \
      softmax_warp_backward<T, T, AccT, log2_elements_value, true>                                               \
          <<<blocks, threads, 0, stream>>>(input_grad, output_grad, softmax_output, element_count, batch_count); \
    } else {                                                                                                     \
      softmax_warp_backward<T, T, AccT, log2_elements_value, false>                                              \
          <<<blocks, threads, 0, stream>>>(input_grad, output_grad, softmax_output, element_count, batch_count); \
    }                                                                                                            \
  } break
      CASE_LOG2_ELEMENTS(0);   // 1
      CASE_LOG2_ELEMENTS(1);   // 2
      CASE_LOG2_ELEMENTS(2);   // 4
      CASE_LOG2_ELEMENTS(3);   // 8
      CASE_LOG2_ELEMENTS(4);   // 16
      CASE_LOG2_ELEMENTS(5);   // 32
      CASE_LOG2_ELEMENTS(6);   // 64
      CASE_LOG2_ELEMENTS(7);   // 128
      CASE_LOG2_ELEMENTS(8);   // 256
      CASE_LOG2_ELEMENTS(9);   // 512
      CASE_LOG2_ELEMENTS(10);  // 1024
#undef CASE_LOG2_ELEMENTS
    }
    return Status::OK();
  }

  const int64_t dims[]{batch_count, 1, 1, element_count};  // cudnn expects 4D shape in NCHW format
  const auto alpha = Consts<T>::One;
  const auto beta = Consts<T>::Zero;
  CudnnTensor input_tensor, output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(dims, CudnnTensor::GetDataType<T>()));
  ORT_RETURN_IF_ERROR(output_tensor.Set(dims, CudnnTensor::GetDataType<T>()));
  return SoftmaxBackward(cudnn_handle, is_log_softmax, &alpha, input_tensor, softmax_output, output_grad, &beta,
                         output_tensor, input_grad);
}

#define SPECIALIZED_SOFTMAX_GRAD_IMPL(type)                                                                     \
  template Status SoftmaxGradImpl<type>(cudaStream_t stream, cudnnHandle_t cudnn_handle, type * input_grad,     \
                                        const type* output_grad, const type* softmax_output, int element_count, \
                                        int batch_count, bool is_log_softmax);

SPECIALIZED_SOFTMAX_GRAD_IMPL(float)
SPECIALIZED_SOFTMAX_GRAD_IMPL(half)
SPECIALIZED_SOFTMAX_GRAD_IMPL(BFloat16)
#ifdef USE_CUDA
SPECIALIZED_SOFTMAX_GRAD_IMPL(double)
#endif

#undef SPECIALIZED_SOFTMAX_GRAD_IMPL
}
}
