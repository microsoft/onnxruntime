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

#include "core/providers/cuda/cu_inc/common.cuh"
#include "softmax_warpwise_impl.cuh"
#include "softmax_blockwise_impl.cuh"
#include "softmax.h"

#include <limits>

namespace onnxruntime {
namespace cuda {

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
Status dispatch_warpwise_softmax_forward(Stream* ort_stream, output_t* dst, const input_t* src, int softmax_elements,
                                         int softmax_elements_stride, int batch_count) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  if (softmax_elements == 0) {
    return Status::OK();
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = (next_power_of_two < GPU_WARP_SIZE_HOST) ? next_power_of_two : GPU_WARP_SIZE_HOST;
    int threads_per_block, shared_memory_size;
    // there are 2 options to save one row of the input matrix: register or shared memory
    // when the number of elements is small, we use register; otherwise, we use shared memory;
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
    if (log2_elements <= 10){
      // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
      // use 128 threads per block to maximimize gpu utilization
      threads_per_block = 128;
      shared_memory_size = 0;
    } else{
      // setting the number of threads per block to 32 will make index offset calculations easier,
      // under this setting, the cuda block number will be equal to batch size.
      threads_per_block = 32;
      // use shared memory to contain one row of elements
      // TODO: one more optimization can be done here: we actually not need to save next_power_of_two elements, we can just save the valid elements
      shared_memory_size = next_power_of_two * sizeof(input_t);
    }
    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {

#define LAUNCH_KERNEL(kernel_name, log2_elements_value)                                                                  \
  kernel_name<input_t, output_t, acc_t, log2_elements_value, is_log_softmax>                                             \
    <<<blocks, threads, shared_memory_size, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);

#define CASE_LOG2_ELEMENTS(log2_elements_value)                                                                          \
  case log2_elements_value: {                                                                                            \
    if constexpr (log2_elements_value <= 10) {                                                                                     \
      LAUNCH_KERNEL(softmax_warp_forward, log2_elements_value)                                                           \
    } else {                                                                                                             \
      LAUNCH_KERNEL(softmax_warp_forward_resource_efficient, log2_elements_value)                                        \
    }                                                                                                                    \
  } break

        CASE_LOG2_ELEMENTS(0);
        CASE_LOG2_ELEMENTS(1);
        CASE_LOG2_ELEMENTS(2);
        CASE_LOG2_ELEMENTS(3);
        CASE_LOG2_ELEMENTS(4);
        CASE_LOG2_ELEMENTS(5);
        CASE_LOG2_ELEMENTS(6);
        CASE_LOG2_ELEMENTS(7);
        CASE_LOG2_ELEMENTS(8);
        CASE_LOG2_ELEMENTS(9);
        CASE_LOG2_ELEMENTS(10);
        CASE_LOG2_ELEMENTS(11); // start to use softmax_warp_forward_resource_efficient instead of softmax_warp_forward for better performance
#undef LAUNCH_KERNEL
#undef CASE_LOG2_ELEMENTS
    } // switch
  } // else
  return CUDA_CALL(cudaGetLastError());
}

#define SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(input_t, output_t, acc_t)                                               \
  template Status dispatch_warpwise_softmax_forward<input_t, output_t, acc_t, false>(Stream* ort_stream,          \
                                                                                     output_t * dst,              \
                                                                                     const input_t* src,          \
                                                                                     int softmax_elements,        \
                                                                                     int softmax_elements_stride, \
                                                                                     int batch_count);            \
  template Status dispatch_warpwise_softmax_forward<input_t, output_t, acc_t, true>(Stream* ort_stream,           \
                                                                                    output_t * dst,               \
                                                                                    const input_t* src,           \
                                                                                    int softmax_elements,         \
                                                                                    int softmax_elements_stride,  \
                                                                                    int batch_count);

SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(float, float, float)
SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(half, half, float)
SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(half, float, float)
SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(double, double, double)
SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(BFloat16, BFloat16, float)

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
Status dispatch_blockwise_softmax_forward(Stream* ort_stream, output_t* output, const input_t* input, int softmax_elements,
                                          int input_stride, int output_stride, int batch_count) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  dim3 grid(batch_count);
  constexpr int ILP = sizeof(float4) / sizeof(input_t);
  dim3 block = SoftMax_getBlockSize(ILP, softmax_elements);
  if (is_log_softmax) {
    softmax_block_forward<ILP, input_t, acc_t, output_t, LogSoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(acc_t), stream>>>(output, const_cast<input_t*>(input),
                                                           softmax_elements, input_stride, output_stride);
  } else {
    softmax_block_forward<ILP, input_t, acc_t, output_t, SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(acc_t), stream>>>(output, const_cast<input_t*>(input),
                                                           softmax_elements, input_stride, output_stride);
  }
  return CUDA_CALL(cudaGetLastError());
}

#define SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(input_t, output_t, acc_t)                    \
  template Status dispatch_blockwise_softmax_forward<input_t, output_t, acc_t, false>(  \
      Stream* ort_stream, output_t * output, const input_t* src, int softmax_elements,  \
      int input_stride, int output_stride, int batch_count);                            \
  template Status dispatch_blockwise_softmax_forward<input_t, output_t, acc_t, true>(   \
      Stream* ort_stream, output_t * output, const input_t* src, int softmax_elements,  \
      int input_stride, int output_stride, int batch_count);

SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(float, float, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(half, half, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(half, float, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(double, double, double)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(BFloat16, BFloat16, float)
}  // namespace cuda
}  // namespace onnxruntime
