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
#include "hip/hip_runtime.h"

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/math/softmax_warpwise_impl.cuh"
#include "core/providers/rocm/math/softmax_blockwise_impl.cuh"
#include "core/providers/rocm/math/softmax.h"

#include <limits>

namespace onnxruntime {
namespace rocm {

template <typename InputT, typename OutputT, typename AccT, bool IsLogSoftmax>
Status dispatch_warpwise_softmax_forward(hipStream_t stream, OutputT* dst, const InputT* src, int softmax_elements, int softmax_elements_stride, int batch_count) {
  if (softmax_elements == 0) {
    return Status::OK();
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = (next_power_of_two < GPU_WARP_SIZE_HOST) ? next_power_of_two : GPU_WARP_SIZE_HOST;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
    int batches_per_warp = 1;
    // use 256 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 256;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
#define LAUNCH_SOFTMAX_WARP_FORWARD(L2E)                                                         \
  case L2E:                                                                                      \
    softmax_warp_forward<InputT, OutputT, AccT, L2E, IsLogSoftmax>                               \
        <<<dim3(blocks), dim3(threads), 0, stream>>>(dst, src, batch_count,                      \
                                                     softmax_elements_stride, softmax_elements); \
    break;
      LAUNCH_SOFTMAX_WARP_FORWARD(0);   // 1
      LAUNCH_SOFTMAX_WARP_FORWARD(1);   // 2
      LAUNCH_SOFTMAX_WARP_FORWARD(2);   // 4
      LAUNCH_SOFTMAX_WARP_FORWARD(3);   // 8
      LAUNCH_SOFTMAX_WARP_FORWARD(4);   // 16
      LAUNCH_SOFTMAX_WARP_FORWARD(5);   // 32
      LAUNCH_SOFTMAX_WARP_FORWARD(6);   // 64
      LAUNCH_SOFTMAX_WARP_FORWARD(7);   // 128
      LAUNCH_SOFTMAX_WARP_FORWARD(8);   // 256
      LAUNCH_SOFTMAX_WARP_FORWARD(9);   // 512
      LAUNCH_SOFTMAX_WARP_FORWARD(10);  // 1024
      default:
        break;
    }
  }
  return HIP_CALL(hipGetLastError());
}

#define SPECIALIZED_SOFTMAX_IMPL(InputT, OutputT, AccT)                            \
  template Status dispatch_warpwise_softmax_forward<InputT, OutputT, AccT, false>( \
      hipStream_t stream, OutputT * dst, const InputT* src, int softmax_elements,  \
      int softmax_elements_stride, int batch_count);                               \
  template Status dispatch_warpwise_softmax_forward<InputT, OutputT, AccT, true>(  \
      hipStream_t stream, OutputT * dst, const InputT* src, int softmax_elements,  \
      int softmax_elements_stride, int batch_count);

SPECIALIZED_SOFTMAX_IMPL(float, float, float)
SPECIALIZED_SOFTMAX_IMPL(half, half, float)
SPECIALIZED_SOFTMAX_IMPL(half, float, float)
SPECIALIZED_SOFTMAX_IMPL(double, double, double)
SPECIALIZED_SOFTMAX_IMPL(BFloat16, BFloat16, float)

template <typename InputT, typename OutputT, typename AccT, bool IsLogSoftmax>
Status dispatch_blockwise_softmax_forward(hipStream_t stream, OutputT* output, const InputT* input, int softmax_elements,
                                          int input_stride, int output_stride, int batch_count) {
  dim3 grid(batch_count);
  constexpr int ILP = sizeof(float4) / sizeof(InputT);
  dim3 block = SoftMax_getBlockSize(ILP, softmax_elements);
  if (IsLogSoftmax) {
    softmax_block_forward<ILP, InputT, AccT, OutputT, LogSoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(AccT), stream>>>(output, const_cast<InputT*>(input),
                                                          softmax_elements, input_stride, output_stride);
  } else {
    softmax_block_forward<ILP, InputT, AccT, OutputT, SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(AccT), stream>>>(output, const_cast<InputT*>(input),
                                                          softmax_elements, input_stride, output_stride);
  }
  return HIP_CALL(hipGetLastError());
}

#define SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(InputT, OutputT, AccT)                      \
  template Status dispatch_blockwise_softmax_forward<InputT, OutputT, AccT, false>(    \
      hipStream_t stream, OutputT * output, const InputT* input, int softmax_elements, \
      int input_stride, int output_stride, int batch_count);                           \
  template Status dispatch_blockwise_softmax_forward<InputT, OutputT, AccT, true>(     \
      hipStream_t stream, OutputT * output, const InputT* input, int softmax_elements, \
      int input_stride, int output_stride, int batch_count);

SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(float, float, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(half, half, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(half, float, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(double, double, double)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(BFloat16, BFloat16, float)
}  // namespace rocm
}  // namespace onnxruntime
