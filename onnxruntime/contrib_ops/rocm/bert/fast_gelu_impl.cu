/*
 The implementation of this file is based on gelu plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Modifications: Add (bias) before Gelu is merged into this op to get better performance.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Add FastGeluKernelVec to leverage vectorized load/write
//                and modify FastGeluKernel to get better performance.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/shared_inc/rocm_call.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl.h"

using namespace onnxruntime::rocm;

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T, unsigned TPB>
__global__ void FastGeluKernel(int input_length, int bias_length, const T* input, const T* bias, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  // constants for approximating the normal cdf
  const T a = T(0.5f);
  const T b = T(0.7978845608028654f);  // sqrt(2.0/M_PI)
  const T c = T(0.035677408136300125f);  // 0.044715 * sqrt(2.0/M_PI)
  const T oneT = T(1.0f);
  const T twoT = T(2.0f);
  if (idx < input_length) {
    const T x = input[idx];
    const T in = (bias == nullptr) ? x : (x + bias[idx % bias_length]);

    // const T cdf = a + a * _Tanh(in * (c * in * in + b));
    const T u = twoT * in * (c * in * in + b);
    const T emu = __expf(-u);
    const T cdf = a + a * (twoT/(oneT + emu) - oneT);

    output[idx] = in * cdf;
  }
}

template <typename T, unsigned TPB, int ILP>
__global__ void FastGeluKernelVec(int input_length, int bias_length, const T* input, const T* bias,
                                  T* output) {
  using VecT = aligned_vector<T, ILP>;
  const T a = T(0.5f);
  const T b = T(0.7978845608028654f);
  const T c = T(0.035677408136300125f);
  const T oneT = T(1.0f);
  const T twoT = T(2.0f);

  const int idx = (blockIdx.x * TPB + threadIdx.x) * ILP;
  if (idx < input_length) {
    using VecT = aligned_vector<T, ILP>;
    T input_v[ILP];
    VecT* input_val = reinterpret_cast<VecT*>(&input_v);
    *input_val = *reinterpret_cast<const VecT*>(&input[idx]);
    T output_v[ILP];
    VecT* output_val = reinterpret_cast<VecT*>(&output_v);
    T bias_v[ILP];
    if (bias != nullptr) {
        VecT* bias_val = reinterpret_cast<VecT*>(&bias_v);
        *bias_val = *reinterpret_cast<const VecT*>(&bias[idx % bias_length]);
    }

    #pragma unroll
    for (int i = 0; i < ILP; i++) {
      const T x = (bias == nullptr) ? input_v[i] : input_v[i] + bias_v[i];
      const T u = twoT * x * (c * x * x + b);
      const T emu = __expf(-u);
      const T cdf = a + a * (twoT/(oneT + emu) - oneT);
      output_v[i] = x * cdf;
    }
    *(reinterpret_cast<VecT*>(&output[idx])) = *reinterpret_cast<VecT*>(&output_v[0]);
  }
}

template <>
bool LaunchFastGeluKernel(hipStream_t stream, int input_length, int bias_length,
                          const float* input, const float* bias, float* output, bool /*use_half2*/) {
  constexpr int block_size = 256;
  const int grid_size = (input_length + block_size - 1) / block_size;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel<float, block_size>), dim3(grid_size), dim3(block_size), 0,
                     stream, input_length, bias_length, input, bias, output);
  return HIP_CALL(hipPeekAtLastError());
}

template <>
bool LaunchFastGeluKernel(hipStream_t stream, int input_length, int bias_length,
                          const half* input, const half* bias, half* output, bool use_half2) {
  constexpr int block_size = 256;
  if (use_half2) {
      if (bias != nullptr) {
        if (0 == (bias_length % 8) && (input_length >= 3145728)) { // 3145728=8*128*3072
          const int grid_size = (input_length / 8 + block_size - 1) / block_size;
          hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernelVec<half, block_size, 8>), dim3(grid_size),
                                             dim3(block_size), 0, stream, input_length, bias_length,
                                             input, bias, output);
        } else if (0 == (bias_length % 4)) {
          const int grid_size = (input_length / 4 + block_size - 1) / block_size;
          hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernelVec<half, block_size, 4>), dim3(grid_size),
                                             dim3(block_size), 0, stream, input_length, bias_length,
                                             input, bias, output);
        } else if (0 == (bias_length % 2)) {
          const int grid_size = (input_length / 2 + block_size - 1) / block_size;
          hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernelVec<half, block_size, 2>), dim3(grid_size),
                                             dim3(block_size), 0, stream, input_length, bias_length,
                                             input, bias, output);
        } else {
          const int grid_size = (input_length + block_size - 1) / block_size;
          hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel<half, block_size>), dim3(grid_size),
                                             dim3(block_size), 0, stream, input_length, bias_length,
                                             input, bias, output);
        }
      } else {
        if (0 == (input_length % 8) && (input_length >= 3145728)) {  // 3145728=8*128*3072
          const int grid_size = (input_length / 8 + block_size - 1) / block_size;
          hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernelVec<half, block_size, 8>), dim3(grid_size),
                                             dim3(block_size), 0, stream, input_length, bias_length,
                                             input, bias, output);
        } else if (0 == (input_length % 4)) {
          const int grid_size = (input_length / 4 + block_size - 1) / block_size;
          hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernelVec<half, block_size, 4>), dim3(grid_size),
                                             dim3(block_size), 0, stream, input_length, bias_length,
                                             input, bias, output);
        } else if (0 == (input_length % 2)) {
          const int grid_size = (input_length / 2 + block_size - 1) / block_size;
          hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernelVec<half, block_size, 2>), dim3(grid_size),
                                             dim3(block_size), 0, stream, input_length, bias_length,
                                             input, bias, output);
        } else {
          const int grid_size = (input_length + block_size - 1) / block_size;
          hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel<half, block_size>), dim3(grid_size),
                                             dim3(block_size), 0, stream, input_length, bias_length,
                                             input, bias, output);
        }
      }
  } else {
    const int grid_size = (input_length + block_size - 1) / block_size;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel<half, block_size>), dim3(grid_size),
                                       dim3(block_size), 0, stream, input_length, bias_length,
                                       input, bias, output);
  }
  return HIP_CALL(hipPeekAtLastError());
}

template <>
bool LaunchFastGeluKernel(hipStream_t stream, int input_length, int bias_length,
                          const BFloat16* input, const BFloat16* bias, BFloat16* output, bool /*use_half2*/) {
  constexpr int block_size = 256;
  const int grid_size = (input_length + block_size - 1) / block_size;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel<BFloat16, block_size>), dim3(grid_size), dim3(block_size), 0,
                     stream, input_length, bias_length, input, bias, output);
  return HIP_CALL(hipPeekAtLastError());
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
