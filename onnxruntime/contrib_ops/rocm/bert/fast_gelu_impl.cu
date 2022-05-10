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

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/shared_inc/rocm_call.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl.h"

using namespace onnxruntime::rocm;

namespace onnxruntime {
namespace contrib {
namespace rocm {

// constants for approximating the normal cdf
constexpr float A = 0.5;

constexpr float B = 0.7978845608028654;  // sqrt(2.0/M_PI)

constexpr float C = 0.035677408136300125;  // 0.044715 * sqrt(2.0/M_PI)

constexpr float one = 1.0;
constexpr float two = 2.0;

template <typename T, unsigned TPB>
__global__ void FastGeluKernel(const T a, const T b, const T c, const T oneT, const T twoT,
                               int input_length, int bias_length, const T* input, const T* bias, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

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

template <unsigned TPB>
__global__ void FastGeluKernel2(const half2 a, const half2 b, const half2 c, const half2 one2, const half2 two2,
                                int input_length, int bias_length, const half2* input, const half2* bias,
                                half2* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < input_length) {
    const half2 x = input[idx];
    const half2 in = (bias == nullptr) ? x : (x + bias[idx % bias_length]);

    // const half2 cdf = a + a * _Tanh(in * (c * in * in + b));
    const half2 u = two2 * in * (c * in * in + b);
    const half2 emu = h2exp(-u);
    const half2 cdf = a + a * (two2/(one2 + emu) - one2);

    output[idx] = in * cdf;
  }
}

template <unsigned TPB>
__global__ void FastGeluKernel4Bias(const half2 a, const half2 b, const half2 c, const half2 one2, const half2 two2,
                                    int input_length, int bias_length, const float2* input, const float2* bias,
                                    float2* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < input_length) {
    float2 input_vec = input[idx];
    float2 bias_vec = bias[idx % bias_length];
    float2 output_vec = output[idx];

    half2* input_half = reinterpret_cast<half2*>(&input_vec);
    half2* bias_half = reinterpret_cast<half2*>(&bias_vec);
    half2* output_half = reinterpret_cast<half2*>(&output_vec);

    half2 lo_data = input_half[0];
    half2 hi_data = input_half[1];
    half2 lo_bias = bias_half[0];
    half2 hi_bias = bias_half[1];

    lo_data += lo_bias;
    hi_data += hi_bias;

    const half2 lo_u = two2 * lo_data * (c * lo_data * lo_data + b);
    const half2 hi_u = two2 * hi_data * (c * hi_data * hi_data + b);
    const half2 lo_emu = h2exp(-lo_u);
    const half2 hi_emu = h2exp(-hi_u);
    const half2 lo_cdf = a + a * (two2/(one2 + lo_emu) - one2);
    const half2 hi_cdf = a + a * (two2/(one2 + hi_emu) - one2);

    output_half[0] = lo_data * lo_cdf;
    output_half[1] = hi_data * hi_cdf;

    output[idx] = output_vec;
  }
}

template <unsigned TPB>
__global__ void FastGeluKernel4(const half2 a, const half2 b, const half2 c, const half2 one2, const half2 two2,
                                int input_length, const float2* input, float2* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < input_length) {
    float2 input_vec = input[idx];
    float2 output_vec = output[idx];

    half2* input_half = reinterpret_cast<half2*>(&input_vec);
    half2* output_half = reinterpret_cast<half2*>(&output_vec);

    half2 lo_data = input_half[0];
    half2 hi_data = input_half[1];

    const half2 lo_u = two2 * lo_data * (c * lo_data * lo_data + b);
    const half2 hi_u = two2 * hi_data * (c * hi_data * hi_data + b);
    const half2 lo_emu = h2exp(-lo_u);
    const half2 hi_emu = h2exp(-hi_u);
    const half2 lo_cdf = a + a * (two2/(one2 + lo_emu) - one2);
    const half2 hi_cdf = a + a * (two2/(one2 + hi_emu) - one2);

    output_half[0] = lo_data * lo_cdf;
    output_half[1] = hi_data * hi_cdf;

    output[idx] = output_vec;
  }
}

template <>
bool LaunchFastGeluKernel(const hipDeviceProp_t& prop, hipStream_t stream, int input_length, int bias_length,
                          const float* input, const float* bias, float* output, bool /*use_half2*/) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel<float, blockSize>), dim3(gridSize), dim3(blockSize), 0,
                     stream, A, B, C, one, two, input_length, bias_length, input, bias, output);

  return HIP_CALL(hipPeekAtLastError());
}

template <>
bool LaunchFastGeluKernel(const hipDeviceProp_t& prop, hipStream_t stream, int input_length, int bias_length,
                          const half* input, const half* bias, half* output, bool use_half2) {
  constexpr int blockSize = 256;
  if (use_half2 && prop.major >= 7 && (0 == (bias_length % 4) || 0 == (bias_length & 1))) {
    const half2 A2 = __float2half2_rn(A);
    const half2 B2 = __float2half2_rn(B);
    const half2 C2 = __float2half2_rn(C);
    const half2 one2 = __float2half2_rn(one);
    const half2 two2 = __float2half2_rn(two);
    if (0 == (bias_length % 4)) {
      const int n = input_length / 4;
      const int gridSize = (n + blockSize - 1) / blockSize;
      const float2* input4 = reinterpret_cast<const float2*>(input);
      const float2* bias4 = reinterpret_cast<const float2*>(bias);
      float2* output4 = reinterpret_cast<float2*>(output);
      if (bias == nullptr)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel4<blockSize>), dim3(gridSize), dim3(blockSize), 0,
                           stream, A2, B2, C2, one2, two2, n, input4, output4);
      else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel4Bias<blockSize>), dim3(gridSize), dim3(blockSize), 0,
                           stream, A2, B2, C2, one2, two2, n, bias_length / 4, input4, bias4, output4);
    } else {
      const int n = input_length / 2;
      const int gridSize = (n + blockSize - 1) / blockSize;
      const half2* input2 = reinterpret_cast<const half2*>(input);
      const half2* bias2 = reinterpret_cast<const half2*>(bias);
      half2* output2 = reinterpret_cast<half2*>(output);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel2<blockSize>), dim3(gridSize), dim3(blockSize), 0,
                         stream, A2, B2, C2, one2, two2, n, bias_length / 2, input2, bias2, output2);
    }
  } else {
    const int gridSize = (input_length + blockSize - 1) / blockSize;
    const half oneT = half(one);
    const half twoT = half(two);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel<half, blockSize>), dim3(gridSize), dim3(blockSize), 0,
                       stream, A, B, C, oneT, twoT, input_length, bias_length, input, bias, output);
  }

  return HIP_CALL(hipPeekAtLastError());
}

template <>
bool LaunchFastGeluKernel(const hipDeviceProp_t& prop, hipStream_t stream, int input_length, int bias_length,
                          const BFloat16* input, const BFloat16* bias, BFloat16* output, bool /*use_half2*/) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  const BFloat16 oneT = BFloat16(one);
  const BFloat16 twoT = BFloat16(two);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(FastGeluKernel<BFloat16, blockSize>), dim3(gridSize), dim3(blockSize), 0,
                     stream, A, B, C, oneT, twoT, input_length, bias_length, input, bias, output);
  return HIP_CALL(hipPeekAtLastError());
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
