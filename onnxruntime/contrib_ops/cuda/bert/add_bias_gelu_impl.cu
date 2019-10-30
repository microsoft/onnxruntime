/*
 The implementation of this file is based on act_bias_act kernel in FasterTransformer:
 https://github.com/NVIDIA/DeepLearningExamples/FasterTransformer/
 
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "add_bias_gelu_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// constants for approximating the normal cdf
constexpr float A = 0.5;
constexpr float B = 0.7978845608028654;  // sqrt(2.0/M_PI)
constexpr float C = 0.035677408136300125;  // 0.044715 * sqrt(2.0/M_PI)

__device__ inline float tanh(const float& x) {
  return tanhf(x);
}

__device__ inline half tanh(const half& x) {
  const float tmp = tanhf(__half2float(x));
  return __float2half(tmp);
}

__device__ inline half2 tanh(const half2& x) {
  // at the moment, there is no half2 tanh builtin
  float2 tmp = (__half22float2(x));
  tmp.x = tanhf(tmp.x);
  tmp.y = tanhf(tmp.y);
  return __float22half2_rn(tmp);
}

template <typename T>
__inline__ __device__ T Gelu(T x, T a, T b, T c) {
  return x * (a + a * tanh(x * (c * x * x + b)));
}

template <typename T>
__global__ void AddBiasGelu(const T* input, const T* bias, T* out, int m, int n, T a, T b, T c) {
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for (int i = 0; i < ite; ++i) {
#if __CUDA_ARCH__ >= 350 || !defined(__CUDA_ARCH__)
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
#else
    reg_bias = bias[i * blockDim.x + tid];
#endif
    row_id = blockIdx.x;

    while (row_id < m) {
      val = input[tid + i * blockDim.x + row_id * n] + reg_bias;
      out[tid + i * blockDim.x + row_id * n] = Gelu<T>(val, a, b, c);
      row_id += gridDim.x;
    }
  }
}

template <>
bool LaunchAddBiasGeluKernel<half>(const half* input, const half* bias, half* output, int m, int n) {
  dim3 grid(m / 4);
  dim3 block(n / 4);
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  const half2 A2 = __floats2half2_rn(A, A);
  const half2 B2 = __floats2half2_rn(B, B);
  const half2 C2 = __floats2half2_rn(C, C);
  AddBiasGelu<half2><<<grid, block, 0>>>(reinterpret_cast<const half2*>(input), reinterpret_cast<const half2*>(bias), reinterpret_cast<half2*>(output), m, n / 2, A2, B2, C2);
#else
  AddBiasGelu<half><<<grid, block, 0>>>(input, bias, output, m, n, A, B, C);
#endif
  return CUDA_CALL(cudaPeekAtLastError());
}

template <>
bool LaunchAddBiasGeluKernel<float>(const float* input, const float* bias, float* output, int m, int n) {
  dim3 grid(m / 4);
  dim3 block(n / 4);
  AddBiasGelu<float><<<grid, block, 0>>>(input, bias, output, m, n, A, B, C);
  return CUDA_CALL(cudaPeekAtLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
