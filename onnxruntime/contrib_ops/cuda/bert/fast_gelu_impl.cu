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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "fast_gelu_impl.h"

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

template <typename T, unsigned TPB>
__global__ void geluKernel(const T a, const T b, const T c, int input_length, int bias_length, const T* input, const T* bias, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < input_length) {
    const T x = input[idx];
    const T in = (bias == nullptr) ? x : (x + bias[idx % bias_length]);
    const T cdf = a + a * tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}

template<>
bool computeGelu<float>(cudaStream_t stream, int input_length, int bias_length, const float* input, const float* bias, float* output) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  geluKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, input_length, bias_length, input, bias, output);

  return CUDA_CALL(cudaPeekAtLastError());
}

template<>
bool computeGelu<half>(cudaStream_t stream, int input_length, int bias_length, const half* input, const half* bias, half* output) {
  const int blockSize = 256;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  if (0 == (bias_length & 1)) {
    const int n = input_length / 2;
    const int gridSize = (n + blockSize - 1) / blockSize;
    const half2 A2 = __floats2half2_rn(A, A);
    const half2 B2 = __floats2half2_rn(B, B);
    const half2 C2 = __floats2half2_rn(C, C);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    half2* output2 = reinterpret_cast<half2*>(output);
    geluKernel<half2, blockSize><<<gridSize, blockSize, 0, stream>>>(A2, B2, C2, n, bias_length / 2, input2, bias2, output2);
  } else 
#endif
  {
    const int gridSize = (input_length + blockSize - 1) / blockSize;
    geluKernel<half, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, input_length, bias_length, input, bias, output);
  }

  return CUDA_CALL(cudaPeekAtLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
