/*
 The implementation of this file is based on skipLayerNorm plugin in TensorRT demo:
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

#include "hip/hip_runtime.h"
#include "layer_norm.cuh"
#include "skip_layer_norm_impl.h"
#include <hip/hip_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernel(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma,
    const T epsilon, T* output) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;
  extern __shared__ unsigned char shared_memory_[];
  T* shared_memory = reinterpret_cast<T*>(shared_memory_);

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  hipcub::KeyValuePair<T, T> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;

  // if (threadIdx.x >= TPB) return;
  shared_memory[threadIdx.x] = input[idx];
  shared_memory[threadIdx.x + TPB] = skip[idx];

  __syncthreads();

  if (threadIdx.x < ld) {
    val = shared_memory[threadIdx.x] + shared_memory[threadIdx.x + TPB];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, hipcub::KeyValuePair<T, T>(rldval, rldval * val));
  }

  LayerNormSmall<T, TPB>(val, thread_data, ld, idx, beta, gamma, epsilon, output);
}

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernelBias(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, const T* bias,
    const T epsilon, T* output) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;
  extern __shared__ unsigned char shared_memory_[];
  T* shared_memory = reinterpret_cast<T*>(shared_memory_);

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  hipcub::KeyValuePair<T, T> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;

  // if (threadIdx.x >= TPB) return; // redundant
  shared_memory[threadIdx.x] = input[idx];
  shared_memory[threadIdx.x + TPB] = skip[idx];
  shared_memory[threadIdx.x + 2 * TPB] = bias[threadIdx.x];
  __syncthreads();

  if (threadIdx.x < ld) {
    val = shared_memory[threadIdx.x] + shared_memory[threadIdx.x + TPB] + shared_memory[threadIdx.x + 2 * TPB];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, hipcub::KeyValuePair<T, T>(rldval, rldval * val));
  }

  LayerNormSmall<T, TPB>(val, thread_data, ld, idx, beta, gamma, epsilon, output);
}

template <typename T>
bool ComputeSkipLayerNorm(
    hipStream_t stream, const int ld, const int n, const T* input, const T* skip,
    const T* beta, const T* gamma, const T* bias, const T epsilon, T* output) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  const int grid_size = n / ld;  // hidden_size: ld,  element_count: n;

  if (ld <= 32) {
    constexpr int block_size = 32;
    if (bias == nullptr) {
      unsigned int shmem = block_size * sizeof(T) * 2;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(SkipLayerNormKernel<T, block_size>), dim3(grid_size), dim3(block_size), shmem,
                     stream, ld, input, skip, beta, gamma, epsilon, output);
    } else {
      unsigned int shmem = block_size * sizeof(T) * 3;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(SkipLayerNormKernelBias<T, block_size>), dim3(grid_size), dim3(block_size), shmem,
                     stream, ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  } else if (ld <= 128) {
    constexpr int block_size = 128;
    if (bias == nullptr) {
      unsigned int shmem = block_size * sizeof(T) * 2;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(SkipLayerNormKernel<T, block_size>), dim3(grid_size), dim3(block_size), shmem,
                     stream, ld, input, skip, beta, gamma, epsilon, output);
    } else {
      unsigned int shmem = block_size * sizeof(T) * 3;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(SkipLayerNormKernelBias<T, block_size>), dim3(grid_size), dim3(block_size), shmem,
                     stream, ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  } else if (ld == 384) {
    constexpr int block_size = 384;
    if (bias == nullptr) {
      unsigned int shmem = block_size * sizeof(T) * 2;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(SkipLayerNormKernel<T, block_size>), dim3(grid_size), dim3(block_size), shmem,
                     stream, ld, input, skip, beta, gamma, epsilon, output);
    } else {
      unsigned int shmem = block_size * sizeof(T) * 3;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(SkipLayerNormKernelBias<T, block_size>), dim3(grid_size), dim3(block_size), shmem,
                     stream, ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  } else {
    constexpr int block_size = 256;
    if (bias == nullptr) {
      unsigned int shmem = block_size * sizeof(T) * 2;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(SkipLayerNormKernel<T, block_size>), dim3(grid_size), dim3(block_size), shmem,
                     stream, ld, input, skip, beta, gamma, epsilon, output);
    } else {
      unsigned int shmem = block_size * sizeof(T) * 3;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(SkipLayerNormKernelBias<T, block_size>), dim3(grid_size), dim3(block_size), shmem,
                     stream, ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  }
  return HIP_CALL(hipPeekAtLastError());
}

bool LaunchSkipLayerNormKernel(
    hipStream_t stream,
    void* output,
    const void* input,
    const void* skip,
    const void* gamma,
    const void* beta,
    const void* bias,
    float epsilon,
    int hidden_size,
    int element_count,
    size_t element_size) {
  if (element_size == 2) {
    return ComputeSkipLayerNorm(
        stream,
        hidden_size,
        element_count,
        reinterpret_cast<const half*>(input),
        reinterpret_cast<const half*>(skip),
        reinterpret_cast<const half*>(beta),
        reinterpret_cast<const half*>(gamma),
        reinterpret_cast<const half*>(bias),
        __float2half_rn(epsilon),
        reinterpret_cast<half*>(output));
  } else {
    return ComputeSkipLayerNorm(
        stream,
        hidden_size,
        element_count,
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(skip),
        reinterpret_cast<const float*>(beta),
        reinterpret_cast<const float*>(gamma),
        reinterpret_cast<const float*>(bias),
        epsilon,
        reinterpret_cast<float*>(output));
  }
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
