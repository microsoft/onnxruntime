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

#include "contrib_ops/cuda/bert/layer_norm.cuh"
#include "contrib_ops/cuda/bert/skip_layer_norm_impl.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr float one = 1.0;

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, const T* bias, 
    const T epsilon, T* output) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;

  if (threadIdx.x < ld) {
    val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[threadIdx.x];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
  }

  LayerNormSmall<T, TPB>(val, thread_data, ld, idx, beta, gamma, epsilon, output);
}

template <typename T, unsigned TPB> // TODO: T is redundant here!
__global__ void SkipLayerNormKernelSmall2(
    const int ld, const half2* input, const half2* skip, const half2* beta,
    const half2* gamma, const half2* bias, const half2 epsilon, half2* output) {
  // const half2 reverse_ld = T(1.f / ld);
  //const half2 reverse_ld = h2rcp(__float2half2_rn(float(ld))); // TODO
  
  // workaround for a llvm bug: https://github.com/intel/llvm/issues/5153
  const half2 one2 = __float2half2_rn(one);
  const half2 ld2 = __float2half2_rn(float(ld));
  const half2 reverse_ld = one2 / ld2;
  
  /*
  const half2 ld2 = __float2half2_rn(float(ld));
  const half2 reverse_ld = h2rcp(ld2);
  */
  const int offset = blockIdx.x * ld; // shall I refactor this offset

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  // cub::KeyValuePair<half2, half2> thread_data(0, 0); // TODO: How to initialize a half2 pair
  cub::KeyValuePair<half2, half2> thread_data(__float2half2_rn(float(0.0)), __float2half2_rn(float(0.0))); // TODO: How to initialize a half2 pair
  const int idx = offset + threadIdx.x;
  half2 val = __float2half2_rn(float(0.0));

  if (threadIdx.x < ld) {
    val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[threadIdx.x];
    const half2 rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<half2, half2>(rldval, rldval * val));
  }

  LayerNormSmall<half2, TPB>(val, thread_data, ld, idx, beta, gamma, epsilon, output);
}

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernel(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, const T* bias, 
    const T epsilon, T* output) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[i];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
    output[idx] = val;
  }

  LayerNorm<T, TPB>(thread_data, ld, offset, beta, gamma, epsilon, output);
}

template <typename T, unsigned TPB> // TODO: T is redundant here!
__global__ void SkipLayerNormKernel2(
    const int ld, const half2* input, const half2* skip, const half2* beta,
    const half2* gamma, const half2* bias, const half2 epsilon, half2* output) {
  // const half2 reverse_ld = T(1.f / ld);
  //const half2 reverse_ld = h2rcp(__float2half2_rn(float(ld))); // TODO
  const half2 one2 = __float2half2_rn(one);
  const half2 ld2 = __float2half2_rn(float(ld));
  const half2 reverse_ld = one2 / ld2; 
  const int offset = blockIdx.x * ld; // shall I refactor this offset

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  // cub::KeyValuePair<half2, half2> thread_data(0, 0); // TODO: How to initialize a half2 pair
  cub::KeyValuePair<half2, half2> thread_data(__float2half2_rn(float(0.0)), __float2half2_rn(float(0.0))); // TODO: How to initialize a half2 pair
  const int idx = offset + threadIdx.x;
  half2 val = __float2half2_rn(float(0.0)); // TODO: Can I initialize half2 like this?

  for (int i = threadIdx.x; i < ld; i += TPB) {
    // val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[threadIdx.x];
    // const half2 rldval = reverse_ld * val;
    // thread_data = pair_sum(thread_data, cub::KeyValuePair<half2, half2>(rldval, rldval * val));
    const int idx = offset + i;
    const half2 val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[i];
    const half2 rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<half2, half2>(rldval, rldval * val));
    output[idx] = val;
  }

  LayerNormSmall<half2, TPB>(val, thread_data, ld, idx, beta, gamma, epsilon, output);
}

template <typename T>
bool ComputeSkipLayerNorm(
    const cudaDeviceProp& prop, cudaStream_t stream, const int ld, const int n, const T* input,
    const T* skip, const T* beta, const T* gamma, const T* bias, const T epsilon, T* output, bool use_half2) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  if (use_half2 && 0 == (n & 1) && prop.major >= 7) {
    const int n2 = n / 2;
    const int grid_size = n2 / ld;

    const half2* input2 = reinterpret_cast<const half2*>(input);
    const half2* skip2 = reinterpret_cast<const half2*>(skip);
    const half2* beta2 = reinterpret_cast<const half2*>(beta);
    const half2* gamma2 = reinterpret_cast<const half2*>(gamma);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    half2* output2 = reinterpret_cast<half2*>(output);
    const half2 epsilon2 = __float2half2_rn(epsilon);

    if (ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall2<half2, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon2, output2);
      // TODO: ld / 2 ???
    } else if (ld <= 128) {
      constexpr int block_size = 128;
      SkipLayerNormKernelSmall2<half2, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon2, output2);
    } else if (ld == 384) {
      constexpr int block_size = 384;
      SkipLayerNormKernelSmall2<half2, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon2, output2);
    } else {
      // TODO: check if half2 also works for this function or not
      constexpr int block_size = 256;
      SkipLayerNormKernel2<half2, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon2, output2);
    }
  } else {
    const int grid_size = n / ld;
    if (ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall<T, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    } else if (ld <= 128) {
      constexpr int block_size = 128;
      SkipLayerNormKernelSmall<T, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    } else if (ld == 384) {
      constexpr int block_size = 384;
      SkipLayerNormKernelSmall<T, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<T, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchSkipLayerNormKernel(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    void* output,
    const void* input,
    const void* skip,
    const void* gamma,
    const void* beta,
    const void* bias,    
    float epsilon,
    int hidden_size,
    int element_count,
    size_t element_size,
    bool use_half2) {
  if (element_size == 2) {
    return ComputeSkipLayerNorm(
        prop,
        stream,
        hidden_size,
        element_count,
        reinterpret_cast<const half*>(input),
        reinterpret_cast<const half*>(skip),
        reinterpret_cast<const half*>(beta),
        reinterpret_cast<const half*>(gamma),
        reinterpret_cast<const half*>(bias),
        __float2half_rn(epsilon),
        reinterpret_cast<half*>(output),
        use_half2);
  } else {
    return ComputeSkipLayerNorm(
        prop,
        stream,
        hidden_size,
        element_count,
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(skip),
        reinterpret_cast<const float*>(beta),
        reinterpret_cast<const float*>(gamma),
        reinterpret_cast<const float*>(bias),
        epsilon,
        reinterpret_cast<float*>(output),
        false);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
