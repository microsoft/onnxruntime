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

template <unsigned TPB>
__global__ void SkipLayerNormKernelSmall2(
  const int ld, const half2* input, const half2* skip, const half2* beta, const half2* gamma, const half2* bias, const half epsilon, half2* output, bool hasBias) {
  const int idx = (ld / 2) * blockIdx.x + threadIdx.x;
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<half, half>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ half mu;     // mean
  __shared__ half rsigma; // 1 / std.dev.
  KeyValuePairSum pair_sum;
  half2 input_vec;
  half2 beta_vec;
  half2 gamma_vec;
  cub::KeyValuePair<half, half> thread_data;
  if (2 * threadIdx.x < ld) {
    input_vec = input[idx];
    const half2 skip_vec = skip[idx];
    const half2 bias_vec = (hasBias) ? bias[threadIdx.x] : __float2half2_rn(0.f);
    const half rld = half(1.f) / half(ld);
    beta_vec = (beta == nullptr) ? __float2half2_rn(0.f) : beta[threadIdx.x];
    gamma_vec = gamma[threadIdx.x];

    input_vec += skip_vec;
    if (hasBias) {
      input_vec += bias_vec;
    }
    thread_data = cub::KeyValuePair<half, half>(rld * (__low2half(input_vec) + __high2half(input_vec)),
                                                rld * (__low2half(input_vec) * __low2half(input_vec)) +
                                                rld * (__high2half(input_vec) * __high2half(input_vec)));
  } else {
    thread_data = cub::KeyValuePair<half, half>(half(0.f), half(0.f));
  }
  const cub::KeyValuePair<half, half> sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);
  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = Rsqrt(sumKV.value - mu * mu + epsilon); // half --> sumKV.value.x + sumKV.value.y - mu * mu + epsilon
  }
  __syncthreads();
  if (2 * threadIdx.x < ld) {
    half output_low = __low2half(gamma_vec) * (__low2half(input_vec) - mu) * rsigma + __low2half(beta_vec);
    half output_high = __high2half(gamma_vec) * (__high2half(input_vec) - mu) * rsigma + __high2half(beta_vec);
    output[idx] = __halves2half2(output_low, output_high); // For debugging purpose
  }
}

template <unsigned TPB>
__global__ void SkipLayerNormKernelVec2(
  const int ld, const half2* input, const half2* skip, const half2* beta, const half2* gamma, const half2* bias, const half epsilon, half2* output, bool hasBias) {
  const int idx = TPB * blockIdx.x + threadIdx.x;
  half2 input_vec = input[idx];
  const half2 skip_vec = skip[idx];
  const half2 beta_vec = (beta == nullptr) ? __float2half2_rn(0.f) : beta[threadIdx.x];
  const half2 gamma_vec = gamma[threadIdx.x];
  const half2 bias_vec = (hasBias) ? bias[threadIdx.x] : __float2half2_rn(0.f);
  const half rld = half(1.f) / half(ld);

  input_vec += skip_vec;
  if (hasBias) {
    input_vec += bias_vec;
  }
  cub::KeyValuePair<half, half> thread_data(rld * (__low2half(input_vec) + __high2half(input_vec)),
                                            rld * (__low2half(input_vec) * __low2half(input_vec)) +
                                            rld * (__high2half(input_vec) * __high2half(input_vec)));
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<half, half>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ half mu;     // mean
  __shared__ half rsigma; // 1 / std.dev.

  KeyValuePairSum pair_sum;
  const cub::KeyValuePair<half, half> sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = Rsqrt(sumKV.value - mu * mu + epsilon);
  }
  __syncthreads();

  half output_low = __low2half(gamma_vec) * (__low2half(input_vec) - mu) * rsigma + __low2half(beta_vec);
  half output_high = __high2half(gamma_vec) * (__high2half(input_vec) - mu) * rsigma + __high2half(beta_vec);
  output[idx] = __halves2half2(output_low, output_high);
}

/* float32 */
bool ComputeSkipLayerNorm(
    const cudaDeviceProp& prop, cudaStream_t stream, const int ld, const int n, const float* input,
    const float* skip, const float* beta, const float* gamma, const float* bias, const float epsilon, float* output, bool use_half2) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  const int grid_size = n / ld;

  if (ld <= 32) {
    constexpr int block_size = 32;
    SkipLayerNormKernelSmall<float, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  } else if (ld <= 128) {
    constexpr int block_size = 128;
    SkipLayerNormKernelSmall<float, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  } else if (ld == 384) {
    constexpr int block_size = 384;
    SkipLayerNormKernelSmall<float, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  } else {
    constexpr int block_size = 256;
    SkipLayerNormKernel<float, block_size><<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

/* half16 */
bool ComputeSkipLayerNorm(
    const cudaDeviceProp& prop, cudaStream_t stream, const int ld, const int n, const half* input,
    const half* skip, const half* beta, const half* gamma, const half* bias, const half epsilon, half* output, bool use_half2) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  if (use_half2 && 0 == (n & 1) && prop.major >= 7) {
    const int grid_size = n / ld;
    const half2* input2 = reinterpret_cast<const half2*>(input);
    const half2* skip2 = reinterpret_cast<const half2*>(skip);
    const half2* beta2 = reinterpret_cast<const half2*>(beta);
    const half2* gamma2 = reinterpret_cast<const half2*>(gamma);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    half2* output2 = reinterpret_cast<half2*>(output);
    const half2 epsilon2 = __half2half2(epsilon);
    constexpr int VPT = 32 / sizeof(half); // 16 (og)
    bool hasBias = (bias == nullptr) ? false : true; // TODO: template args (define in .cc file)

    if (ld <= 32) {
      constexpr int block_size = 32; // for testing
      SkipLayerNormKernelSmall2<block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon, output2, hasBias);
    } else if (ld == 128) {
      constexpr int block_size = 128 / 2; // for testing
      SkipLayerNormKernelVec2<block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon, output2, hasBias);
    } else if (ld == 384) {
      constexpr int block_size = 384 / 2; // for testing
      SkipLayerNormKernelVec2<block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon, output2, hasBias);
    } else if (ld == 768) {
      constexpr int block_size = 768 / 2; // for testing
      SkipLayerNormKernelVec2<block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon, output2, hasBias);
    } else if (ld == 1024) {
      constexpr int block_size = 1024 / 2; // for testing
      SkipLayerNormKernelVec2<block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input2, skip2, beta2, gamma2, bias2, epsilon, output2, hasBias);
    } else {
      // TODO: check if half2 also works for this function or not
      constexpr int block_size = 256;
      SkipLayerNormKernel<half, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  } else {
    const int grid_size = n / ld;
    if (ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall<half, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    } else if (ld <= 128) {
      constexpr int block_size = 128;
      SkipLayerNormKernelSmall<half, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    } else if (ld == 384) {
      constexpr int block_size = 384;
      SkipLayerNormKernelSmall<half, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<half, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

template<>
bool LaunchSkipLayerNormKernel(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    half* output,
    const half* input,
    const half* skip,
    const half* gamma,
    const half* beta,
    const half* bias,
    float epsilon,
    int hidden_size,
    int element_count,
    size_t element_size,
    bool use_half2) {

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
}

template<>
bool LaunchSkipLayerNormKernel(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    float* output,
    const float* input,
    const float* skip,
    const float* gamma,
    const float* beta,
    const float* bias,
    float epsilon,
    int hidden_size,
    int element_count,
    size_t element_size,
    bool use_half2) {

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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

