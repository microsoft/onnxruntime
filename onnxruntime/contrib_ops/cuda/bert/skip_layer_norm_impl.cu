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

// Modifications: Add SkipLayerNormKernelSmallVec and SkipLayerNormKernelVec to
//                leverage vectorized load/write.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

template <unsigned TPB, int ILP>
__global__ void SkipLayerNormKernelSmallVec(
    const int ld, const half* input, const half* skip, const half* beta, const half* gamma,
    const half* bias, const half epsilon, half* output, bool hasBias) {
  const half rld = half(1.f) / half(ld);
  const int idx = blockIdx.x * ld + threadIdx.x * ILP;  // grid_size = n / ld

  using VecT = aligned_vector<half, ILP>;
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<half, half>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ half mu;      // mean
  __shared__ half rsigma;  // 1 / std.dev.
  KeyValuePairSum pair_sum;

  using VecT = aligned_vector<half, ILP>;
  half input_v[ILP], skip_v[ILP], beta_v[ILP], gamma_v[ILP], bias_v[ILP], output_v[ILP];

  VecT* input_val = reinterpret_cast<VecT*>(&input_v);
  *input_val = *reinterpret_cast<const VecT*>(&input[idx]);

  VecT* skip_val = reinterpret_cast<VecT*>(&skip_v);
  *skip_val = *reinterpret_cast<const VecT*>(&skip[idx]);

  if (beta != nullptr) {
    VecT* beta_val = reinterpret_cast<VecT*>(&beta_v);
    *beta_val = *reinterpret_cast<const VecT*>(&beta[threadIdx.x * ILP]);
  }

  VecT* gamma_val = reinterpret_cast<VecT*>(&gamma_v);
  *gamma_val = *reinterpret_cast<const VecT*>(&gamma[threadIdx.x * ILP]);

  if (hasBias) {
    VecT* bias_val = reinterpret_cast<VecT*>(&bias_v);
    *bias_val = *reinterpret_cast<const VecT*>(&bias[threadIdx.x * ILP]);
  }

  VecT* output_val = reinterpret_cast<VecT*>(&output_v);

  cub::KeyValuePair<half, half> thread_data;
  if (ILP * threadIdx.x < ld) {
    half rldval_sum = half(0.f);
    half rldvalsq_sum = half(0.f);
    #pragma unroll
    for (int i = 0; i < ILP; i++) {
      input_v[i] += skip_v[i];
      if (hasBias) {
        input_v[i] += bias_v[i];
      }
      const half rldval = rld * input_v[i];
      rldval_sum += rldval;
      rldvalsq_sum += rldval * input_v[i];
    }
    thread_data = cub::KeyValuePair<half, half>(rldval_sum, rldvalsq_sum);
  } else {
    thread_data = cub::KeyValuePair<half, half>(half(0.f), half(0.f));
  }
  const cub::KeyValuePair<half, half> sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);
  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = Rsqrt(sumKV.value - mu * mu + epsilon);
  }
  __syncthreads();
  if (ILP * threadIdx.x < ld) {
    #pragma unroll
    for (int i = 0; i < ILP; i++) {
      output_v[i] = (beta != nullptr) ? gamma_v[i] * (input_v[i] - mu) * rsigma + beta_v[i] :
                                        gamma_v[i] * (input_v[i] - mu) * rsigma;
    }
    *(reinterpret_cast<VecT*>(&output[idx])) = *reinterpret_cast<VecT*>(&output_v[0]);
  }
}

template <unsigned TPB, int ILP>
__global__ void SkipLayerNormKernelVec(
    const int ld, const half* input, const half* skip, const half* beta, const half* gamma,
    const half* bias, const half epsilon, half* output, bool hasBias) {
  const half rld = half(1.f) / half(ld);
  const int idx = blockIdx.x * ld + threadIdx.x * ILP; // TPB = "ld / ILP", grid_size = n / ld

  using VecT = aligned_vector<half, ILP>;
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<half, half>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ half mu;      // mean
  __shared__ half rsigma;  // 1 / std.dev.
  KeyValuePairSum pair_sum;

  using VecT = aligned_vector<half, ILP>;
  half input_v[ILP], skip_v[ILP], beta_v[ILP], gamma_v[ILP], bias_v[ILP], output_v[ILP];

  VecT* input_val = reinterpret_cast<VecT*>(&input_v);
  *input_val = *reinterpret_cast<const VecT*>(&input[idx]);

  VecT* skip_val = reinterpret_cast<VecT*>(&skip_v);
  *skip_val = *reinterpret_cast<const VecT*>(&skip[idx]);

  if (beta != nullptr) {
    VecT* beta_val = reinterpret_cast<VecT*>(&beta_v);
    *beta_val = *reinterpret_cast<const VecT*>(&beta[threadIdx.x * ILP]);
  }

  VecT* gamma_val = reinterpret_cast<VecT*>(&gamma_v);
  *gamma_val = *reinterpret_cast<const VecT*>(&gamma[threadIdx.x * ILP]);

  if (hasBias) {
    VecT* bias_val = reinterpret_cast<VecT*>(&bias_v);
    *bias_val = *reinterpret_cast<const VecT*>(&bias[threadIdx.x * ILP]);
  }

  VecT* output_val = reinterpret_cast<VecT*>(&output_v);

  half rldval_sum = half(0.f);
  half rldvalsq_sum = half(0.f);
  #pragma unroll
  for (int i = 0; i < ILP; i++) {
    input_v[i] += skip_v[i];
    if (hasBias) {
      input_v[i] += bias_v[i];
    }
    const half rldval = rld * input_v[i];
    rldval_sum += rldval;
    rldvalsq_sum += rldval * input_v[i];
  }
  cub::KeyValuePair<half, half> thread_data(rldval_sum, rldvalsq_sum);
  const cub::KeyValuePair<half, half> sumKV = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);
  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = Rsqrt(sumKV.value - mu * mu + epsilon);
  }
  __syncthreads();

  #pragma unroll
  for (int i = 0; i < ILP; i++) {
    output_v[i] = (beta != nullptr) ? gamma_v[i] * (input_v[i] - mu) * rsigma + beta_v[i] :
                                      gamma_v[i] * (input_v[i] - mu) * rsigma;
  }
  *(reinterpret_cast<VecT*>(&output[idx])) = *reinterpret_cast<VecT*>(&output_v[0]);
}

/* float32 */
bool ComputeSkipLayerNorm(cudaStream_t stream, const int ld, const int n, const float* input,
    const float* skip, const float* beta, const float* gamma, const float* bias, const float epsilon,
    float* output, bool use_half2) {
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
    SkipLayerNormKernel<float, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

/* half16 */
bool ComputeSkipLayerNorm(
    cudaStream_t stream, const int ld, const int n, const half* input, const half* skip, const half* beta,
    const half* gamma, const half* bias, const half epsilon, half* output, bool use_half2) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  if (use_half2 && 0 == (n & 1)) {
    const int grid_size = n / ld;
    bool hasBias = (bias == nullptr) ? false : true;
    if (ld <= 32) {
      constexpr int block_size = 32 / 4;
      SkipLayerNormKernelSmallVec<block_size, 4>
         <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld == 64) {
      constexpr int block_size = 64 / 4;
      SkipLayerNormKernelVec<block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld == 128) {
      constexpr int block_size = 128 / 4;
      SkipLayerNormKernelVec<block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld == 384) {
      constexpr int block_size = 384 / 4;
      SkipLayerNormKernelVec<block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld == 768) {
      constexpr int block_size = 768 / 4;
      SkipLayerNormKernelVec<block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld == 1024) {
      constexpr int block_size = 1024 / 4;
      SkipLayerNormKernelVec<block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else {
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
    } else if (ld <= 64) {
      constexpr int block_size = 64;
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
