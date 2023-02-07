/*
 The implementation of this file is based on bert plugins in TensorRT demo:
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
#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ inline T Rsqrt(const T& x);

template <>
__device__ inline float Rsqrt(const float& x) {
  return rsqrtf(x);
}

template <>
__device__ inline half Rsqrt(const half& x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return hrsqrt(x);
#else
  return half(rsqrtf(float(x)));
#endif
}

__device__ inline half2 AddHalf2(const half2 a, const half2 b) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return __hadd2(a, b);
#else
  return __halves2half2(__hadd(a.x, b.x), __hadd(a.y, b.y));
#endif
}

struct KeyValuePairSum {
  __device__ inline cub::KeyValuePair<float, float> operator()(const cub::KeyValuePair<float, float>& a,
                                                               const cub::KeyValuePair<float, float>& b) {
    return cub::KeyValuePair<float, float>(a.key + b.key, a.value + b.value);
  }

  __device__ inline cub::KeyValuePair<half, half> operator()(const cub::KeyValuePair<half, half>& a,
                                                             const cub::KeyValuePair<half, half>& b) {
    const half2 a2 = __halves2half2(a.key, a.value);
    const half2 b2 = __halves2half2(b.key, b.value);
    const half2 res = AddHalf2(a2, b2);
    return cub::KeyValuePair<half, half>(__low2half(res), __high2half(res));
  }

  __device__ inline cub::KeyValuePair<half2, half2> operator()(const cub::KeyValuePair<half2, half2>& a,
                                                               const cub::KeyValuePair<half2, half2>& b) {
    return cub::KeyValuePair<half2, half2>(AddHalf2(a.key, b.key), AddHalf2(a.value, b.value));
  }
};

template <typename T, int TPB>
__device__ inline void LayerNorm(
    const cub::KeyValuePair<T, T>& thread_data, const int ld, const int offset, const T* beta,
    const T* gamma, const T epsilon, T* output) {
  // Assuming thread_data is already divided by ld

  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<T, T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  KeyValuePairSum pair_sum;
  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = Rsqrt(sum_kv.value - mu * mu + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = output[idx];
    const T g(gamma[i]);
    const T b = (nullptr == beta) ? (T)0 : beta[i];
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, int TPB>
__device__ inline void SimplifiedLayerNorm(
    const T& thread_data, const int ld, const int offset, const T* gamma, const T epsilon, T* output) {
  // Assuming thread_data is already divided by ld

  using BlockReduce = cub::BlockReduce<T, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T rsigma;  // 1 / std.dev.

  const T sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    rsigma = Rsqrt(sum + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = output[idx];
    const T g(gamma[i]);
    output[idx] = g * val * rsigma;
  }
}

template <typename T, int TPB, int ILP>
__device__ inline void LayerNormSmall(const T* input_v, const cub::KeyValuePair<T, T>& thread_data,
                                      const int ld, const int idx, const T* beta, const T* gamma,
                                      const T epsilon, T* output) {
  // Assuming thread_data is already divided by ld
  // Small settings: the block covers the leading dimension TPB >= ld. The input
  // value is available in a register
  using VecT = aligned_vector<T, ILP>;
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<T, T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.
  T beta_v[ILP], gamma_v[ILP], output_v[ILP];

  if (beta != nullptr) {
    VecT* beta_val = reinterpret_cast<VecT*>(&beta_v);
    *beta_val = *reinterpret_cast<const VecT*>(&beta[threadIdx.x * ILP]);
  }
  VecT* gamma_val = reinterpret_cast<VecT*>(&gamma_v);
  *gamma_val = *reinterpret_cast<const VecT*>(&gamma[threadIdx.x * ILP]);

  VecT* output_val = reinterpret_cast<VecT*>(&output_v);

  KeyValuePairSum pair_sum;
  const cub::KeyValuePair<T, T> sum_kv = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = Rsqrt(sum_kv.value - mu * mu + epsilon);
  }
  __syncthreads();

  if (ILP * threadIdx.x < ld) {
#pragma unroll
    for (int i = 0; i < ILP; i++) {
      output_v[i] = (beta != nullptr)
                        ? gamma_v[i] * (input_v[i] - mu) * rsigma + beta_v[i]
                        : gamma_v[i] * (input_v[i] - mu) * rsigma;
    }
    *(reinterpret_cast<VecT*>(&output[idx])) = *output_val;
  }
}

template <typename T, int TPB, int ILP>
__device__ inline void SimplifiedLayerNormSmall(const T* input_v, const T& thread_data, const int ld, const int idx,
                                                const T* gamma, const T epsilon, T* output) {
  // Assuming thread_data is already divided by ld
  // Small settings: the block covers the leading dimension TPB >= ld. The input
  // value is available in a register
  using VecT = aligned_vector<T, ILP>;
  using BlockReduce = cub::BlockReduce<T, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T rsigma;  // 1 / std.dev.
  T gamma_v[ILP], output_v[ILP];

  VecT* gamma_val = reinterpret_cast<VecT*>(&gamma_v);
  *gamma_val = *reinterpret_cast<const VecT*>(&gamma[threadIdx.x * ILP]);

  VecT* output_val = reinterpret_cast<VecT*>(&output_v);

  const T sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    rsigma = Rsqrt(sum + epsilon);
  }
  __syncthreads();

  if (ILP * threadIdx.x < ld) {
#pragma unroll
    for (int i = 0; i < ILP; i++) {
      output_v[i] = gamma_v[i] * input_v[i] * rsigma;
    }
    *(reinterpret_cast<VecT*>(&output[idx])) = *output_val;
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
