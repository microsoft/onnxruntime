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
#include <cuda_bf16.h>
#include <cublas_v2.h>

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

template <>
__device__ inline nv_bfloat16 Rsqrt(const nv_bfloat16& x) {
  return hrsqrt(x);
}

__device__ inline nv_bfloat162 AddHalf2(const nv_bfloat162 a, const nv_bfloat162 b) {
  return __hadd2(a, b);
}

struct WelfordData {
  float mean;
  float m2;
  int count;

  __device__ inline void Update(float value) {
    ++count;
    const float delta = value - mean;
    mean += delta / static_cast<float>(count);
    m2 += delta * (value - mean);
  }
};

struct WelfordCombine {
  __device__ inline WelfordData operator()(const WelfordData& a, const WelfordData& b) const {
    if (a.count == 0) return b;
    if (b.count == 0) return a;

    const int count = a.count + b.count;
    const float delta = b.mean - a.mean;
    const float b_weight = static_cast<float>(b.count) / static_cast<float>(count);
    return {a.mean + delta * b_weight,
            a.m2 + b.m2 + delta * delta * static_cast<float>(a.count) * b_weight,
            count};
  }
};

template <typename T, int TPB>
__device__ inline void LayerNorm(
    const WelfordData& thread_data, const float origin, const int ld, const int offset, const T* beta,
    const T* gamma, const float epsilon, T* output) {
  // Statistics and centering stay relative to origin to retain small FP32 differences.

  using BlockReduce = cub::BlockReduce<WelfordData, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float mu;      // mean
  __shared__ float rsigma;  // 1 / std.dev.

  const auto stats = BlockReduce(temp_storage).Reduce(thread_data, WelfordCombine{});

  if (threadIdx.x == 0) {
    mu = stats.mean;
    rsigma = rsqrtf(stats.m2 / static_cast<float>(stats.count) + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const float val = static_cast<float>(output[idx]);
    const float g = static_cast<float>(gamma[i]);
    const float b = (nullptr == beta) ? 0.f : static_cast<float>(beta[i]);
    output[idx] = static_cast<T>(g * ((val - origin) - mu) * rsigma + b);
  }
}

template <typename T, int TPB>
__device__ inline void SimplifiedLayerNorm(
    const float& thread_data, const int ld, const int offset, const T* gamma, const float epsilon, T* output) {
  // Assuming thread_data is already divided by ld
  // Uses fp32 accumulation to avoid overflow in fp16/bf16.

  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float rsigma;  // 1 / std.dev.

  const float sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    rsigma = rsqrtf(sum + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const float val = static_cast<float>(output[idx]);
    const float g = static_cast<float>(gamma[i]);
    output[idx] = static_cast<T>(g * val * rsigma);
  }
}

template <typename T, int TPB, int ILP>
__device__ inline void LayerNormSmall(const T* input_v, const WelfordData& thread_data, const float origin,
                                      const int ld, const int idx, const T* beta, const T* gamma,
                                      const float epsilon, T* output) {
  // Input values are in registers; statistics and centering are relative to origin.
  using VecT = aligned_vector<T, ILP>;
  using BlockReduce = cub::BlockReduce<WelfordData, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float mu;      // mean
  __shared__ float rsigma;  // 1 / std.dev.
  T gamma_v[ILP], output_v[ILP];

  const bool is_valid = ILP * threadIdx.x < ld;
  T beta_v[ILP];
  if (is_valid) {
    if (beta != nullptr) {
      VecT* beta_val = reinterpret_cast<VecT*>(&beta_v);
      *beta_val = *reinterpret_cast<const VecT*>(&beta[threadIdx.x * ILP]);
    }

    VecT* gamma_val = reinterpret_cast<VecT*>(&gamma_v);
    *gamma_val = *reinterpret_cast<const VecT*>(&gamma[threadIdx.x * ILP]);
  }

  const auto stats = BlockReduce(temp_storage).Reduce(thread_data, WelfordCombine{});

  if (threadIdx.x == 0) {
    mu = stats.mean;
    rsigma = rsqrtf(stats.m2 / static_cast<float>(stats.count) + epsilon);
  }
  __syncthreads();

  if (is_valid) {
#pragma unroll
    for (int i = 0; i < ILP; i++) {
      const float in_f = static_cast<float>(input_v[i]);
      const float g_f = static_cast<float>(gamma_v[i]);
      const float b_f = (beta != nullptr) ? static_cast<float>(beta_v[i]) : 0.f;
      output_v[i] = static_cast<T>(g_f * ((in_f - origin) - mu) * rsigma + b_f);
    }

    VecT* output_val = reinterpret_cast<VecT*>(&output_v);
    *(reinterpret_cast<VecT*>(&output[idx])) = *output_val;
  }
}

template <typename T, int TPB, int ILP>
__device__ inline void SimplifiedLayerNormSmall(const T* input_v, const float& thread_data, const int ld, const int idx,
                                                const T* gamma, const float epsilon, T* output) {
  // Assuming thread_data is already divided by ld
  // Small settings: the block covers the leading dimension TPB >= ld. The input
  // value is available in a register
  // Uses fp32 accumulation to avoid overflow in fp16/bf16.
  using VecT = aligned_vector<T, ILP>;
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float rsigma;  // 1 / std.dev.

  const bool is_valid = ILP * threadIdx.x < ld;

  T gamma_v[ILP], output_v[ILP];

  if (is_valid) {
    VecT* gamma_val = reinterpret_cast<VecT*>(&gamma_v);
    *gamma_val = *reinterpret_cast<const VecT*>(&gamma[threadIdx.x * ILP]);
  }

  const float sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    rsigma = rsqrtf(sum + epsilon);
  }
  __syncthreads();

  if (is_valid) {
#pragma unroll
    for (int i = 0; i < ILP; i++) {
      const float in_f = static_cast<float>(input_v[i]);
      const float g_f = static_cast<float>(gamma_v[i]);
      output_v[i] = static_cast<T>(g_f * in_f * rsigma);
    }

    VecT* output_val = reinterpret_cast<VecT*>(&output_v);
    *(reinterpret_cast<VecT*>(&output[idx])) = *output_val;
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
