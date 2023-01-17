#include "hip/hip_runtime.h"
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

#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <hipcub/hipcub.hpp>
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/shared_inc/rocm_call.h"

using namespace onnxruntime::rocm;

namespace onnxruntime {
namespace contrib {
namespace rocm {

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
  return half(rsqrtf(static_cast<float>(x)));
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
  __device__ inline hipcub::KeyValuePair<float, float> operator()(const hipcub::KeyValuePair<float, float>& a,
                                                                  const hipcub::KeyValuePair<float, float>& b) {
    return hipcub::KeyValuePair<float, float>(a.key + b.key, a.value + b.value);
  }

  __device__ inline hipcub::KeyValuePair<half, half> operator()(const hipcub::KeyValuePair<half, half>& a,
                                                                const hipcub::KeyValuePair<half, half>& b) {
    const half2 a2 = __halves2half2(a.key, a.value);
    const half2 b2 = __halves2half2(b.key, b.value);
    const half2 res = AddHalf2(a2, b2);
    return hipcub::KeyValuePair<half, half>(__low2half(res), __high2half(res));
  }

  __device__ inline hipcub::KeyValuePair<half2, half2> operator()(const hipcub::KeyValuePair<half2, half2>& a,
                                                                  const hipcub::KeyValuePair<half2, half2>& b) {
    return hipcub::KeyValuePair<half2, half2>(AddHalf2(a.key, b.key), AddHalf2(a.value, b.value));
  }
};

template <typename T, int TPB>
__device__ inline void LayerNorm(
    const hipcub::KeyValuePair<T, T>& thread_data, const int ld, const int offset, const T* beta,
    const T* gamma, const T epsilon, T* output) {
  // Assuming thread_data is already divided by ld

  using BlockReduce = hipcub::BlockReduce<hipcub::KeyValuePair<T, T>, TPB>;
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

template <typename T, int TPB, int ILP>
__device__ inline void LayerNormVec(
    const hipcub::KeyValuePair<T, T>& thread_data, const int ld, const int offset, const T* beta,
    const T* gamma, const T epsilon, T* output) {
  // Assuming thread_data is already divided by ld
  using VecT = aligned_vector<T, ILP>;
  using BlockReduce = hipcub::BlockReduce<hipcub::KeyValuePair<T, T>, TPB>;
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

  if (ILP * threadIdx.x < ld) {
    T beta_v[ILP], gamma_v[ILP], output_v[ILP];
    VecT* gamma_val = reinterpret_cast<VecT*>(&gamma_v);
    VecT* output_val = reinterpret_cast<VecT*>(&output_v);

    for (int i = threadIdx.x * ILP; i < ld; i += TPB * ILP) {
      int idx = offset + i;
      if (beta != nullptr) {
        VecT* beta_val = reinterpret_cast<VecT*>(&beta_v);
        *beta_val = *reinterpret_cast<const VecT*>(&beta[i]);
      }
      *gamma_val = *reinterpret_cast<const VecT*>(&gamma[i]);
      *output_val = *reinterpret_cast<const VecT*>(&output[idx]);
      #pragma unroll
      for (int k = 0; k < ILP; k++) {
        output_v[k] = (beta != nullptr) ? gamma_v[k] * (output_v[k] - mu) * rsigma + beta_v[k] :
                                          gamma_v[k] * (output_v[k] - mu) * rsigma;
      }
      *(reinterpret_cast<VecT*>(&output[idx])) = *reinterpret_cast<VecT*>(&output_v[0]);
    }
  }
}

template <typename T, int TPB, int ILP>
__device__ inline void LayerNormSmall(const T* input_v, const hipcub::KeyValuePair<T, T>& thread_data,
                                      const int ld, const int idx, const T* beta, const T* gamma,
                                      const T epsilon, T* output) {
  // Assuming thread_data is already divided by ld
  // Small settings: the block covers the leading dimension TPB >= ld. The input
  // value is available in a register
  using VecT = aligned_vector<T, ILP>;
  using BlockReduce = hipcub::BlockReduce<hipcub::KeyValuePair<T, T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.
  T beta_v[ILP], gamma_v[ILP], output_v[ILP];

  KeyValuePairSum pair_sum;
  const hipcub::KeyValuePair<T, T> sum_kv = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = Rsqrt(sum_kv.value - mu * mu + epsilon);
  }
  __syncthreads();

  if (ILP * threadIdx.x < ld) {
    if (beta != nullptr) {
      VecT* beta_val = reinterpret_cast<VecT*>(&beta_v);
      *beta_val = *reinterpret_cast<const VecT*>(&beta[threadIdx.x * ILP]);
    }

    VecT* gamma_val = reinterpret_cast<VecT*>(&gamma_v);
    *gamma_val = *reinterpret_cast<const VecT*>(&gamma[threadIdx.x * ILP]);

    #pragma unroll
    for (int i = 0; i < ILP; i++) {
      output_v[i] = (beta != nullptr) ? gamma_v[i] * (input_v[i] - mu) * rsigma + beta_v[i] :
                                        gamma_v[i] * (input_v[i] - mu) * rsigma;
    }
    *(reinterpret_cast<VecT*>(&output[idx])) = *reinterpret_cast<VecT*>(&output_v[0]);
  }
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
