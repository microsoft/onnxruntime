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

template <typename U, typename V, int TPB>
__device__ inline void LayerNorm(
    const hipcub::KeyValuePair<U, U>& thread_data, const int ld, const int offset, const V* beta,
    const V* gamma, const U epsilon, V* output) {
  // Assuming thread_data is already divided by ld

  using BlockReduce = hipcub::BlockReduce<hipcub::KeyValuePair<U, U>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U mu;      // mean
  __shared__ U rsigma;  // 1 / std.dev.

  KeyValuePairSum pair_sum;
  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = Rsqrt(sum_kv.value - mu * mu + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const U val = static_cast<U>(output[idx]);
    const U g = static_cast<U>(gamma[i]);
    const U b = (nullptr == beta) ? U(0.f) : static_cast<U>(beta[i]);
    output[idx] = static_cast<V>(g * (val - mu) * rsigma + b);
  }
}

template <typename U, typename V, int TPB>
__device__ inline void SimplifiedLayerNorm(
    const U& thread_data, const int ld, const int offset, const V* gamma, const U epsilon, V* output) {
  // Assuming thread_data is already divided by ld

  using BlockReduce = hipcub::BlockReduce<U, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U rsigma;  // 1 / std.dev.

  const U sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    rsigma = Rsqrt(sum + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const U val = static_cast<U>(output[idx]);
    const U g = static_cast<U>(gamma[i]);
    output[idx] = static_cast<V>(g * val * rsigma);
  }
}

template <typename U, typename V, int TPB, int ILP>
__device__ inline void SimplifiedLayerNormVec(
    const U& thread_data, const int ld, const int offset, const V* gamma, const U epsilon, V* output) {
  // Assuming thread_data is already divided by ld
  using VecV = aligned_vector<V, ILP>;
  using BlockReduce = hipcub::BlockReduce<U, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U rsigma;  // 1 / std.dev.

  const U sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    rsigma = Rsqrt(sum + epsilon);
  }
  __syncthreads();

  if (ILP * threadIdx.x < ld) {
    for (int i = threadIdx.x * ILP; i < ld; i += TPB * ILP) {
      int idx = offset + i;
      const VecV gamma_v = *reinterpret_cast<const VecV*>(gamma + i);
      VecV output_v = *reinterpret_cast<const VecV*>(output + idx);

      #pragma unroll
      for (int k = 0; k < ILP; k++) {
        output_v.val[k] = U(gamma_v.val[k]) * U(output_v.val[k]) * rsigma;
      }
      *(reinterpret_cast<VecV*>(output + idx)) = output_v;
    }
  }
}

template <typename U, typename V, int TPB, int ILP>
__device__ inline void LayerNormVec(
    const hipcub::KeyValuePair<U, U>& thread_data, const int ld, const int offset, const V* beta,
    const V* gamma, const U epsilon, V* output) {
  // Assuming thread_data is already divided by ld
  using VecV = aligned_vector<V, ILP>;
  using BlockReduce = hipcub::BlockReduce<hipcub::KeyValuePair<U, U>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U mu;      // mean
  __shared__ U rsigma;  // 1 / std.dev.

  KeyValuePairSum pair_sum;
  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = Rsqrt(sum_kv.value - mu * mu + epsilon);
  }
  __syncthreads();

  if (ILP * threadIdx.x < ld) {
    for (int i = threadIdx.x * ILP; i < ld; i += TPB * ILP) {
      int idx = offset + i;
      const VecV beta_v = (beta != nullptr) ? *reinterpret_cast<const VecV*>(beta + i) : VecV();
      const VecV gamma_v = *reinterpret_cast<const VecV*>(gamma + i);
      VecV output_v = *reinterpret_cast<const VecV*>(output + idx);

      #pragma unroll
      for (int k = 0; k < ILP; k++) {
        output_v.val[k] = (beta != nullptr) ? U(gamma_v.val[k]) * (U(output_v.val[k]) - mu) * rsigma + U(beta_v.val[k]) :
                                              U(gamma_v.val[k]) * (U(output_v.val[k]) - mu) * rsigma;
      }
      *(reinterpret_cast<VecV*>(output + idx)) = output_v;
    }
  }
}

template <typename T, typename U, typename V, int TPB, int ILP>
__device__ inline void LayerNormSmall(const T* input_v, const hipcub::KeyValuePair<U, U>& thread_data,
                                      const int ld, const int idx, const V* beta, const V* gamma,
                                      const U epsilon, V* output) {
  // Assuming thread_data is already divided by ld
  // Small settings: the block covers the leading dimension TPB >= ld. The input
  // value is available in a register
  using VecV = aligned_vector<V, ILP>;
  using BlockReduce = hipcub::BlockReduce<hipcub::KeyValuePair<U, U>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U mu;      // mean
  __shared__ U rsigma;  // 1 / std.dev.

  KeyValuePairSum pair_sum;
  const hipcub::KeyValuePair<U, U> sum_kv = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = Rsqrt(sum_kv.value - mu * mu + epsilon);
  }
  __syncthreads();

  if (ILP * threadIdx.x < ld) {
    const VecV beta_v = (beta != nullptr) ? *reinterpret_cast<const VecV*>(beta + threadIdx.x * ILP) : VecV();
    const VecV gamma_v = *reinterpret_cast<const VecV*>(gamma + threadIdx.x * ILP);
    VecV output_v;

    #pragma unroll
    for (int i = 0; i < ILP; i++) {
      output_v.val[i] = (beta != nullptr) ? U(gamma_v.val[i]) * (U(input_v[i]) - mu) * rsigma + U(beta_v.val[i]) :
                                            U(gamma_v.val[i]) * (U(input_v[i]) - mu) * rsigma;
    }
    *(reinterpret_cast<VecV*>(output + idx)) = output_v;
  }
}

template <typename T, typename U, typename V, int TPB, int ILP>
__device__ inline void SimplifiedLayerNormSmall(const T* input_v, const U& thread_data, const int ld, const int idx,
                                                const V* gamma, const U epsilon, V* output) {
  // Assuming thread_data is already divided by ld
  // Small settings: the block covers the leading dimension TPB >= ld. The input
  // value is available in a register
  using VecV = aligned_vector<V, ILP>;
  using BlockReduce = hipcub::BlockReduce<U, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U rsigma;  // 1 / std.dev.

  const U sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    rsigma = Rsqrt(sum + epsilon);
  }
  __syncthreads();

  if (ILP * threadIdx.x < ld) {
    const VecV gamma_v = *reinterpret_cast<const VecV*>(gamma + threadIdx.x * ILP);
    VecV output_v;

    #pragma unroll
    for (int i = 0; i < ILP; i++) {
      output_v.val[i] = U(gamma_v.val[i]) * U(input_v[i]) * rsigma;
    }
    *(reinterpret_cast<VecV*>(output + idx)) = output_v;
  }
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
