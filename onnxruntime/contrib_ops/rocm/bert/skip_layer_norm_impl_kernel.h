// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_fp16.h>
#include "contrib_ops/rocm/bert/layer_norm.cuh"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
T maybe2half(float x);

template <>
float maybe2half(float x) {
  return x;
}

template <>
half maybe2half(float x) {
  return __float2half_rn(x);
}

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernel(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, const T* bias,
    const T epsilon, T* output) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  hipcub::KeyValuePair<T, T> thread_data(0, 0);

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[i];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, hipcub::KeyValuePair<T, T>(rldval, rldval * val));
    output[idx] = val;
  }

  LayerNorm<T, TPB>(thread_data, ld, offset, beta, gamma, epsilon, output);
}

// Vectorized kernel
template <typename T, unsigned TPB, int ILP>
__global__ void SkipLayerNormKernelVec(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma,
    const T* bias, const T epsilon, T* output, bool hasBias) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  hipcub::KeyValuePair<T, T> thread_data(0, 0);

  using VecT = aligned_vector<T, ILP>;
  T input_v[ILP], skip_v[ILP], bias_v[ILP];
  if (threadIdx.x * ILP < ld) {
    VecT* input_val = reinterpret_cast<VecT*>(&input_v);
    VecT* skip_val = reinterpret_cast<VecT*>(&skip_v);

    for (int i = threadIdx.x * ILP; i < ld; i += TPB * ILP) {
      int idx = offset + i;

      *input_val = *reinterpret_cast<const VecT*>(&input[idx]);
      *skip_val = *reinterpret_cast<const VecT*>(&skip[idx]);
      if (hasBias) {
        VecT* bias_val = reinterpret_cast<VecT*>(&bias_v);
        *bias_val = *reinterpret_cast<const VecT*>(&bias[i]);
      }

      #pragma unroll
      for (int k = 0; k < ILP; k++) {
        input_v[k] += hasBias ? skip_v[k] + bias_v[k] : skip_v[k];
        const T rldval = reverse_ld * input_v[k];
        thread_data = pair_sum(thread_data, hipcub::KeyValuePair<T, T>(rldval, rldval * input_v[k]));
      }
      *(reinterpret_cast<VecT*>(&output[idx])) = *reinterpret_cast<VecT*>(&input_v[0]);
    }
  }

  LayerNormVec<T, TPB, ILP>(thread_data, ld, offset, beta, gamma, epsilon, output);
}

// Vectorized kernel
template <typename T, unsigned TPB, int ILP>
__global__ void SkipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma,
    const T* bias, const T epsilon, T* output, bool hasBias) {
  const T rld = T(1.f / ld);
  const int idx = blockIdx.x * ld + threadIdx.x * ILP;  // grid_size = n / ld

  using VecT = aligned_vector<T, ILP>;
  T input_v[ILP], skip_v[ILP], bias_v[ILP];

  hipcub::KeyValuePair<T, T> thread_data(T(0.f), T(0.f));

  if (ILP * threadIdx.x < ld) {
    VecT* input_val = reinterpret_cast<VecT*>(&input_v);
    *input_val = *reinterpret_cast<const VecT*>(&input[idx]);

    VecT* skip_val = reinterpret_cast<VecT*>(&skip_v);
    *skip_val = *reinterpret_cast<const VecT*>(&skip[idx]);

    if (hasBias) {
      VecT* bias_val = reinterpret_cast<VecT*>(&bias_v);
      *bias_val = *reinterpret_cast<const VecT*>(&bias[threadIdx.x * ILP]);
    }

    T rldval_sum = T(0.f);
    T rldvalsq_sum = T(0.f);
    #pragma unroll
    for (int i = 0; i < ILP; i++) {
      input_v[i] += hasBias ? skip_v[i] + bias_v[i] : skip_v[i];
      const T rldval = rld * input_v[i];
      rldval_sum += rldval;
      rldvalsq_sum += rldval * input_v[i];
    }
    thread_data = hipcub::KeyValuePair<T, T>(rldval_sum, rldvalsq_sum);
  }
  LayerNormSmall<T, TPB, ILP>(input_v, thread_data, ld, idx, beta, gamma, epsilon, output);
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
