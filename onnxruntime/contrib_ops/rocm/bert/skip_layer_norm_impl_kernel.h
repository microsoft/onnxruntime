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

template <typename T, typename U, typename V, unsigned TPB>
__global__ void SkipLayerNormKernel(
    const int ld, const T* input, const T* skip, const V* beta, const V* gamma, const T* bias,
    const U epsilon, V* output, T* skip_input_bias_add_output) {
  const U reverse_ld = U(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  hipcub::KeyValuePair<U, U> thread_data(U(0.f), U(0.f));

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const U val = (bias == nullptr) ? static_cast<U>(input[idx]) + static_cast<U>(skip[idx]) : static_cast<U>(input[idx]) + static_cast<U>(skip[idx]) + static_cast<U>(bias[i]);
    const U rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, hipcub::KeyValuePair<U, U>(rldval, rldval * val));

    if (skip_input_bias_add_output != nullptr) {
      skip_input_bias_add_output[idx] = static_cast<T>(val);
    }

    output[idx] = static_cast<V>(val);
  }

  LayerNorm<U, V, TPB>(thread_data, ld, offset, beta, gamma, epsilon, output);
}

// Vectorized kernel
template <typename T, typename U, typename V, unsigned TPB, int ILP>
__global__ void SkipLayerNormKernelVec(
    const int ld, const T* input, const T* skip, const V* beta, const V* gamma,
    const T* bias, const U epsilon, V* output, T* skip_input_bias_add_output,
    bool hasBias, bool hasSkipInputBiasAdditionOutput) {
  const U reverse_ld = U(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  hipcub::KeyValuePair<U, U> thread_data(U(0.f), U(0.f));

  using VecT = aligned_vector<T, ILP>;
  using VecV = aligned_vector<V, ILP>;
  if (threadIdx.x * ILP < ld) {
    for (int i = threadIdx.x * ILP; i < ld; i += TPB * ILP) {
      int idx = offset + i;

      const VecT input_v = *reinterpret_cast<const VecT*>(input + idx);
      const VecT skip_v = *reinterpret_cast<const VecT*>(skip + idx);
      const VecT bias_v = hasBias ? *reinterpret_cast<const VecT*>(bias + i) : VecT();
      VecT skip_input_bias_add_output_v, output_v;

#pragma unroll
      for (int k = 0; k < ILP; k++) {
        const U val = hasBias ? static_cast<U>(input_v.val[k]) + static_cast<U>(skip_v.val[k]) + static_cast<U>(bias_v.val[k]) : static_cast<U>(input_v.val[k]) + static_cast<U>(skip_v.val[k]);
        const U rldval = reverse_ld * val;

        if (hasSkipInputBiasAdditionOutput) {
          skip_input_bias_add_output_v.val[k] = static_cast<T>(val);
        }
        thread_data = pair_sum(thread_data, hipcub::KeyValuePair<U, U>(rldval, rldval * val));
        output_v.val[k] = static_cast<V>(val);
      }

      if (hasSkipInputBiasAdditionOutput) {
        *(reinterpret_cast<VecT*>(skip_input_bias_add_output + idx)) = skip_input_bias_add_output_v;
      }

      *(reinterpret_cast<VecV*>(output + idx)) = output_v;
    }
  }

  LayerNormVec<U, V, TPB, ILP>(thread_data, ld, offset, beta, gamma, epsilon, output);
}

// Vectorized kernel
template <typename T, typename U, typename V, unsigned TPB, int ILP>
__global__ void SkipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const V* beta, const V* gamma,
    const T* bias, const U epsilon, V* output, T* skip_input_bias_add_output,
    bool hasBias, bool hasSkipInputBiasAdditionOutput) {
  const U rld = U(1.f / ld);
  const int idx = blockIdx.x * ld + threadIdx.x * ILP;  // grid_size = n / ld

  using VecT = aligned_vector<T, ILP>;
  hipcub::KeyValuePair<U, U> thread_data(U(0.f), U(0.f));

  VecT input_v;
  if (ILP * threadIdx.x < ld) {
    input_v = *reinterpret_cast<const VecT*>(input + idx);
    const VecT skip_v = *reinterpret_cast<const VecT*>(skip + idx);
    const VecT bias_v = hasBias ? *reinterpret_cast<const VecT*>(bias + threadIdx.x * ILP) : VecT();
    VecT skip_input_bias_add_output_v;

    U rldval_sum = U(0.f);
    U rldvalsq_sum = U(0.f);
#pragma unroll
    for (int i = 0; i < ILP; i++) {
      const U val = hasBias ? static_cast<U>(input_v.val[i]) + static_cast<U>(skip_v.val[i]) + static_cast<U>(bias_v.val[i]) : static_cast<U>(input_v.val[i]) + static_cast<U>(skip_v.val[i]);

      if (hasSkipInputBiasAdditionOutput) {
        skip_input_bias_add_output_v.val[i] = static_cast<T>(val);
      }

      const U rldval = rld * val;
      rldval_sum += rldval;
      rldvalsq_sum += rldval * val;
      input_v.val[i] = static_cast<T>(val);
    }

    if (hasSkipInputBiasAdditionOutput) {
      *(reinterpret_cast<VecT*>(skip_input_bias_add_output + idx)) = skip_input_bias_add_output_v;
    }

    thread_data = hipcub::KeyValuePair<U, U>(rldval_sum, rldvalsq_sum);
  }
  LayerNormSmall<T, U, V, TPB, ILP>(input_v.val, thread_data, ld, idx, beta, gamma, epsilon, output);
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
