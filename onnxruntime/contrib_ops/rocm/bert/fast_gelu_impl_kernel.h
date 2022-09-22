// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/tunable/util.h"
#include "core/providers/rocm/cu_inc/common.cuh"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T, unsigned TPB>
__global__ void FastGeluKernel(int input_length, int bias_length, const T* input, const T* bias, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  // constants for approximating the normal cdf
  const T a = T(0.5f);
  const T b = T(0.7978845608028654f);  // sqrt(2.0/M_PI)
  const T c = T(0.035677408136300125f);  // 0.044715 * sqrt(2.0/M_PI)
  const T oneT = T(1.0f);
  const T twoT = T(2.0f);
  if (idx < input_length) {
    const T x = input[idx];
    const T in = (bias == nullptr) ? x : (x + bias[idx % bias_length]);

    // const T cdf = a + a * _Tanh(in * (c * in * in + b));
    const T u = twoT * in * (c * in * in + b);
    const T emu = __expf(-u);
    const T cdf = a + a * (twoT/(oneT + emu) - oneT);

    output[idx] = in * cdf;
  }
}

template <typename T, unsigned TPB, int ILP>
__global__ void FastGeluKernelVec(int input_length, int bias_length, const T* input, const T* bias,
                                  T* output) {
  using VecT = onnxruntime::rocm::aligned_vector<T, ILP>;
  const T a = T(0.5f);
  const T b = T(0.7978845608028654f);
  const T c = T(0.035677408136300125f);
  const T oneT = T(1.0f);
  const T twoT = T(2.0f);

  const int idx = (blockIdx.x * TPB + threadIdx.x) * ILP;
  if (idx < input_length) {
    T input_v[ILP];
    VecT* input_val = reinterpret_cast<VecT*>(&input_v);
    *input_val = *reinterpret_cast<const VecT*>(&input[idx]);
    T output_v[ILP];
    VecT* output_val = reinterpret_cast<VecT*>(&output_v);
    T bias_v[ILP];
    if (bias != nullptr) {
        VecT* bias_val = reinterpret_cast<VecT*>(&bias_v);
        *bias_val = *reinterpret_cast<const VecT*>(&bias[idx % bias_length]);
    }

    #pragma unroll
    for (int i = 0; i < ILP; i++) {
      const T x = (bias == nullptr) ? input_v[i] : (T)(input_v[i] + bias_v[i]);
      const T u = twoT * x * (c * x * x + b);
      const T emu = __expf(-u);
      const T cdf = a + a * (twoT/(oneT + emu) - oneT);
      output_v[i] = x * cdf;
    }
    *(reinterpret_cast<VecT*>(&output[idx])) = *output_val;
  }
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
