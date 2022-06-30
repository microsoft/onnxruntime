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

// Modifications: Add SkipLayerNormKernelVec to
//                leverage vectorized load/write.
//                and templatize ComputeSkipLayerNorm for different
//                data types.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/layer_norm.cuh"
#include "contrib_ops/cuda/bert/skip_layer_norm_impl.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

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

// Vectorized kernel
template <typename T, unsigned TPB, int ILP>
__global__ void SkipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma,
    const T* bias, const T epsilon, T* output, bool hasBias) {
  const T rld = T(1.f / ld);
  const int idx = blockIdx.x * ld + threadIdx.x * ILP;  // grid_size = n / ld

  using VecT = aligned_vector<T, ILP>;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  T input_v[ILP], skip_v[ILP], bias_v[ILP], output_v[ILP];

  VecT* input_val = reinterpret_cast<VecT*>(&input_v);
  *input_val = *reinterpret_cast<const VecT*>(&input[idx]);

  VecT* skip_val = reinterpret_cast<VecT*>(&skip_v);
  *skip_val = *reinterpret_cast<const VecT*>(&skip[idx]);

  if (hasBias) {
    VecT* bias_val = reinterpret_cast<VecT*>(&bias_v);
    *bias_val = *reinterpret_cast<const VecT*>(&bias[threadIdx.x * ILP]);
  }

  cub::KeyValuePair<T, T> thread_data(T(0.f), T(0.f));

  if (ILP * threadIdx.x < ld) {
    T rldval_sum = T(0.f);
    T rldvalsq_sum = T(0.f);
    #pragma unroll
    for (int i = 0; i < ILP; i++) {
      input_v[i] += hasBias ? skip_v[i] + bias_v[i]: skip_v[i];
      const T rldval = rld * input_v[i];
      rldval_sum += rldval;
      rldvalsq_sum += rldval * input_v[i];
    }
    thread_data = cub::KeyValuePair<T, T>(rldval_sum, rldvalsq_sum);
  }
  LayerNormSmall<T, TPB, ILP>(input_v, thread_data, ld, idx, beta, gamma, epsilon, output);
}

/* half16 and float32 */
template <typename T>
bool ComputeSkipLayerNorm(
    cudaStream_t stream, const int ld, const int n, const T* input, const T* skip, const T* beta,
    const T* gamma, const T* bias, const T epsilon, T* output) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  bool hasBias = (bias == nullptr) ? false : true;
  if (0 == (n % 4)) {
    const int grid_size = n / ld;
    if (ld <= 32) {
      constexpr int block_size = 32 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4>
         <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld <= 64) {
      constexpr int block_size = 64 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld <= 128) {
      constexpr int block_size = 128 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld <= 384) {
      constexpr int block_size = 384 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld <= 768) {
      constexpr int block_size = 768 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld <= 1024) {
      constexpr int block_size = 1024 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<T, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  } else {
    const int grid_size = n / ld;
    if (ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall<T, block_size, 1>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld <= 64) {
      constexpr int block_size = 64;
      SkipLayerNormKernelSmall<T, block_size, 1>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld <= 128) {
      constexpr int block_size = 128;
      SkipLayerNormKernelSmall<T, block_size, 1>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else if (ld == 384) {
      constexpr int block_size = 384;
      SkipLayerNormKernelSmall<T, block_size, 1>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output, hasBias);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<T, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
    }
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchSkipLayerNormKernel(
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
    size_t element_size) {
  if (element_size == 2) {
    return ComputeSkipLayerNorm<half>(
        stream,
        hidden_size,
        element_count,
        reinterpret_cast<const half*>(input),
        reinterpret_cast<const half*>(skip),
        reinterpret_cast<const half*>(beta),
        reinterpret_cast<const half*>(gamma),
        reinterpret_cast<const half*>(bias),
        __float2half_rn(epsilon),
        reinterpret_cast<half*>(output));
  } else {
    return ComputeSkipLayerNorm<float>(
        stream,
        hidden_size,
        element_count,
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(skip),
        reinterpret_cast<const float*>(beta),
        reinterpret_cast<const float*>(gamma),
        reinterpret_cast<const float*>(bias),
        epsilon,
        reinterpret_cast<float*>(output));
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

