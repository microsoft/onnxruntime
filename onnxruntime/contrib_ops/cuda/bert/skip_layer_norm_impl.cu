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
    const int ld, const T* input, const T* skip,
    const T* beta, const T* gamma, const T* bias,
    const T epsilon, T* output, T* skip_input_add_output) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;

    if (skip_input_add_output != nullptr) {
      skip_input_add_output[idx] = input[idx] + skip[idx];
    }

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
    const T* bias, const T epsilon, T* output, T* skip_input_add_output,
    bool hasBias, bool hasSkipInputAdditionOutput) {
  const T rld = T(1.f / ld);
  const int idx = blockIdx.x * ld + threadIdx.x * ILP;  // grid_size = n / ld

  using VecT = aligned_vector<T, ILP>;

  T input_v[ILP], skip_v[ILP], bias_v[ILP], skip_input_add_output_v[ILP];

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
      if (hasSkipInputAdditionOutput) {
        skip_input_add_output_v[i] = input_v[i] + skip_v[i];
      }

      input_v[i] += hasBias ? skip_v[i] + bias_v[i] : skip_v[i];
      const T rldval = rld * input_v[i];
      rldval_sum += rldval;
      rldvalsq_sum += rldval * input_v[i];
    }

    if (hasSkipInputAdditionOutput) {
      *(reinterpret_cast<VecT*>(&skip_input_add_output[idx])) = *reinterpret_cast<VecT*>(&skip_input_add_output_v);
    }

    thread_data = cub::KeyValuePair<T, T>(rldval_sum, rldvalsq_sum);
  }
  LayerNormSmall<T, TPB, ILP>(input_v, thread_data, ld, idx, beta, gamma, epsilon, output);
}

template <typename T>
Status LaunchSkipLayerNormKernel(
    cudaStream_t stream, T* output, T* skip_input_add_output, const T* input, const T* skip, const T* gamma,
    const T* beta, const T* bias, float epsilon, const int ld, const int element_count,
    size_t element_size) {
  // this must be true because n is the total size of the tensor
  assert(element_count % ld == 0);
  bool hasBias = (bias == nullptr) ? false : true;
  bool hasSkipInputAdditionOutput = (skip_input_add_output == nullptr) ? false : true;

  if (0 == (ld % 4)) {
    const int grid_size = element_count / ld;
    if (ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall<T, block_size, 1>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output,
                                                 skip_input_add_output, hasBias, hasSkipInputAdditionOutput);
    } else if (ld <= 64) {
      constexpr int block_size = 64 / 2;
      SkipLayerNormKernelSmall<T, block_size, 2>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output,
                                                 skip_input_add_output, hasBias, hasSkipInputAdditionOutput);
    } else if (ld <= 128) {
      constexpr int block_size = 128 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output,
                                                 skip_input_add_output, hasBias, hasSkipInputAdditionOutput);
    } else if (ld <= 384) {
      constexpr int block_size = 384 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output,
                                                 skip_input_add_output, hasBias, hasSkipInputAdditionOutput);
    } else if (ld <= 768) {
      constexpr int block_size = 768 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output,
                                                 skip_input_add_output, hasBias, hasSkipInputAdditionOutput);
    } else if (ld <= 1024) {
      constexpr int block_size = 1024 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output,
                                                 skip_input_add_output, hasBias, hasSkipInputAdditionOutput);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<T, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output, skip_input_add_output);
    }
  } else {
    const int grid_size = element_count / ld;
    if (ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall<T, block_size, 1>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output,
                                                 skip_input_add_output, hasBias, hasSkipInputAdditionOutput);
    } else if (ld <= 64) {
      constexpr int block_size = 64;
      SkipLayerNormKernelSmall<T, block_size, 1>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output,
                                                 skip_input_add_output, hasBias, hasSkipInputAdditionOutput);
    } else if (ld <= 128) {
      constexpr int block_size = 128;
      SkipLayerNormKernelSmall<T, block_size, 1>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output,
                                                 skip_input_add_output, hasBias, hasSkipInputAdditionOutput);
    } else if (ld == 384) {
      constexpr int block_size = 384;
      SkipLayerNormKernelSmall<T, block_size, 1>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output,
                                                 skip_input_add_output, hasBias, hasSkipInputAdditionOutput);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<T, block_size>
          <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias,
                                                 maybe2half<T>(epsilon), output, skip_input_add_output);
    }
  }
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchSkipLayerNormKernel<float>(cudaStream_t stream, float* output, float* skip_input_add_output,
                                                 const float* input,
                                                 const float* skip, const float* gamma, const float* beta,
                                                 const float* bias, float epsilon, const int ld,
                                                 const int element_count, size_t element_size);

template Status LaunchSkipLayerNormKernel<half>(cudaStream_t stream, half* output, half* skip_input_add_output,
                                                const half* input,
                                                const half* skip, const half* gamma, const half* beta,
                                                const half* bias, float epsilon, const int ld,
                                                const int element_count, size_t element_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
