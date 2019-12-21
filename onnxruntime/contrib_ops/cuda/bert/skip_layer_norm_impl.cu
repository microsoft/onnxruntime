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

#include "layer_norm.cuh"
#include "skip_layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, unsigned TPB, bool hasBias>
__global__ void SkipLayerNormKernelSmall(const int ld,
                                         const T* input,
                                         const T* skip,
                                         const T* beta,
                                         const T* gamma,
                                         const T* bias,
                                         T* output) {
  const T reverse_ld = T(1) / T(ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;

  if (threadIdx.x < ld) {
    val = hasBias ? input[idx] + skip[idx] + bias[threadIdx.x] : input[idx] + skip[idx];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
  }

  LayerNormSmall<T, TPB>(val, thread_data, ld, idx, beta, gamma, output);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void SkipLayerNormKernel(const int ld,
                                    const T* input,
                                    const T* skip,
                                    const T* beta,
                                    const T* gamma,
                                    const T* bias,
                                    T* output) {
  const T reverse_ld = T(1) / T(ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    T val = hasBias ? input[idx] + skip[idx] + bias[i] : input[idx] + skip[idx];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
    output[idx] = val;
  }

  LayerNorm<T, TPB>(thread_data, ld, offset, beta, gamma, output);
}

#define InvokeSkipLayerNormKernel(func, block_size)     \
  if (bias != nullptr) {                                \
    func<T, block_size, true>                           \
        <<<grid_size, block_size, 0, stream>>>(ld,      \
                                               input,   \
                                               skip,    \
                                               beta,    \
                                               gamma,   \
                                               bias,    \
                                               output); \
  } else {                                              \
    func<T, block_size, false>                          \
        <<<grid_size, block_size, 0, stream>>>(ld,      \
                                               input,   \
                                               skip,    \
                                               beta,    \
                                               gamma,   \
                                               bias,    \
                                               output); \
  }

template <typename T>
bool ComputeSkipLayerNorm(cudaStream_t stream,
                          const int ld,
                          const int n,
                          const T* input,
                          const T* skip,
                          const T* beta,
                          const T* gamma,
                          const T* bias,
                          T* output) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  const int grid_size = n / ld;

  if (ld <= 32) {
    InvokeSkipLayerNormKernel(SkipLayerNormKernelSmall, 32)
  } else if (ld <= 128) {
    InvokeSkipLayerNormKernel(SkipLayerNormKernelSmall, 128)
  } else if (ld == 384) {
    InvokeSkipLayerNormKernel(SkipLayerNormKernelSmall, 384)
  } else {
    InvokeSkipLayerNormKernel(SkipLayerNormKernel, 256)
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchSkipLayerNormKernel(void* output,
                               const void* input,
                               const void* skip,
                               const void* gamma,
                               const void* beta,
                               const void* bias,
                               const int batch_size,
                               const int hidden_size,
                               const int element_count,
                               const size_t element_size) {
  // use default stream
  const cudaStream_t stream = nullptr;

  if (element_size == 2) {
    return ComputeSkipLayerNorm(
        stream,
        hidden_size,
        element_count,
        reinterpret_cast<const half*>(input),
        reinterpret_cast<const half*>(skip),
        reinterpret_cast<const half*>(beta),
        reinterpret_cast<const half*>(gamma),
        reinterpret_cast<const half*>(bias),
        reinterpret_cast<half*>(output));
  } else {
    return ComputeSkipLayerNorm(
        stream,
        hidden_size,
        element_count,
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(skip),
        reinterpret_cast<const float*>(beta),
        reinterpret_cast<const float*>(gamma),
        reinterpret_cast<const float*>(bias),
        reinterpret_cast<float*>(output));
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
