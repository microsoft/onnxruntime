// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The CUDA kernel is modified from SeqLen2Spatial plugin of TensorRT 8.5.
/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cub/cub.cuh>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/diffusion/bias_add_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, int32_t C, int32_t TPB>
__global__ void BiasAddKernel(T const* input, T const* bias, T const* residual, T* output) {
  int32_t base_offset = blockIdx.x * C + threadIdx.x;
  int32_t bias_offset = threadIdx.x;

#pragma unroll
  for (int32_t i = 0; i < C / TPB; ++i) {
    output[base_offset] = input[base_offset] + bias[bias_offset] + residual[base_offset];
    base_offset += TPB;
    bias_offset += TPB;
  }
}

template <typename T, unsigned TPB>
__global__ void BiasAddLargeKernel(
    int32_t const ld, const T* input, const T* bias, const T* residual, T* output) {
  int32_t const offset = blockIdx.x * ld;

  for (int32_t i = threadIdx.x; i < ld; i += TPB) {
    int32_t const base_offset = offset + i;
    output[base_offset] = input[base_offset] + bias[i] + residual[base_offset];
  }
}

template __global__ void BiasAddKernel<float, 320, 320>(float const*, float const*, float const*, float*);
template __global__ void BiasAddKernel<float, 640, 320>(float const*, float const*, float const*, float*);
template __global__ void BiasAddKernel<float, 1280, 320>(float const*, float const*, float const*, float*);
template __global__ void BiasAddKernel<half, 320, 320>(half const*, half const*, half const*, half*);
template __global__ void BiasAddKernel<half, 640, 320>(half const*, half const*, half const*, half*);
template __global__ void BiasAddKernel<half, 1280, 320>(half const*, half const*, half const*, half*);

template <typename T>
void LaunchBiasAddKernel(cudaStream_t stream, int32_t grid_size, int32_t num_channels,
                         T const* input, T const* bias, T const* residual, T* output) {
  switch (num_channels) {
    case 320:
      (BiasAddKernel<T, 320, 320>)<<<grid_size, 320, 0, stream>>>(input, bias, residual, output);
      break;
    case 640:
      (BiasAddKernel<T, 640, 320>)<<<grid_size, 320, 0, stream>>>(input, bias, residual, output);
      break;
    case 1280:
      (BiasAddKernel<T, 1280, 320>)<<<grid_size, 320, 0, stream>>>(input, bias, residual, output);
      break;
    default:
      BiasAddLargeKernel<T, 256><<<grid_size, 256, 0, stream>>>(num_channels, input, bias, residual, output);
      break;
  }
}

template void LaunchBiasAddKernel<float>(cudaStream_t stream, int32_t grid_size, int32_t num_channels,
                                         float const* input, float const* bias, float const* residual, float* output);

template void LaunchBiasAddKernel<half>(cudaStream_t stream, int32_t grid_size, int32_t num_channels,
                                        half const* input, half const* bias, half const* residual, half* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
