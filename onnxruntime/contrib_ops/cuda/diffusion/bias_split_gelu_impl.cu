// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The CUDA kernel is modified from SplitGelu plugin of TensorRT 8.5.
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
#include "contrib_ops/cuda/diffusion/bias_split_gelu_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, int32_t HHS, int32_t TPB>
__global__ void biasSplitGeluKernel(T const* input, T const* bias, T* output) {
  int32_t index_input = blockIdx.x * HHS * 2 + threadIdx.x;
  int32_t index_output = blockIdx.x * HHS + threadIdx.x;
  int32_t index_bias = threadIdx.x;

#pragma unroll
  for (int32_t i = 0; i < HHS / TPB; ++i) {
    auto value_left = (float)(input[index_input] + bias[index_bias]);
    auto value_right = (float)(input[index_input + HHS] + bias[index_bias + HHS]);

    // Gelu is applied to right side only: Gelu(x) = x * 0.5 * (erf(x / sqrt(2)) + 1.0)
    float gelu_right = value_right * 0.5f * (erff(value_right / static_cast<float>(M_SQRT2)) + 1.0f);
    float result = value_left * gelu_right;
    output[index_output] = static_cast<T>(result);
    index_input += TPB;
    index_output += TPB;
    index_bias += TPB;
  }
  return;
}

template <typename T>
void LaunchBiasSplitGeluKernel(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size,
                               T const* input, T const* bias, T* output) {
  constexpr int32_t TPB = 256;  // thread per block
  switch (half_hidden_size) {
    case 1280:
      (biasSplitGeluKernel<T, 1280, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
      break;
    case 2560:
      (biasSplitGeluKernel<T, 2560, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
      break;
    case 5120:
      (biasSplitGeluKernel<T, 5120, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
      break;
    default:
      ORT_NOT_IMPLEMENTED("Not implemented");
  }
}

template __global__ void biasSplitGeluKernel<float, 1280, 256>(float const*, float const*, float*);
template __global__ void biasSplitGeluKernel<float, 2560, 256>(float const*, float const*, float*);
template __global__ void biasSplitGeluKernel<float, 5120, 256>(float const*, float const*, float*);
template __global__ void biasSplitGeluKernel<half, 1280, 256>(half const*, half const*, half*);
template __global__ void biasSplitGeluKernel<half, 2560, 256>(half const*, half const*, half*);
template __global__ void biasSplitGeluKernel<half, 5120, 256>(half const*, half const*, half*);

template void LaunchBiasSplitGeluKernel<float>(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size,
                                               float const* input, float const* bias, float* output);

template void LaunchBiasSplitGeluKernel<half>(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size,
                                              half const* input, half const* bias, half* output);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
