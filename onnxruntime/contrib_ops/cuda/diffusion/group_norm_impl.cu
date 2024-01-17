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

// The CUDA kernel is modified from GroupNorm plugin of TensorRT 8.5
// Modifications: heuristic channels per block; support epsilon; support skip and bias; update coding style.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/diffusion/group_norm_impl.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "contrib_ops/cuda/diffusion/group_norm_common_base.h"
#include "contrib_ops/cuda/diffusion/group_norm_impl_kernel.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void GroupNormNHWCSum(GroupNormNHWCParams<T> const& params, cudaStream_t stream) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = DivUp(params.c, params.channels_per_block);

  // The number of blocks to compute all the activations in a given instance.
  grid.y = DivUp(params.hw, params.hw_per_block);

  // The number of instances.
  grid.z = params.n;

#define LAUNCH_GROUPNORM_SUM(ThreadsPerBlock, VecSize)                                               \
  GroupNormNHWCSumKernel<T, ThreadsPerBlock, VecSize>                                                \
      <<<grid, ThreadsPerBlock, 0, stream>>>(                                                        \
          params.skip_workspace, params.group_sum_buffer, params.src, params.skip, params.bias,       \
          params.channels_per_block, params.hw_per_block, params.hw, params.hwc, params.c,           \
          params.channels_per_group, params.groups, params.groups_per_block, params.broadcast_skip); \
  break;

  // Threads_per_block is half of values in kSizes since CHANNELS_PER_THREAD = 2.
  switch (params.threads_per_block) {
    case 256:
      LAUNCH_GROUPNORM_SUM(256, CHANNELS_PER_THREAD)
    case 192:
      LAUNCH_GROUPNORM_SUM(192, CHANNELS_PER_THREAD)
    case 160:
      LAUNCH_GROUPNORM_SUM(160, CHANNELS_PER_THREAD)
    case 128:
      LAUNCH_GROUPNORM_SUM(128, CHANNELS_PER_THREAD)
    case 64:
      LAUNCH_GROUPNORM_SUM(64, CHANNELS_PER_THREAD)
  }
}

template <typename T>
void GroupNormNHWCScale(GroupNormNHWCParams<T> const& params, cudaStream_t stream) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = DivUp(params.c, params.channels_per_block);
  // The number of blocks to compute all the activations in a given instance.
  grid.y = DivUp(params.hw, params.hw_per_block);
  // The number of instances.
  grid.z = params.n;

#define LAUNCH_GROUPNORM_SCALE(ThreadsPerBlock, VecSize)                                                           \
  GroupNormNHWCScaleKernel<T, VecSize>                                                                             \
      <<<grid, ThreadsPerBlock, 0, stream>>>(                                                                      \
          params.dst, params.src, params.skip, params.gamma, params.beta, params.skip_workspace,                   \
          params.group_sum_buffer, params.epsilon, params.c, params.channels_per_block, params.channels_per_group, \
          params.groups, params.hwc, params.inv_hw_channels_per_group, params.hw, params.hw_per_block,             \
          params.use_silu);                                                                                        \
  break;

  // Threads_per_block is half of values in kSizes since CHANNELS_PER_THREAD = 2.
  switch (params.threads_per_block) {
    case 256:
      LAUNCH_GROUPNORM_SCALE(256, CHANNELS_PER_THREAD)
    case 192:
      LAUNCH_GROUPNORM_SCALE(192, CHANNELS_PER_THREAD)
    case 160:
      LAUNCH_GROUPNORM_SCALE(160, CHANNELS_PER_THREAD)
    case 128:
      LAUNCH_GROUPNORM_SCALE(128, CHANNELS_PER_THREAD)
    case 64:
      LAUNCH_GROUPNORM_SCALE(64, CHANNELS_PER_THREAD)
  }
}

template <typename T>
Status LaunchGroupNormKernel(
    CudaTuningContext* tuning_ctx,
    Stream* ort_stream,
    T* output,
    T* add_out,
    const T* input,
    const T* skip,
    const T* bias,
    const float* gamma,
    const float* beta,
    void* workspace,
    float epsilon,
    int batch_size,
    int num_channels,
    int height,
    int width,
    int num_groups,
    bool use_silu,
    bool broadcast_skip,
    int channels_per_block) {

  // tuning_ctx only used for ROCm EP.
  ORT_UNUSED_PARAMETER(tuning_ctx);

  GroupNormNHWCParams<T> params(output, add_out, input, skip, bias, gamma, beta, workspace, epsilon,
                                batch_size, num_channels, height, width, num_groups, use_silu,
                                broadcast_skip, channels_per_block);

  if (params.channels_per_block % params.channels_per_group != 0 ||
      params.channels_per_block > kMaxSize ||
      (params.channels_per_group % CHANNELS_PER_THREAD != 0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "GroupNorm in CUDA does not support the input: n=", batch_size,
                           " h=", height,
                           " w=", width,
                           " c=", num_channels,
                           " groups=", num_groups);
  }

  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
      params.group_sum_buffer, 0, GetGroupNormWorkspaceSizeInBytes(batch_size, num_groups), stream));

  GroupNormNHWCSum<T>(params, stream);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("workspace", params.group_sum_buffer, batch_size, 2, num_groups);

  GroupNormNHWCScale<T>(params, stream);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  return Status::OK();
}

template Status LaunchGroupNormKernel<half>(CudaTuningContext* tuning_ctx, Stream* stream, half* output, half* add_out,
                                            const half* input, const half* skip, const half* bias,
                                            const float* gamma, const float* beta, void* workspace,
                                            float epsilon, int batch_size, int num_channels,
                                            int height, int width, int num_groups, bool silu,
                                            bool broadcast_skip, int channels_per_block);

template Status LaunchGroupNormKernel<float>(CudaTuningContext* tuning_ctx, Stream* stream, float* output, float* add_out,
                                             const float* input, const float* skip, const float* bias,
                                             const float* gamma, const float* beta, void* workspace,
                                             float epsilon, int batch_size, int num_channels,
                                             int height, int width, int num_groups, bool silu,
                                             bool broadcast_skip, int channels_per_block);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
