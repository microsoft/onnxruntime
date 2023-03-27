#include "hip/hip_runtime.h"
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

// The ROCM kernel is hipified from CUDA kernel.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "contrib_ops/rocm/diffusion/group_norm_common.h"
#include "contrib_ops/rocm/diffusion/group_norm_impl.h"
#include "contrib_ops/rocm/diffusion/group_norm_impl_kernel.h"
#include "contrib_ops/rocm/transformers/dump_rocm_tensor.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
void groupNormNHWCSum(const GroupNormNHWCParams<T>* params) {
  // Make sure the values are as we expect.
  ORT_ENFORCE(params->c % params->cPerBlock == 0 && params->hw % params->hwPerBlock == 0);
  // Make sure a group does not span multiple blocks.
  ORT_ENFORCE(params->cPerBlock % params->cPerGroup == 0);

  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params->c / params->cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params->hw, params->hwPerBlock);
  // The number of instances.
  grid.z = params->n;

  switch (params->cPerBlock) {
    case 320:
      groupNormNHWCSumKernel<T, 256, 2><<<grid, 256, 0, params->stream>>>(params->src, params->redBuffer, params->cPerBlock, params->hwPerBlock,
                                                                          params->hw, params->hwc, params->c, params->cPerGroup, params->groups, params->groupsPerBlock);
      break;
    case 480:
      groupNormNHWCSumKernel<T, 256, 2><<<grid, 256, 0, params->stream>>>(params->src, params->redBuffer, params->cPerBlock, params->hwPerBlock,
                                                                          params->hw, params->hwc, params->c, params->cPerGroup, params->groups, params->groupsPerBlock);
      break;
    case 256:
      groupNormNHWCSumKernel<T, 128, 2><<<grid, 128, 0, params->stream>>>(params->src, params->redBuffer, params->cPerBlock, params->hwPerBlock,
                                                                          params->hw, params->hwc, params->c, params->cPerGroup, params->groups, params->groupsPerBlock);
      break;
    case 128:
      groupNormNHWCSumKernel<T, 64, 2><<<grid, 64, 0, params->stream>>>(params->src, params->redBuffer, params->cPerBlock, params->hwPerBlock,
                                                                        params->hw, params->hwc, params->c, params->cPerGroup, params->groups, params->groupsPerBlock);
      break;
    default:
      ORT_NOT_IMPLEMENTED("Not implemented");
  }
}

template <typename T, int ThreadsPerBlock, int VecSize>
Status GroupNormNHWCSumOp(const GroupNormNHWCParams<T>* params) {
  dim3 grid;
  grid.x = params->c / params->cPerBlock;
  grid.y = divUp(params->hw, params->hwPerBlock);
  grid.z = params->n;

  groupNormNHWCSumKernel<T, ThreadsPerBlock, VecSize>
      <<<grid, ThreadsPerBlock, 0, params->stream>>>(
          params->src, params->redBuffer, params->cPerBlock, params->hwPerBlock,
          params->hw, params->hwc, params->c, params->cPerGroup, params->groups, params->groupsPerBlock);
  return HIP_CALL(hipGetLastError());
}

template <typename T>
void groupNormNHWCScale(const GroupNormNHWCParams<T>* params) {
  // Make sure the dimensions are aligned with what we expect.
  ORT_ENFORCE(params->c % params->cPerBlock == 0);
  // Make sure a group does not span multiple blocks.
  ORT_ENFORCE(params->cPerBlock % params->cPerGroup == 0);

  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params->c / params->cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params->hw, params->hwPerBlock);
  // The number of instances.
  grid.z = params->n;

  switch (params->cPerBlock) {
    case 320:
      groupNormNHWCScaleKernel<T, 256, 2><<<grid, 256, 0, params->stream>>>(params->dst, params->src, params->gamma, params->beta, params->redBuffer, params->c, params->cPerBlock,
                                                                            params->cPerGroup, params->groups, params->hwc, params->invHWC, params->hw, params->hwPerBlock, params->withSwish);
      break;
    case 480:
      groupNormNHWCScaleKernel<T, 256, 2><<<grid, 256, 0, params->stream>>>(params->dst, params->src, params->gamma, params->beta, params->redBuffer, params->c, params->cPerBlock,
                                                                            params->cPerGroup, params->groups, params->hwc, params->invHWC, params->hw, params->hwPerBlock, params->withSwish);
      break;
    case 256:
      groupNormNHWCScaleKernel<T, 128, 2><<<grid, 128, 0, params->stream>>>(params->dst, params->src, params->gamma, params->beta, params->redBuffer, params->c, params->cPerBlock,
                                                                            params->cPerGroup, params->groups, params->hwc, params->invHWC, params->hw, params->hwPerBlock, params->withSwish);
      break;
    case 128:
      groupNormNHWCScaleKernel<T, 64, 2><<<grid, 64, 0, params->stream>>>(params->dst, params->src, params->gamma, params->beta, params->redBuffer, params->c, params->cPerBlock,
                                                                          params->cPerGroup, params->groups, params->hwc, params->invHWC, params->hw, params->hwPerBlock, params->withSwish);
      break;
    default:
      ORT_NOT_IMPLEMENTED("Not implemented");
  }
}

template <typename T, int ThreadsPerBlock, int VecSize>
Status GroupNormNHWCScaleOp(const GroupNormNHWCParams<T>* params) {
  dim3 grid;
  grid.x = params->c / params->cPerBlock;
  grid.y = divUp(params->hw, params->hwPerBlock);
  grid.z = params->n;

  groupNormNHWCScaleKernel<T, ThreadsPerBlock, VecSize>
      <<<grid, ThreadsPerBlock, 0, params->stream>>>(
          params->dst, params->src, params->gamma, params->beta, params->redBuffer, params->c, params->cPerBlock,
          params->cPerGroup, params->groups, params->hwc, params->invHWC, params->hw, params->hwPerBlock, params->withSwish);
  return HIP_CALL(hipGetLastError());
}

template <typename T, int ThreadsPerBlock, int VecSize>
class GroupNormNHWCOp {
 public:
  Status operator()(const GroupNormNHWCParams<T>* params) {
    DUMP_TENSOR_INIT();
    DUMP_TENSOR("input", params->src, params->n, params->c, params->h * params->w);
    DUMP_TENSOR("gamma", params->gamma, 1, params->c);
    DUMP_TENSOR("beta", params->beta, 1, params->c);
    HIP_RETURN_IF_ERROR(hipMemsetAsync(params->redBuffer, 0, GetGroupNormWorkspaceSizeInBytes(), params->stream));
    auto status = GroupNormNHWCSumOp<T, ThreadsPerBlock, VecSize>(params);
    ORT_RETURN_IF_ERROR(status);
    DUMP_TENSOR("workspace", params->redBuffer, params->n, params->groups, 2);
    HIP_RETURN_IF_ERROR(hipGetLastError());
    status = GroupNormNHWCScaleOp<T, ThreadsPerBlock, VecSize>(params);
    ORT_RETURN_IF_ERROR(status);
    HIP_RETURN_IF_ERROR(hipGetLastError());
    DUMP_TENSOR("output", params->dst, params->n, params->c, params->h * params->w);
    return Status::OK();
  }

  Status IsSupported(const GroupNormNHWCParams<T>* params) {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        !(params->c % VecSize == 0 && params->cPerGroups % VecSize == 0));
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!(params->cPerBlock % params->cPerGroups == 0 &&
                                                params->cPerBlock <= ThreadsPerBlock * VecSize &&
                                                params->c % params->cPerBlock == 0 &&
                                                params->hw % params->hwPerBlock == 0));

    return Status::OK();
  }
};

template <typename T>
Status LaunchGroupNormKernel(
    hipStream_t stream,
    T* output,
    const T* input,
    const float* gamma,
    const float* beta,
    void* workspace,
    float epsilon,
    int batch_size,
    int num_channels,
    int height,
    int width,
    int num_groups,
    bool use_swish_activation) {
  if (batch_size > static_cast<int>(kMaxGroupNormBatchSize)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED,
                           "only support batch_size <= 32. Got", batch_size);
  }

  if (num_groups != static_cast<int>(kGroupNormNumberOfGroups)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED,
                           "only num_groups=32 is supported. Got", num_groups);
  }

  GroupNormNHWCParams<T> params(nullptr, stream, output, reinterpret_cast<float*>(workspace), input, gamma, beta,
                                batch_size, height, width, num_channels, num_groups, use_swish_activation);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("input", input, batch_size, num_channels, height * width);
  DUMP_TENSOR("gamma", gamma, 1, num_channels);
  DUMP_TENSOR("beta", beta, 1, num_channels);
  HIP_RETURN_IF_ERROR(hipMemsetAsync(params.redBuffer, 0, GetGroupNormWorkspaceSizeInBytes(), stream));
  groupNormNHWCSum<T>(&params);
  DUMP_TENSOR("workspace", params.redBuffer, batch_size, num_groups, 2);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  groupNormNHWCScale<T>(&params);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  DUMP_TENSOR("output", output, batch_size, num_channels, height * width);
  return Status::OK();
}

template Status LaunchGroupNormKernel<half>(hipStream_t stream, half* output,
                                            const half* input, const float* gamma, const float* beta, void* workspace,
                                            float epsilon, int batch_size, int num_channels,
                                            int height, int width, int num_groups, bool swish);

template Status LaunchGroupNormKernel<float>(hipStream_t stream, float* output,
                                             const float* input, const float* gamma, const float* beta, void* workspace,
                                             float epsilon, int batch_size, int num_channels,
                                             int height, int width, int num_groups, bool swish);
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
