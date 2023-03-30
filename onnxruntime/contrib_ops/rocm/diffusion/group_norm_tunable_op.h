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

namespace {
int computeStaticSelectionCPerGroup(int c) {
  int cPerBlock = 320;
  switch (c) {
    case 960:
    case 1920:
      cPerBlock = 480;
      break;
    case 512:
    case 256:
      cPerBlock = 256;
      break;
    case 128:
      cPerBlock = 128;
      break;
    default:
      cPerBlock = 320;
  }
  return cPerBlock;
}
}  // namespace

template <typename T>
void groupNormNHWCSum(const GroupNormNHWCParams<T>* params) {
  int cPerBlock = computeStaticSelectionCPerGroup(params->c);
  // Make sure the values are as we expect.
  ORT_ENFORCE(params->c % cPerBlock == 0 && params->hw % params->hwPerBlock == 0);
  // Make sure a group does not span multiple blocks.
  ORT_ENFORCE(cPerBlock % params->cPerGroup == 0);

  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params->c / cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params->hw, params->hwPerBlock);
  // The number of instances.
  grid.z = params->n;

#define LAUNCH_GROUPNORM_SUM(ThreadsPerBlock, VecSize)                                                       \
  groupNormNHWCSumKernel<T, ThreadsPerBlock, VecSize>                                                        \
      <<<grid, ThreadsPerBlock, 0, params->stream>>>(params->src, params->redBuffer, cPerBlock,              \
                                                     params->hwPerBlock, params->hw, params->hwc, params->c, \
                                                     params->cPerGroup, params->groups);                     \
  break;

  switch (cPerBlock) {
    case 320:
      LAUNCH_GROUPNORM_SUM(256, 2)
    case 480:
      LAUNCH_GROUPNORM_SUM(256, 2)
    case 256:
      LAUNCH_GROUPNORM_SUM(128, 2)
    case 128:
      LAUNCH_GROUPNORM_SUM(64, 2)
    default:
      ORT_NOT_IMPLEMENTED("Not implemented");
  }
}

template <typename T, int CPerBlock, int ThreadsPerBlock, int VecSize>
Status GroupNormNHWCSumOp(const GroupNormNHWCParams<T>* params) {
  dim3 grid;
  grid.x = params->c / CPerBlock;
  grid.y = divUp(params->hw, params->hwPerBlock);
  grid.z = params->n;

  groupNormNHWCSumKernel<T, ThreadsPerBlock, VecSize>
      <<<grid, ThreadsPerBlock, 0, params->stream>>>(
          params->src, params->redBuffer, CPerBlock, params->hwPerBlock,
          params->hw, params->hwc, params->c, params->cPerGroup, params->groups);
  return HIP_CALL(hipGetLastError());
}

template <typename T>
void groupNormNHWCScale(const GroupNormNHWCParams<T>* params) {
  int cPerBlock = computeStaticSelectionCPerGroup(params->c);
  // Make sure the dimensions are aligned with what we expect.
  ORT_ENFORCE(params->c % cPerBlock == 0);
  // Make sure a group does not span multiple blocks.
  ORT_ENFORCE(cPerBlock % params->cPerGroup == 0);

  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params->c / cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params->hw, params->hwPerBlock);
  // The number of instances.
  grid.z = params->n;

#define LAUNCH_GROUPNORM_SCALE(ThreadsPerBlock, VecSize)                                                             \
  groupNormNHWCScaleKernel<T, ThreadsPerBlock, VecSize>                                                              \
      <<<grid, ThreadsPerBlock, 0, params->stream>>>(params->dst, params->src, params->gamma, params->beta,          \
                                                     params->redBuffer, params->epsilon, params->c, cPerBlock,       \
                                                     params->cPerGroup, params->groups, params->hwc, params->invHWC, \
                                                     params->hw, params->hwPerBlock, params->withSwish);             \
  break;

  switch (cPerBlock) {
    case 320:
      LAUNCH_GROUPNORM_SCALE(256, 2)
    case 480:
      LAUNCH_GROUPNORM_SCALE(256, 2)
    case 256:
      LAUNCH_GROUPNORM_SCALE(128, 2)
    case 128:
      LAUNCH_GROUPNORM_SCALE(64, 2)
    default:
      ORT_NOT_IMPLEMENTED("Not implemented");
  }
}

template <typename T, int CPerBlock, int ThreadsPerBlock, int VecSize>
Status GroupNormNHWCScaleOp(const GroupNormNHWCParams<T>* params) {
  dim3 grid;
  grid.x = params->c / CPerBlock;
  grid.y = divUp(params->hw, params->hwPerBlock);
  grid.z = params->n;

  groupNormNHWCScaleKernel<T, ThreadsPerBlock, VecSize>
      <<<grid, ThreadsPerBlock, 0, params->stream>>>(
          params->dst, params->src, params->gamma, params->beta, params->redBuffer, params->epsilon, params->c, CPerBlock,
          params->cPerGroup, params->groups, params->hwc, params->invHWC, params->hw, params->hwPerBlock, params->withSwish);
  return HIP_CALL(hipGetLastError());
}

template <typename T, int CPerBlock, int ThreadsPerBlock, int VecSize>
class GroupNormNHWCOp {
 public:
  Status operator()(const GroupNormNHWCParams<T>* params) {
    DUMP_TENSOR_INIT();
    DUMP_TENSOR("input", params->src, params->n, params->c, params->h * params->w);
    DUMP_TENSOR("gamma", params->gamma, 1, params->c);
    DUMP_TENSOR("beta", params->beta, 1, params->c);
    HIP_RETURN_IF_ERROR(hipMemsetAsync(params->redBuffer, 0, GetGroupNormWorkspaceSizeInBytes(), params->stream));
    auto status = GroupNormNHWCSumOp<T, CPerBlock, ThreadsPerBlock, VecSize>(params);
    ORT_RETURN_IF_ERROR(status);
    DUMP_TENSOR("workspace", params->redBuffer, params->n, params->groups, 2);
    HIP_RETURN_IF_ERROR(hipGetLastError());
    status = GroupNormNHWCScaleOp<T, CPerBlock, ThreadsPerBlock, VecSize>(params);
    ORT_RETURN_IF_ERROR(status);
    HIP_RETURN_IF_ERROR(hipGetLastError());
    DUMP_TENSOR("output", params->dst, params->n, params->c, params->h * params->w);
    return Status::OK();
  }

  Status IsSupported(const GroupNormNHWCParams<T>* params) {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        !(params->c % VecSize == 0 && params->cPerGroup % VecSize == 0));
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!(CPerBlock % params->cPerGroup == 0 &&
                                                params->c % CPerBlock == 0 &&
                                                params->hw % params->hwPerBlock == 0));
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!(CPerBlock <= ThreadsPerBlock * VecSize &&
                                                CPerBlock > (ThreadsPerBlock - onnxruntime::rocm::GPU_WARP_SIZE) * VecSize));

    return Status::OK();
  }
};

template <typename T>
Status GroupNormNHWCStaticSelection(const GroupNormNHWCParams<T>* params) {
  DUMP_TENSOR_INIT();
  DUMP_TENSOR("input", params->src, params->n, params->c, params->h * params->w);
  DUMP_TENSOR("gamma", params->gamma, 1, params->c);
  DUMP_TENSOR("beta", params->beta, 1, params->c);
  HIP_RETURN_IF_ERROR(hipMemsetAsync(params->redBuffer, 0, GetGroupNormWorkspaceSizeInBytes(), params->stream));
  groupNormNHWCSum<T>(params);
  DUMP_TENSOR("workspace", params->redBuffer, params->n, params->groups, 2);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  groupNormNHWCScale<T>(params);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  DUMP_TENSOR("output", params->dst, params->n, params->c, params->h * params->w);
  return Status::OK();
}

#define ADD_OP_FOR_ALL_CONFIG(name, c_per_block, threads_per_block, vec_size)            \
  if (c_per_block <= threads_per_block * vec_size &&                                     \
      c_per_block > (threads_per_block - onnxruntime::rocm::GPU_WARP_SIZE) * vec_size) { \
    this->RegisterOp(name<T, c_per_block, threads_per_block, vec_size>{});               \
  }

#define ADD_OP_FOR_ALL_VEC_SIZE(name, c_per_block, threads_per_block) \
  ADD_OP_FOR_ALL_CONFIG(name, c_per_block, threads_per_block, 1);     \
  ADD_OP_FOR_ALL_CONFIG(name, c_per_block, threads_per_block, 2);     \
  ADD_OP_FOR_ALL_CONFIG(name, c_per_block, threads_per_block, 4);     \
  ADD_OP_FOR_ALL_CONFIG(name, c_per_block, threads_per_block, 8);     \
  ADD_OP_FOR_ALL_CONFIG(name, c_per_block, threads_per_block, 16);

#define ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name, c_per_block) \
  ADD_OP_FOR_ALL_VEC_SIZE(name, c_per_block, 64)                         \
  ADD_OP_FOR_ALL_VEC_SIZE(name, c_per_block, 128)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, c_per_block, 192)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, c_per_block, 256)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, c_per_block, 320)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, c_per_block, 384)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, c_per_block, 448)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, c_per_block, 512)

#define ADD_OP_FOR_ALL_C_PER_BLOCK_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name) \
  ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name, 64)                   \
  ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name, 128)                  \
  ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name, 160)                  \
  ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name, 256)                  \
  ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name, 320)                  \
  ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name, 480)

template <typename T>
class GroupNormNHWCTunableOp : public TunableOp<GroupNormNHWCParams<T>> {
 public:
  GroupNormNHWCTunableOp() {
    this->RegisterOp(GroupNormNHWCStaticSelection<T>);
    ADD_OP_FOR_ALL_C_PER_BLOCK_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(GroupNormNHWCOp)
  }
};

#undef ADD_OP_FOR_ALL_CONFIG
#undef ADD_OP_FOR_ALL_VEC_SIZE
#undef ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE
#undef ADD_OP_FOR_ALL_C_PER_BLOCK_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
