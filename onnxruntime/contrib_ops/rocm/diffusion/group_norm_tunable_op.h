// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "contrib_ops/rocm/diffusion/group_norm_ck.cuh"
#include "contrib_ops/rocm/diffusion/group_norm_common.h"
#include "contrib_ops/rocm/diffusion/group_norm_impl.h"
#include "contrib_ops/rocm/diffusion/group_norm_impl_kernel.cuh"
#include "contrib_ops/rocm/diffusion/group_norm_triton.cuh"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using onnxruntime::rocm::GPU_WARP_SIZE;

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
  grid.y = CeilDiv(params->hw, params->hwPerBlock);
  // The number of instances.
  grid.z = params->n;

#define LAUNCH_GROUPNORM_SUM(ThreadsPerBlock, VecSize)                \
  groupNormNHWCSumKernel<T, ThreadsPerBlock, VecSize>                 \
      <<<grid, ThreadsPerBlock, 0, params->StreamHandle()>>>(         \
          params->src, params->redBuffer, params->cPerBlock,          \
          params->hwPerBlock, params->hw, params->hwc, params->c,     \
          params->cPerGroup, params->groups, params->groupsPerBlock); \
  break;

  switch (params->cPerBlock) {
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

template <typename T, int ThreadsPerBlock, int VecSize>
Status GroupNormNHWCSumOp(const GroupNormNHWCParams<T>* params) {
  dim3 grid;
  grid.x = params->c / params->cPerBlock;
  grid.y = CeilDiv(params->hw, params->hwPerBlock);
  grid.z = params->n;

  groupNormNHWCSumKernel<T, ThreadsPerBlock, VecSize>
      <<<grid, ThreadsPerBlock, 0, params->StreamHandle()>>>(
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
  grid.y = CeilDiv(params->hw, params->hwPerBlock);
  // The number of instances.
  grid.z = params->n;

#define LAUNCH_GROUPNORM_SCALE(ThreadsPerBlock, VecSize)                    \
  groupNormNHWCScaleKernel<T, ThreadsPerBlock, VecSize>                     \
      <<<grid, ThreadsPerBlock, 0, params->StreamHandle()>>>(               \
          params->dst, params->src, params->gamma, params->beta,            \
          params->redBuffer, params->epsilon, params->c, params->cPerBlock, \
          params->cPerGroup, params->groups, params->hwc, params->invHWC,   \
          params->hw, params->hwPerBlock, params->withSwish);               \
  break;

  switch (params->cPerBlock) {
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

template <typename T, int ThreadsPerBlock, int VecSize>
Status GroupNormNHWCScaleOp(const GroupNormNHWCParams<T>* params) {
  dim3 grid;
  grid.x = params->c / params->cPerBlock;
  grid.y = CeilDiv(params->hw, params->hwPerBlock);
  grid.z = params->n;

  groupNormNHWCScaleKernel<T, ThreadsPerBlock, VecSize>
      <<<grid, ThreadsPerBlock, 0, params->StreamHandle()>>>(
          params->dst, params->src, params->gamma, params->beta, params->redBuffer, params->epsilon, params->c, params->cPerBlock,
          params->cPerGroup, params->groups, params->hwc, params->invHWC, params->hw, params->hwPerBlock, params->withSwish);
  return HIP_CALL(hipGetLastError());
}

template <typename T, int ThreadsPerBlock, int VecSize>
class GroupNormNHWCOp {
 public:
  Status operator()(const GroupNormNHWCParams<T>* params) {
    HIP_RETURN_IF_ERROR(hipMemsetAsync(params->redBuffer, 0, GetGroupNormWorkspaceSizeInBytes(), params->StreamHandle()));
    auto status = GroupNormNHWCSumOp<T, ThreadsPerBlock, VecSize>(params);
    ORT_RETURN_IF_ERROR(status);
    HIP_RETURN_IF_ERROR(hipGetLastError());
    status = GroupNormNHWCScaleOp<T, ThreadsPerBlock, VecSize>(params);
    ORT_RETURN_IF_ERROR(status);
    HIP_RETURN_IF_ERROR(hipGetLastError());
    return Status::OK();
  }

  Status IsSupported(const GroupNormNHWCParams<T>* params) {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        !(params->c % VecSize == 0 && params->cPerGroup % VecSize == 0),
        "The number of channels (", params->c, ") or the number of channels per group (", params->cPerGroup,
        ") isn't divisible by the number of vector size: ", VecSize);
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!(params->cPerBlock % params->cPerGroup == 0 &&
                                                params->c % params->cPerBlock == 0 && params->hw % params->hwPerBlock == 0),
                                              "The value of attributes don't meet the requirements.");
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!(params->cPerBlock <= ThreadsPerBlock * VecSize &&
                                                params->cPerBlock > (ThreadsPerBlock - GPU_WARP_SIZE) * VecSize),
                                              "Configuration: Threads (", ThreadsPerBlock, "), vector size (",
                                              VecSize, ") is redundant for the number of channels per group: ", params->cPerBlock);

    return Status::OK();
  }
};

template <typename T>
Status GroupNormNHWCStaticSelection(const GroupNormNHWCParams<T>* params) {
  HIP_RETURN_IF_ERROR(hipMemsetAsync(params->redBuffer, 0, GetGroupNormWorkspaceSizeInBytes(), params->StreamHandle()));
  groupNormNHWCSum<T>(params);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  groupNormNHWCScale<T>(params);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  return Status::OK();
}

#define ADD_OP_FOR_ALL_VEC_SIZE(name, threads_per_block) \
  this->RegisterOp(name<T, threads_per_block, 1>{});     \
  this->RegisterOp(name<T, threads_per_block, 2>{});     \
  this->RegisterOp(name<T, threads_per_block, 4>{});

#define ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name) \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 64)                         \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 128)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 192)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 256)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 320)

template <typename T>
class GroupNormNHWCTunableOp : public TunableOp<GroupNormNHWCParams<T>> {
 public:
  GroupNormNHWCTunableOp() {
    this->RegisterOp(GroupNormNHWCStaticSelection<T>);
    ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(GroupNormNHWCOp)

#ifdef USE_COMPOSABLE_KERNEL
    for (auto&& [_, op] : GetCKGroupNormNHWCTypeStringAndOps<T, /*AccT=*/float, /*WithSwish=*/false>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }

    for (auto&& [_, op] : GetCKGroupNormNHWCTypeStringAndOps<T, /*AccT=*/float, /*WithSwish=*/true>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif  // USE_COMPOSABLE_KERNEL

#ifdef USE_TRITON_KERNEL
    for (auto&& [_, op] : GetTritonGroupNormNHWCTypeStringAndOps<T, /*WithSwish=*/false>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
    for (auto&& [_, op] : GetTritonGroupNormNHWCTypeStringAndOps<T, /*WithSwish=*/true>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif
  }
};

#undef ADD_OP_FOR_ALL_VEC_SIZE
#undef ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
