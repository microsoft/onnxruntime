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
void GroupNormNHWCSum(const GroupNormNHWCTunableParams<T>* params) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = DivUp(params->c, params->channels_per_block);
  // The number of blocks to compute all the activations in a given instance.
  grid.y = DivUp(params->hw, params->hw_per_block);
  // The number of instances.
  grid.z = params->n;

#define LAUNCH_GROUPNORM_SUM(ThreadsPerBlock, VecSize)                                                   \
  GroupNormNHWCSumKernel<T, ThreadsPerBlock, VecSize>                                                    \
      <<<grid, ThreadsPerBlock, 0, params->StreamHandle()>>>(                                            \
          params->skip_workspace, params->group_sum_buffer, params->src, params->skip, params->bias,     \
          params->channels_per_block, params->hw_per_block, params->hw, params->hwc, params->c,          \
          params->channels_per_group, params->groups, params->groups_per_block, params->broadcast_skip); \
  break;

  // Threads_per_block is half of values in kSizes since CHANNELS_PER_THREAD = 2.
  switch (params->threads_per_block) {
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
    default:
      ORT_NOT_IMPLEMENTED("Not implemented");
  }
}

template <typename T, int ThreadsPerBlock, int VecSize>
Status GroupNormNHWCSumOp(const GroupNormNHWCTunableParams<T>* params) {
  dim3 grid;
  grid.x = DivUp(params->c, params->channels_per_block);
  grid.y = DivUp(params->hw, params->hw_per_block);
  grid.z = params->n;

  GroupNormNHWCSumKernel<T, ThreadsPerBlock, VecSize>
      <<<grid, ThreadsPerBlock, 0, params->StreamHandle()>>>(
          params->skip_workspace, params->group_sum_buffer, params->src, params->skip, params->bias,
          params->channels_per_block, params->hw_per_block, params->hw, params->hwc, params->c,
          params->channels_per_group, params->groups, params->groups_per_block, params->broadcast_skip);
  return HIP_CALL(hipGetLastError());
}

template <typename T>
void GroupNormNHWCScale(const GroupNormNHWCTunableParams<T>* params) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = DivUp(params->c, params->channels_per_block);
  // The number of blocks to compute all the activations in a given instance.
  grid.y = DivUp(params->hw, params->hw_per_block);
  // The number of instances.
  grid.z = params->n;

#define LAUNCH_GROUPNORM_SCALE(ThreadsPerBlock, VecSize)                                               \
  GroupNormNHWCScaleKernel<T, VecSize>                                                                 \
      <<<grid, ThreadsPerBlock, 0, params->StreamHandle()>>>(                                          \
          params->dst, params->src, params->skip, params->gamma, params->beta, params->skip_workspace, \
          params->group_sum_buffer, params->epsilon, params->c, params->channels_per_block,            \
          params->channels_per_group, params->groups, params->hwc, params->inv_hw_channels_per_group,  \
          params->hw, params->hw_per_block, params->use_silu);                                         \
  break;

  // Threads_per_block is half of values in kSizes since CHANNELS_PER_THREAD = 2.
  switch (params->threads_per_block) {
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
    default:
      ORT_NOT_IMPLEMENTED("Not implemented");
  }
}

template <typename T, int ThreadsPerBlock, int VecSize>
Status GroupNormNHWCScaleOp(const GroupNormNHWCTunableParams<T>* params) {
  dim3 grid;
  grid.x = DivUp(params->c, params->channels_per_block);
  grid.y = DivUp(params->hw, params->hw_per_block);
  grid.z = params->n;

  GroupNormNHWCScaleKernel<T, VecSize>
      <<<grid, ThreadsPerBlock, 0, params->StreamHandle()>>>(
          params->dst, params->src, params->skip, params->gamma, params->beta, params->skip_workspace,
          params->group_sum_buffer, params->epsilon, params->c, params->channels_per_block, params->channels_per_group,
          params->groups, params->hwc, params->inv_hw_channels_per_group, params->hw, params->hw_per_block,
          params->use_silu);
  return HIP_CALL(hipGetLastError());
}

template <typename T, int ThreadsPerBlock, int VecSize>
class GroupNormNHWCOp {
 public:
  Status operator()(const GroupNormNHWCTunableParams<T>* params) {
    HIP_RETURN_IF_ERROR(hipMemsetAsync(params->group_sum_buffer,
                                       0,
                                       GetGroupNormWorkspaceSizeInBytes(params->n, params->groups),
                                       params->StreamHandle()));
    auto status = GroupNormNHWCSumOp<T, ThreadsPerBlock, VecSize>(params);
    ORT_RETURN_IF_ERROR(status);
    HIP_RETURN_IF_ERROR(hipGetLastError());
    status = GroupNormNHWCScaleOp<T, ThreadsPerBlock, VecSize>(params);
    ORT_RETURN_IF_ERROR(status);
    HIP_RETURN_IF_ERROR(hipGetLastError());
    return Status::OK();
  }

  Status IsSupported(const GroupNormNHWCTunableParams<T>* params) {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        !(params->c % VecSize == 0 && params->channels_per_group % VecSize == 0),
        "The number of channels (", params->c, ") or the number of channels per group (", params->channels_per_group,
        ") isn't divisible by the number of vector size: ", VecSize);
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!(params->channels_per_block <= ThreadsPerBlock * VecSize &&
                                                params->channels_per_block > (ThreadsPerBlock - GPU_WARP_SIZE) * VecSize),
                                              "Configuration: Threads (", ThreadsPerBlock, "), vector size (",
                                              VecSize, ") is redundant for the number of channels per group: ",
                                              params->channels_per_block);

    return Status::OK();
  }
};

template <typename T>
Status GroupNormNHWCStaticSelection(const GroupNormNHWCTunableParams<T>* params) {
  HIP_RETURN_IF_ERROR(hipMemsetAsync(params->group_sum_buffer,
                                     0,
                                     GetGroupNormWorkspaceSizeInBytes(params->n, params->groups),
                                     params->StreamHandle()));
  GroupNormNHWCSum<T>(params);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  GroupNormNHWCScale<T>(params);
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
class GroupNormNHWCTunableOp : public TunableOp<GroupNormNHWCTunableParams<T>> {
 public:
  GroupNormNHWCTunableOp() {
    this->RegisterOp(GroupNormNHWCStaticSelection<T>);
    ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(GroupNormNHWCOp)

#ifdef USE_COMPOSABLE_KERNEL
    for (auto&& [_, op] : GetCKGroupNormNHWCTypeStringAndOps<T, /*AccT=*/float, /*WithSilu=*/false>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }

    for (auto&& [_, op] : GetCKGroupNormNHWCTypeStringAndOps<T, /*AccT=*/float, /*WithSilu=*/true>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif  // USE_COMPOSABLE_KERNEL

#ifdef USE_TRITON_KERNEL
    for (auto&& [_, op] : GetTritonGroupNormNHWCTypeStringAndOps<T, /*WithSilu=*/false>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
    for (auto&& [_, op] : GetTritonGroupNormNHWCTypeStringAndOps<T, /*WithSilu=*/true>()) {
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
