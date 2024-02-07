// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "contrib_ops/rocm/diffusion/group_norm_common.h"
#include "core/providers/rocm/triton_kernel.h"

using namespace onnxruntime::rocm;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#ifdef USE_TRITON_KERNEL

namespace {

template <typename T, bool WithSwish>
std::string GetGroupNormTritonGroupName() {
  std::string ret = "GroupNormTriton_";
  std::string swish_suffix = WithSwish ? "Swish_" : "Pass_";
  ret += swish_suffix;
  ret += GetDataTypeName<T>();
  return ret;
}

}  // namespace

template <typename T, bool WithSwish>
auto GetTritonGroupNormNHWCTypeStringAndOps() {
  std::vector<std::pair<std::string, tunable::Op<GroupNormNHWCParams<T>>>> ret;
  auto group_name = GetGroupNormTritonGroupName<T, WithSwish>();
  auto* kernel_list = GetOrtTritonKernelByGroup(group_name);
  if (kernel_list == nullptr) {
    return ret;
  }

  for (auto i : *kernel_list) {
    // Check params match
    auto* metadata = GetOrtTritonKernelMetadata(i);
    auto block_size = metadata->constants.at("BLOCK_SIZE");
    auto hw_size = metadata->constants.at("HW_SIZE");
    auto impl = [i, block_size, hw_size](const GroupNormNHWCParams<T>* params) -> Status {
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->cPerGroup > block_size || params->cPerGroup * 2 <= block_size,
          "Arg block_size (", block_size, ") is not the next power of 2 of cPerGroup (", params->cPerGroup, ").");
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->hw % hw_size != 0, "Arg hw_size (", hw_size, ") is not a divisor of hw (", params->hw, ").");
      if constexpr (WithSwish) {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!params->withSwish, "Swish version does not support GN w/o swish.");
      } else {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(params->withSwish, "Pass version does not support GN w/ swish.");
      }
      // Construct args for launch kernel
      struct {
        void* X;
        void* Y;
        const void* gamma;
        const void* beta;
        int hw;
        int c;
        int c_per_group;
        float eps;
      } args = {
          (void*)params->src,
          (void*)params->dst,
          (const void*)params->gamma,
          (const void*)params->beta,
          params->hw,
          params->c,
          params->cPerGroup,
          params->epsilon};

      // Grid dim is (batch_count, groups, 1)
      return LaunchTritonKernel(params->StreamHandle(), i, params->n, params->groups, 1, &args, sizeof(args));
    };
    ret.emplace_back(std::make_pair(metadata->name, std::move(impl)));
  }
  return ret;
}

#endif  // USE_TRITON_KERNEL

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
