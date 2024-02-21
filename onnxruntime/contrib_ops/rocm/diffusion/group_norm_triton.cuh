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

template <typename T, bool WithSilu>
std::string GetGroupNormTritonGroupName() {
  std::string ret = "GroupNormTriton_";
  std::string silu_suffix = WithSilu ? "Silu_" : "Pass_";
  ret += silu_suffix;
  ret += GetDataTypeName<T>();
  return ret;
}

}  // namespace

template <typename T, bool WithSilu>
auto GetTritonGroupNormNHWCTypeStringAndOps() {
  std::vector<std::pair<std::string, tunable::Op<GroupNormNHWCTunableParams<T>>>> ret;
  auto group_name = GetGroupNormTritonGroupName<T, WithSilu>();
  auto* kernel_list = GetOrtTritonKernelByGroup(group_name);
  if (kernel_list == nullptr) {
    return ret;
  }

  for (auto i : *kernel_list) {
    // Check params match
    auto* metadata = GetOrtTritonKernelMetadata(i);
    auto block_size = metadata->constants.at("BLOCK_SIZE");
    auto hw_size = metadata->constants.at("HW_SIZE");
    auto impl = [i, block_size, hw_size](const GroupNormNHWCTunableParams<T>* params) -> Status {
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF((params->skip != nullptr || params->bias != nullptr),
                                                "Input skip or bias is not supported by triton kernel.");
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->channels_per_group > block_size || params->channels_per_group * 2 <= block_size,
          "Arg block_size (", block_size, ") is not the next power of 2 of channels_per_group (",
          params->channels_per_group, ").");
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->hw % hw_size != 0, "Arg hw_size (", hw_size, ") is not a divisor of hw (", params->hw, ").");
      if constexpr (WithSilu) {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!params->use_silu, "Silu version does not support GN w/o silu.");
      } else {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(params->use_silu, "Pass version does not support GN w/ silu.");
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
          params->channels_per_group,
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
