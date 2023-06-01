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

inline int NextPowerOfTwo(int n) {
  int ret = 1;
  while (ret < n) {
    ret <<= 1;
  }
  return ret;
}

template <typename T, bool WithSwish>
auto GetTritonGroupNormNHWCTypeStringAndOps() {
  std::vector<std::pair<std::string, tunable::Op<GroupNormNHWCParams<T>>>> ret;
  auto group_name = GetGroupNormTritonGroupName<T, WithSwish>();
  auto* kernel_list = GetOrtTritonKernelByGroup(group_name);
  if (kernel_list == nullptr) {
    return ret;
  }

  for (auto i : *kernel_list) {
    // check params match
    auto* metadata = GetOrtTritonKernelMetadata(i);
    auto block_size = -1;
    const std::string block_name = "BLOCK_SIZE";
    if (metadata->constants.count(block_name) != 0) {
      block_size = metadata->constants.at(block_name);
    }
    auto impl = [i, block_size](const GroupNormNHWCParams<T>* params) -> Status {
      auto min_block_size = NextPowerOfTwo(params->cPerGroup);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        block_size != min_block_size, "Only keep BLOCK_SIZE == next power of 2 of cPerGroup");
      if constexpr (WithSwish) {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!params->withSwish, "Swish version only supports GN w/ swish");
      } else {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(params->withSwish, "Pass version only supports GN w/o swish");
      }
      // construct args for launch kernel
      struct {
        void* X;
        void* Y;
        const void* gamma;
        const void* beta;
        int h;
        int w;
        int c;
        int c_per_group;
        float eps;
      } args = {
          (void*)params->src,
          (void*)params->dst,
          (const void*)params->gamma,
          (const void*)params->beta,
          params->h,
          params->w,
          params->c,
          params->cPerGroup,
          params->epsilon};

      // grid dim is (batch_count, groups, 1)
      return LaunchTritonKernel(params->stream, i, params->n, params->groups, 1, &args, sizeof(args));
    };
    ret.emplace_back(std::make_pair(metadata->name, std::move(impl)));
  }
  return ret;
}

#endif  // USE_TRITON_KERNEL

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
