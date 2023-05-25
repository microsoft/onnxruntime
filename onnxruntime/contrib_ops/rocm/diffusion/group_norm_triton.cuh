// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "contrib_ops/rocm/diffusion/group_norm_common.h"
#include "core/providers/rocm/triton_kernel.h"

namespace onnxruntime {
namespace rocm {

#ifdef USE_TRITON_KERNEL

namespace {

template <typename T>
std::string GetGroupNormTritonGroupName() {
  std::string ret = "GroupNormTriton_";
  ret += GetDataTypeName<T>();
  return ret;
}

}  // namespace

template <typename T>
auto GetTritonGroupNormNHWCTypeStringAndOps() {
  std::vector<std::pair<std::string, tunable::Op<GroupNormNHWCParams<T>>>> ret;
  auto group_name = GetGroupNormTritonGroupName<T>();
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
      // construct args for launch kernel
      struct {
        void* X;
        void* Y;
        const void* gamma;
        const void* beta;
        int32_t h;
        int32_t w;
        int32_t c;
        float eps;
      } args = {
          (void*)params->input,
          (void*)params->output,
          (const float*)params->gamma,
          (const float*)params->beta,
          params->h,
          params->w,
          params->c,
          params->epsilon};

      // grid dim is (batch_count, 1, 1)
      return LaunchTritonKernel(params->stream, i, params->batch_count, 1, 1, &args, sizeof(args));
    };
    ret.emplace_back(std::make_pair(metadata->name, std::move(impl)));
  }
  return ret;
}

#endif  // USE_TRITON_KERNEL

}  // namespace rocm
}  // namespace onnxruntime
