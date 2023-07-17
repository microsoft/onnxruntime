// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/providers/rocm/math/softmax_common.h"
#include "core/providers/rocm/triton_kernel.h"

namespace onnxruntime {
namespace rocm {

#ifdef USE_TRITON_KERNEL

namespace {

template <typename T>
std::string GetSoftmaxTritonGroupName() {
  std::string ret = "softmax_";
  ret += GetDataTypeName<T>();
  return ret;
}

}  // end of namespace

template <typename T, typename OutputT>
auto GetSoftmaxTritonOps() {
  std::vector<std::pair<std::string, tunable::Op<SoftmaxParams<T, OutputT>>>> ret;
  auto group_name = GetSoftmaxTritonGroupName<T>();
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
    auto impl = [i, block_size](const SoftmaxParams<T, OutputT>* params) -> Status {
      if (params->is_log_softmax) {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(true, "log_softmax is not supported.");
      }
      if (block_size < params->softmax_elements) {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(true, "BLOCK_SIZE (", block_size, ") is not supported.");
      }
      // construct args for launch kernel
      struct {
        void* out;
        const void* in;
        int in_stride;
        int out_stride;
        int n_cols;
      } args = {(void*)params->output, (const void*)params->input, params->input_stride, params->output_stride, params->softmax_elements};

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
