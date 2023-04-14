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
std::string GetSoftmaxTritonName(int num_elements, bool is_log_softmax) {
  std::string ret = "softmax_";
  if (is_log_softmax) {
    ret = "log_softmax_";
  }
  std::string type_name = GetDataTypeName<T>();
  ret += type_name;
  ret += "_";

  // search for most suitable block_size
  // the block size must larger than num_elements
  int block_size = 1024;
  while (num_elements > block_size) {
    block_size *= 2;
  }
  // current max block_size is 16384
  if (block_size > 16384) {
    return ret;  // this will cause LaunchKernel return not supported error
  }
  ret += std::to_string(block_size);
  return ret;
}

template <typename T>
std::string GetSoftmaxTritonGroupName() {
  std::string ret = "softmax_";
  ret += GetDataTypeName<T>();
  return ret;
}

}  // end of namespace

// TODO: onnxruntime softmax is not support output has different type between input
// so actually T and OutputT should be same type
template <typename T, typename OutputT>
Status SoftmaxTritonOp(const SoftmaxParams<T, OutputT>* params) {
  auto fname = GetSoftmaxTritonName<T>(params->softmax_elements, params->is_log_softmax);

  // construct args for launch kernel
  struct {
    void *out;
    const void *in;
    int in_stride;
    int out_stride;
    int n_cols;
  } args = {(void*)params->output, (const void*)params->input, params->input_stride, params->output_stride, params->softmax_elements};

  // grid dim is (batch_count, 1, 1)
  return LaunchTritonKernel(params->stream, fname, params->batch_count, 1, 1, &args, sizeof(args));
}

template <typename T, typename OutputT>
auto GetSoftmaxTritonOps() {
  std::vector<std::pair<std::string, tunable::Op<SoftmaxParams<T, OutputT>>>> ret;
  auto group_name = GetSoftmaxTritonGroupName<T>();
  auto *kernel_list = GetRocmTritonKernelByGroup(group_name);
  if (kernel_list == nullptr) {
    return ret;
  }

  for (auto i : *kernel_list) {
    // check params match
    auto *metadata = GetRocmTritonKernelMetadata(i);
    auto block_size = -1;
    const std::string block_name = "BLOCK_SIZE";
    if (metadata->constants.count(block_name) != 0) {
      block_size = metadata->constants.at(block_name);
    }
    auto impl = [i, block_size](const SoftmaxParams<T, OutputT> *params) -> Status {
      if (params->log_softmax_) {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(true, "log_softmax not support");
      }
      if (block_size < params->softmax_elements) {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(true, "BLOCK_SIZE not support");
      }
      // construct args for launch kernel
      struct {
        void *out;
        const void *in;
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
