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

#ifdef ENABLE_TRITON_LIB

namespace {
// TODO: onnxruntime softmax is not support output has different type between input
// so actually there is only one dtype

#define SOFTMAX_PRE_NAME(type) "sfotmax_" #type "_"

template <typename T>
std::string GetSoftmaxTritonName(int softmax_elements) {
  int block_size = NextPowerOf2(softmax_elements);
  std::string ret = SOFTMAX_PRE_NAME(T);
  if (block_size <= 1024) {
    block_size = 1024;
  } else if (block_size <= 2048) {
    block_size = 2048;
  } else if (block_size <= 4096) {
    block_size = 4096;
  } else {
    return ret;
  }
  ret += std::to_string(block_size);
  return ret;
}
}  // end of namespace

template <typename T>
Status SoftmaxTritonOp(const SoftmaxParams<InputT, OutputT>* params) {
  status = Status::OK();
  auto fname = GetSoftmaxTritonName(params->softmax_elements);

  // construct args for launch kernel
  struct {
    void *out;
    const void *int;
    int in_stride;
    int out_stride;
    int n_cols;
  } args = {(void*)params->output, (const void*)params->input, params->input_stride, params->output_stride, params->softmax_elements};

  // grid dim is (batch_count, 1, 1)
  status = LaunchTritonKernel(params->stream, fname, params->batch_count, 1, 1, &args, sizeof(args));

  return status;
}
#endif  // ENABLE_TRITON_LIB

}  // namespace rocm
}  // namespace onnxruntime
