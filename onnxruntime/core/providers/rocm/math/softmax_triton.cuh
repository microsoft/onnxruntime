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

template <typename T>
std::string GetSoftmaxTritonName(int softmax_elements) {
  int block_size = NextPowerOf2(softmax_elements);
  std::string ret = "softmax_";
  std::string type_name = GetDataTypeName<T>();
  ret += type_name;
  ret += "_";
  if (block_size <= 1024) {
    block_size = 1024;
  }
  ret += std::to_string(block_size);
  return ret;
}

}  // end of namespace

template <typename T, typename OutputT>
Status SoftmaxTritonOp(const SoftmaxParams<T, OutputT>* params) {
  auto fname = GetSoftmaxTritonName<T>(params->softmax_elements);

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
#endif  // ENABLE_TRITON_LIB

}  // namespace rocm
}  // namespace onnxruntime
