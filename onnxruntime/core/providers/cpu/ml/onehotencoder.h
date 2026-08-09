// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class OneHotEncoderOp final : public OpKernel {
 public:
  explicit OneHotEncoderOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  std::unordered_map<int64_t, size_t> cats_int64s_;
  std::unordered_map<std::string, size_t> cats_strings_;
  int64_t zeros_;
  int64_t num_categories_;
};
}  // namespace ml
}  // namespace onnxruntime
