// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
class ImputerOp final : public OpKernel {
 public:
  explicit ImputerOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<float> imputed_values_float_;
  float replaced_value_float_;
  std::vector<int64_t> imputed_values_int64_;
  int64_t replaced_value_int64_;
};
}  // namespace ml
}  // namespace onnxruntime
