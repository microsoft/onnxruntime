// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class ScalerOp final : public OpKernel {
 public:
  explicit ScalerOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<float> scale_;
  std::vector<float> offset_;
};
}  // namespace ml
}  // namespace onnxruntime
