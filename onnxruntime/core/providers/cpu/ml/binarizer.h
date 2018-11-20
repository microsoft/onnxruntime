// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class BinarizerOp final : public OpKernel {
 public:
  explicit BinarizerOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  const float threshold_;
};
}  // namespace ml
}  // namespace onnxruntime
