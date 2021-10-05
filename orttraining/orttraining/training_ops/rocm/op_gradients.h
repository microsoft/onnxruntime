// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {
namespace rocm {

template <typename T>
class SigmoidGrad final : public OpKernel {
 public:
  explicit SigmoidGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SigmoidGrad);
};
} // namespace rocm
} // namespace onnxruntime
