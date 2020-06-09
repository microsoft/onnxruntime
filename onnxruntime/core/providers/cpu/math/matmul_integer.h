// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename T1>
class MatMulInteger final : public OpKernel {
 public:
  MatMulInteger(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace onnxruntime
