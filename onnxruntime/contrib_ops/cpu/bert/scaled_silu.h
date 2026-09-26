// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime::contrib {

template <typename T>
class ScaledSiLU final : public OpKernel {
 public:
  explicit ScaledSiLU(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
};

}  // namespace onnxruntime::contrib
