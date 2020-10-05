// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
class BroadcastGradientArgs final : public OpKernel {
 public:
  BroadcastGradientArgs(const OpKernelInfo& info) : OpKernel{info} {
  }

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace contrib
}  // namespace onnxruntime
