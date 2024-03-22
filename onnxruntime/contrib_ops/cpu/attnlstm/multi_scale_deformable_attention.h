// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class MultiScaleDeformableAttention final : public OpKernel {
  public:
  MultiScaleDeformableAttention(const OpKernelInfo& info);
  [[nodiscard]] Status Compute(_Inout_ OpKernelContext* context) const override;
};

}   // namespace contrib
}   // namespace onnxruntime
