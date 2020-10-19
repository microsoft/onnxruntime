// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class GatherElementsGrad final : public OpKernel {
 public:
  GatherElementsGrad(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(0));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace contrib
}  // namespace onnxruntime
