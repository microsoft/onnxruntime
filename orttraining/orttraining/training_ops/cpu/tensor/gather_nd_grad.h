// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/gather_nd.h"

namespace onnxruntime {

class GatherNDGrad final : public OpKernel, protected GatherNDBase {
 public:
  explicit GatherNDGrad(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault("batch_dims", &batch_dims_, static_cast<int64_t>(0));
  }
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
