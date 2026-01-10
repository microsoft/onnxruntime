// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename T>
class AffineGrid final : public OpKernel {
 public:
  AffineGrid(const OpKernelInfo& info) : OpKernel(info) {
    int64_t align_corners = info.GetAttrOrDefault<int64_t>("align_corners", 0);
    align_corners_ = (align_corners != 0);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool align_corners_;
};

}  // namespace onnxruntime
