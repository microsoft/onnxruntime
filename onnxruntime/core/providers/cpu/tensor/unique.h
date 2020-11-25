// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class Unique final : public OpKernel {
 public:
  explicit Unique(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      flatten_ = true;
    }

    sort_ = info.GetAttrOrDefault<int64_t>("sorted", 1) == 1;
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext& context) const;

  bool sort_{true};
  bool flatten_{false};
  int64_t axis_{0};
};
}  // namespace onnxruntime
