// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class Compress final : public OpKernel {
 public:
  Compress(const OpKernelInfo& info) : OpKernel(info) {
    has_axis_ = info.GetAttr("axis", &axis_).IsOK();
  }

  Status Compute(OpKernelContext* context) const override;

  private:
  int64_t axis_;
  bool has_axis_;
};

}  // namespace onnxruntime
