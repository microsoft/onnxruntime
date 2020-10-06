// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T, typename ScaleT>
class Scale final : public OpKernel {
 public:
  Scale(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  bool scale_down_;
};

}  // namespace contrib
}  // namespace onnxruntime
