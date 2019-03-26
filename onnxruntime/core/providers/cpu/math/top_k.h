// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
template <int OpSet, typename T>
class TopK final : public OpKernel {
 public:
  TopK(const OpKernelInfo& op_kernel_info);

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  int axis_;
  unsigned k_;
};
}  // namespace onnxruntime