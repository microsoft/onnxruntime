// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include <unordered_set>
#include "core/util/eigen_common_wrapper.h"

namespace onnxruntime {
namespace contrib {

// Reverse kernel
class Reverse final : public OpKernel {
 public:
  explicit Reverse(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    op_kernel_info.GetAttrs("axes", attr_axes_).IsOK();
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

  std::vector<int64_t> attr_axes_;
};
}  // namespace contrib
}  // namespace onnxruntime