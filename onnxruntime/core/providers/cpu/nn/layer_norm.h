// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename T>
class LayerNorm final : public OpKernel {
 public:
  LayerNorm(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t axis_;
  int64_t keep_dims_;
  float epsilon_;
};

}  // namespace onnxruntime
