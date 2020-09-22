// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

template <typename T, bool simplified>
class LayerNorm final : public OpKernel {
 public:
  LayerNorm(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t axis_;
  float epsilon_;
};

}  // namespace contrib
}  // namespace onnxruntime
