// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

class GroupNorm final : public OpKernel {
 public:
  GroupNorm(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;

 private:
  template<typename T>
  Status ComputeImpl(OpKernelContext* context, const Tensor* X, const Tensor* scale, const Tensor* bias) const;
  
  Status ComputeHelper(OpKernelContext* context, const Tensor* X, const Tensor* scale, const Tensor* bias) const;

  float epsilon_;
  int64_t num_groups_;
  int64_t stash_type_;
};

}  // namespace onnxruntime