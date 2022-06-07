// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class ATen : public OpKernel {
 public:
  ATen(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("operator", &op_name_));
    overload_name_ = info.GetAttrOrDefault<std::string>("overload_name", "");
  }

  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  std::string op_name_;
  std::string overload_name_;
};

bool IsATenOperatorExecutorInitialized();
Status ExecuteReduceSumATen(OpKernelContext* p_ctx, const gsl::span<const int64_t>& axes, bool keepdims);

}  // namespace contrib
}  // namespace onnxruntime
