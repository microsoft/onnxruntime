// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class ATenOp : public OpKernel {
 public:
  ATenOp(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("name", &op_name_));
    overload_name_ = info.GetAttrOrDefault<std::string>("overload_name", "");
    requires_grad_ = info.GetAttrsOrDefault<int64_t>("requires_grad");
  }

  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  std::string op_name_;
  std::string overload_name_;
  std::vector<int64_t> requires_grad_;
};

class ATenOpGrad : public OpKernel {
 public:
  ATenOpGrad(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* p_ctx) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
