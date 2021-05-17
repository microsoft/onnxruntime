// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/language_interop_ops/torch/torch_proxy.h"
#include "core/language_interop_ops/torch/custom_function_register.h"
#include "orttraining/training_ops/cpu/torch/torch_custom_function_kernel_base.h"

namespace onnxruntime {
namespace contrib {

// Pytorch's torch.autograd.Function.apply(...) wrapper.
class PythonOp final : public OpKernel, public PythonOpBase {
 public:
  PythonOp(const OpKernelInfo& info) : OpKernel(info), PythonOpBase(info) {}
  Status Compute(OpKernelContext* context) const override;
};

// Pytorch's torch.autograd.Function.backward(...) wrapper.
class PythonOpGrad final : public OpKernel, public PythonOpGradBase {
 public:
  PythonOpGrad(const OpKernelInfo& info) : OpKernel(info), PythonOpGradBase(info){};
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
