// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/language_interop_ops/pyop/pyop_lib_proxy.h"
#include "core/torch_custom_function/torch_custom_function_register.h"

namespace onnxruntime {
namespace contrib {

// Pytorch's torch.autograd.Function.apply(...) wrapper.
class PythonOp final : public OpKernel {
 public:
  PythonOp(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("name", &name_));
    input_types_ = info.GetAttrsOrDefault("input_types", std::vector<int64_t>());
    output_types_ = info.GetAttrsOrDefault("output_types", std::vector<int64_t>());
  }

  Status Compute(OpKernelContext* context) const override;

  ~PythonOp() {
  }

 private:
  // Name of containing class. For example, MyReLU.
  std::string name_;
  // Input types of MyReLU.apply(...).
  std::vector<int64_t> input_types_;
  // Output types of MyReLU.apply(...).
  std::vector<int64_t> output_types_;
};

// Pytorch's torch.autograd.Function.backward(...) wrapper.
class PythonOpGrad final : public OpKernel {
 public:
  PythonOpGrad(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("name", &name_));
    input_types_ = info.GetAttrsOrDefault("input_types", std::vector<int64_t>());
    output_types_ = info.GetAttrsOrDefault("output_types", std::vector<int64_t>());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  // Name of containing class. For example, MyReLU.
  std::string name_;
  // Input types of MyReLU.apply(...).
  std::vector<int64_t> input_types_;
  // Output types of MyReLU.apply(...).
  std::vector<int64_t> output_types_;
};

}  // namespace contrib
}  // namespace onnxruntime
