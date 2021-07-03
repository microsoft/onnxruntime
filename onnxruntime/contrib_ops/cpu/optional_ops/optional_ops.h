// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class Optional final : public OpKernel {
 public:
  explicit Optional(const OpKernelInfo& info) : OpKernel(info) {
    // if (info.GetAttr<onnx::TypeProto>("type", &type_proto_).IsOK()) {
    //   type_available_ = true;
    // }
  }

  common::Status Compute(OpKernelContext* context) const override;

 private:
  onnx::TypeProto type_proto_;
  bool type_available_ = false;
};

class OptionalHasElement final : public OpKernel {
 public:
  explicit OptionalHasElement(const OpKernelInfo& info) : OpKernel(info) {}

  common::Status Compute(OpKernelContext* context) const override;
};

class OptionalGetElement final : public OpKernel {
 public:
  explicit OptionalGetElement(const OpKernelInfo& info) : OpKernel(info) {}

  common::Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
