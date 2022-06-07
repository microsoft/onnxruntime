// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License

#pragma once
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace acl {

template <typename T>
class Relu : public OpKernel {
 public:
  explicit Relu(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace acl
}  // namespace onnxruntime
