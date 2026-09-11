// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_node_proto_helper.h"
#include "deform_conv_attributes.h"

namespace onnxruntime {

template <typename T>
class DeformConv : public OpKernel {
 public:
  explicit DeformConv(const OpKernelInfo& info) : OpKernel(info), attrs_(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  DeformConvAttributes attrs_;
};

}  // namespace onnxruntime
