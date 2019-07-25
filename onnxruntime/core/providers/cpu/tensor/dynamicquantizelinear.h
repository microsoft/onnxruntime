// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class DynamicQuantizeLinear final : public OpKernel {
 public:
  DynamicQuantizeLinear(const OpKernelInfo& info) : OpKernel(info) {
   ORT_ENFORCE(info.GetAttrOrDefault("to", int64_t(2)) == int64_t(2), "Only uint8 output data type supported.");
  }

  Status Compute(OpKernelContext* context) const override;

};
}  // namespace contrib
}  // namespace onnxruntime
