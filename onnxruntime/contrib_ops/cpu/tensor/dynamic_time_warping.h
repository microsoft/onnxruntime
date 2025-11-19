// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include <core/common/safeint.h>

namespace onnxruntime {
namespace contrib {

using onnxruntime::OpKernelContext;
using onnxruntime::OpKernelInfo;

class DynamicTimeWarping : public OpKernel {
 public:
  DynamicTimeWarping(const OpKernelInfo& info) : OpKernel(info) {}

  ~DynamicTimeWarping() = default;

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
