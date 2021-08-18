// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {

class Size final : public OpKernel {
 public:
  Size(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
