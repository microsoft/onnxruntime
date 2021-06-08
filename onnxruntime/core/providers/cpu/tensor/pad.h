// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "padbase.h"

namespace onnxruntime {

struct Pad final : public OpKernel, public PadBase {
  explicit Pad(const OpKernelInfo& info) : OpKernel(info), PadBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
