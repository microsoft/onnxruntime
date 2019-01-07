// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/conv_base.h"

namespace onnxruntime {

class ConvInteger : public OpKernel, public ConvBase {
 public:
  ConvInteger(const OpKernelInfo& info) : OpKernel(info), ConvBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
